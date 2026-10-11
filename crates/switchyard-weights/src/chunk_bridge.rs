//! Chunk delivery and completed-copy acknowledgements; no provider or CUDA APIs.
use crate::BridgeProgress;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Instant,
};
use switchyard_artifacts::{
    Error, Result,
    range_stream::{BufferLimits, RangeStream, ReadyRange},
    tensors::TensorManifest,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::UnixListener,
};
pub struct ChunkBridge {
    listener: UnixListener,
    path: PathBuf,
}
impl Drop for ChunkBridge {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
impl ChunkBridge {
    pub fn bind(path: &Path) -> Result<Self> {
        let listener = UnixListener::bind(path)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok(Self {
            listener,
            path: path.into(),
        })
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub async fn serve(
        &self,
        receiver: &RangeStream,
        manifest: &TensorManifest,
        limits: &BufferLimits,
        epoch: Instant,
        on_progress: impl Fn(BridgeProgress),
    ) -> Result<BridgeProgress> {
        let (connection, _) = self.listener.accept().await?;
        receiver.consumer_available();
        let (input, mut output) = connection.into_split();
        let mut input = BufReader::new(input);
        output
            .write_all(&serde_json::to_vec(
                &json!({"format":"switchyard-weight-chunks-v1","tensors":manifest,"limits":limits}),
            )?)
            .await?;
        output.write_all(b"\n").await?;
        let entries = manifest
            .tensors
            .iter()
            .flat_map(|t| t.ranges.iter().map(|r| r.entry))
            .collect::<BTreeSet<_>>();
        let mut active = BTreeMap::<usize, ReadyRange>::new();
        let mut pending = false;
        let mut ended = false;
        let mut line = Vec::new();
        let mut report = BridgeProgress::default();
        loop {
            let mut bounded_input = (&mut input).take((4096 - line.len()) as u64);
            tokio::select! {
                read=bounded_input.read_until(b'\n',&mut line)=> {
                    if read?==0 {receiver.cancel();return Err(Error::Cancelled)}
                    if !line.ends_with(b"\n") {return Err(Error::Invalid("chunk command bound".into()))}
                    let command:Value=serde_json::from_slice(&line)?;
                    line.clear();
                    match command["op"].as_str() {
                        Some("next")=>{if pending||ended||active.len()>=limits.transfer_concurrency {return Err(Error::Invalid("chunk request ownership limit".into()))}pending=true;},
                        Some("ack")=>{let id=command["id"].as_u64().ok_or_else(||Error::Invalid("chunk acknowledgement id".into()))? as usize;let chunk=active.remove(&id).ok_or_else(||Error::Invalid("unknown or duplicate chunk acknowledgement".into()))?;
                            if command["cuda_complete"]!=true {return Err(Error::Invalid("chunk consumed before CUDA completion".into()))}
                            report.chunks_consumed+=1;report.bytes_consumed+=chunk.spec.length;chunk.consumed();on_progress(report.clone());},
                        Some("finish")=>{if !ended||!active.is_empty()||pending {return Err(Error::Invalid("unfinished chunk delivery".into()))}let gpu:crate::bridge::GpuTransferReport=serde_json::from_value(command["gpu"].clone())?;
                        if gpu.bytes>report.bytes_delivered || gpu.copies>report.chunks_delivered || !gpu.staging_seconds.is_finite() || !gpu.dma_seconds.is_finite() || gpu.staging_seconds<0.0 || gpu.dma_seconds<0.0 {return Err(Error::Invalid("chunk GPU report values".into()))}
                        report.tensors_delivered=manifest.tensors.len();report.gpu=Some(gpu);report.assignment_complete_seconds=Some(epoch.elapsed().as_secs_f64());on_progress(report.clone());return Ok(report)},
                        _=>return Err(Error::Invalid("chunk delivery operation".into())),
                    }
                },
                chunk=receiver.next_matching(|s|entries.contains(&s.entry)),if pending=> {
                    pending=false;let chunk=chunk?;
                    if let Some(chunk)=chunk {let message=json!({"chunk":chunk.spec,"location":chunk.location(),"path":chunk.descriptor_path()});chunk.copying();report.bytes_delivered+=chunk.spec.length;report.chunks_delivered+=1;report.first_tensor_seconds.get_or_insert(epoch.elapsed().as_secs_f64());output.write_all(&serde_json::to_vec(&message)?).await?;output.write_all(b"\n").await?;active.insert(chunk.spec.id,chunk);on_progress(report.clone());}
                    else {ended=true;output.write_all(b"{\"done\":true}\n").await?;}
                }
            }
        }
    }
}
