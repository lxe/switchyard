//! Single-consumer Unix descriptor bridge, owned independently of the engine process.
use crate::WeightStream;
use std::{
    collections::BTreeMap,
    fs,
    os::unix::{fs::PermissionsExt, io::AsRawFd},
    path::{Path, PathBuf},
    time::Instant,
};
use switchyard_artifacts::{Error, Result, extents::ExtentHydration};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::UnixListener,
};

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BridgeProgress {
    pub tensors_delivered: usize,
    pub bytes_delivered: u64,
    pub first_tensor_seconds: Option<f64>,
    pub assignment_complete_seconds: Option<f64>,
    pub gpu: Option<GpuTransferReport>,
}
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuTransferReport {
    pub copies: usize,
    pub bytes: u64,
    pub first_copy_finished_seconds: Option<f64>,
    pub last_copy_finished_seconds: Option<f64>,
    pub staging_seconds: f64,
    pub dma_seconds: f64,
}
pub struct WeightBridge {
    listener: UnixListener,
    path: PathBuf,
}
impl Drop for WeightBridge {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
impl WeightBridge {
    pub fn bind(path: &Path) -> Result<Self> {
        let listener = UnixListener::bind(path)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok(Self {
            listener,
            path: path.to_owned(),
        })
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub async fn serve(
        &self,
        hydration: &ExtentHydration,
        epoch: Instant,
        on_connected: impl FnOnce(),
        on_progress: impl Fn(BridgeProgress),
    ) -> Result<BridgeProgress> {
        let (connection, _) = self.listener.accept().await?;
        on_connected();
        let (input, mut output) = connection.into_split();
        let mut input = BufReader::new(input);
        let mut stream = WeightStream::new(hydration, 100_000_000)?;
        let mut files = BTreeMap::new();
        let mut report = BridgeProgress::default();
        loop {
            let mut command = [0_u8; 5];
            input.read_exact(&mut command).await?;
            if &command != b"next\n" {
                return Err(Error::Invalid("weights bridge command".into()));
            }
            let mut unexpected = [0_u8; 1];
            let ready = tokio::select! {
                ready=stream.next_ready(hydration)=>ready?,
                result=input.read(&mut unexpected)=>{if result?==0{return Err(Error::Cancelled);}return Err(Error::Invalid("pipelined weights bridge command".into()));}
            };
            let Some(ready) = ready else {
                output.write_all(b"{\"done\":true}\n").await?;
                let mut command = [0_u8; 7];
                input.read_exact(&mut command).await?;
                if &command == b"report\n" {
                    let mut line = String::new();
                    (&mut input).take(4096).read_line(&mut line).await?;
                    if !line.ends_with('\n') {
                        return Err(Error::Invalid("GPU report bound".into()));
                    }
                    let gpu: GpuTransferReport = serde_json::from_str(&line)?;
                    if gpu.copies > report.tensors_delivered
                        || gpu.bytes > report.bytes_delivered
                        || !gpu.staging_seconds.is_finite()
                        || !gpu.dma_seconds.is_finite()
                        || gpu.staging_seconds < 0.0
                        || gpu.dma_seconds < 0.0
                        || [
                            gpu.first_copy_finished_seconds,
                            gpu.last_copy_finished_seconds,
                        ]
                        .into_iter()
                        .flatten()
                        .any(|t| {
                            !t.is_finite() || t < 0.0 || t > epoch.elapsed().as_secs_f64() + 1.0
                        })
                    {
                        return Err(Error::Invalid("GPU report values".into()));
                    }
                    report.gpu = Some(gpu);
                } else if &command != b"finish\n" {
                    return Err(Error::Cancelled);
                }

                report.assignment_complete_seconds = Some(epoch.elapsed().as_secs_f64());
                on_progress(report.clone());
                return Ok(report);
            };
            let file = match files.entry(ready.tensor.entry) {
                std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::btree_map::Entry::Vacant(e) => e.insert(ready.bytes.open_file()?),
            };
            let message = serde_json::json!({"tensor":ready.tensor,"path":format!("/proc/{}/fd/{}",std::process::id(),file.as_raw_fd()),"ready_seconds":epoch.elapsed().as_secs_f64()});
            output.write_all(&serde_json::to_vec(&message)?).await?;
            output.write_all(b"\n").await?;
            report.tensors_delivered += 1;
            report.bytes_delivered += ready.tensor.length;
            report
                .first_tensor_seconds
                .get_or_insert(epoch.elapsed().as_secs_f64());
            on_progress(report.clone());
        }
    }
}
