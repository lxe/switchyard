//! Streaming legacy TAR pieces with bounded channels and exclusively owned publication.
use super::{ExactReader, extract_reader, sample_rate};
use crate::{Chunk, Error, Phase, Result, State, Ticket};
use futures_util::{StreamExt, TryStreamExt};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{OwnedSemaphorePermit, RwLock, Semaphore, mpsc, oneshot};

const PIECE_BYTES: usize = 64 << 10;
const CHANNEL_PIECES: usize = 4;
/// Headroom for twenty channels, decoders, HTTP buffering, and concurrent runtime extraction.
pub(super) fn buffer_reserve(concurrency: usize) -> u64 {
    // Both model and generic runtime zstd decoders cap their window at128MiB. The
    // shared permit covers decoding, so at most concurrency histories can coexist.
    (136 << 20) * concurrency as u64 + (64 << 20)
}
const RECEIPT: &str = ".switchyard-archive";

pub(super) struct Workspace {
    pub(super) path: PathBuf,
}
impl Workspace {
    pub(super) fn new(root: &Path) -> Result<Self> {
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random).map_err(|_| Error::Task)?;
        let suffix = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let path = root.join(format!(".model-hydration-{suffix}"));
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new().mode(0o700).create(&path)?;
        Ok(Self { path })
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
struct Cancellation(Arc<AtomicBool>);
impl Drop for Cancellation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
fn real_parents(path: &Path) -> Result<()> {
    for parent in path.ancestors() {
        if !fs::symlink_metadata(parent)?.is_dir() {
            return Err(Error::Manifest(
                "model cache contains an unsafe parent".into(),
            ));
        }
    }
    Ok(())
}
pub(super) fn cached(ticket: &Ticket, root: &Path) -> Result<bool> {
    real_parents(root)?;
    let dest = root.join("model");
    match fs::symlink_metadata(&dest) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
        Ok(meta) if !meta.is_dir() => {
            return Err(Error::Manifest(
                "model cache is not a real directory".into(),
            ));
        }
        Ok(_) => {}
    }
    let receipt = dest.join(RECEIPT);
    use std::os::unix::fs::PermissionsExt;
    let metadata = fs::symlink_metadata(&receipt)?;
    if !metadata.is_file()
        || metadata.len() != 64
        || metadata.permissions().mode() & 0o777 != 0o600
        || fs::read_to_string(&receipt)? != ticket.manifest.sha256
    {
        return Err(Error::Manifest(
            "model cache has a different or missing trusted identity".into(),
        ));
    }
    Ok(true)
}

pub(super) async fn hydrate(
    ticket: &Ticket,
    root: &Path,
    state: Arc<RwLock<State>>,
    limit: Arc<Semaphore>,
) -> Result<()> {
    if cached(ticket, root)? {
        state.write().await.model_cached = true;
        return Ok(());
    }
    let workspace = Arc::new(Workspace::new(root)?);
    let stage = workspace.path.join("model");
    fs::create_dir(&stage)?;
    let cancellation = Cancellation(Arc::new(AtomicBool::new(false)));
    let seen = Arc::new(Mutex::new(BTreeSet::new()));
    let client = reqwest::Client::builder()
        .user_agent("Switchyard/0.1")
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(ticket.stall_s))
        .timeout(Duration::from_secs(ticket.lifetime_s.clamp(1, 600)))
        .build()
        .map_err(|_| Error::Transfer("stream client initialization".into()))?;
    let mut shards: BTreeMap<usize, Vec<Chunk>> = BTreeMap::new();
    for chunk in &ticket.manifest.chunks {
        shards.entry(chunk.shard).or_default().push(chunk.clone());
    }
    for chunks in shards.values_mut() {
        chunks.sort_by_key(|chunk| chunk.seq);
    }
    let started = Instant::now();
    {
        let mut st = state.write().await;
        st.phase = Phase::Hydrating;
        st.bytes_downloaded = 0;
        st.model_cached = false;
        st.volume_bytes_per_s = None;
    }
    futures_util::stream::iter(shards.into_values())
        .map(|chunks| {
            let ownership = workspace.clone();
            let seen = seen.clone();
            let cancelled = cancellation.0.clone();
            let client = client.clone();
            let state = state.clone();
            let limit = limit.clone();
            let stage = stage.clone();
            async move {
                let mut jobs = Vec::new();
                let mut readers: Vec<Box<dyn Read + Send>> = Vec::new();
                for chunk in chunks {
                    let (sender, receiver) = mpsc::channel(CHANNEL_PIECES);
                    let (consumed, completion) = oneshot::channel();
                    let permit = Arc::new(Mutex::new(None));
                    let reader = ChannelReader {
                        receiver,
                        pending: io::Cursor::new(Vec::new()),
                        cancelled: cancelled.clone(),
                    };
                    let decoded: Box<dyn Read + Send> = if ticket.manifest.compress == "zstd" {
                        // Decoder construction reads its header; do that on the blocking worker.
                        Box::new(LazyDecoder {
                            source: Some(reader),
                            decoded: None,
                        })
                    } else {
                        Box::new(reader)
                    };
                    readers.push(Box::new(DecodedPiece {
                        inner: ExactReader {
                            inner: decoded,
                            remaining: chunk.raw,
                            checked_end: false,
                        },
                        consumed: Some(consumed),
                        permit: permit.clone(),
                    }));
                    jobs.push((chunk, sender, completion, permit));
                }
                let manifest = ticket.manifest.clone();
                let extraction = tokio::task::spawn_blocking(move || {
                    let _ownership = ownership;
                    let reader = PieceReaders {
                        readers: readers.into_iter(),
                        current: None,
                    };
                    let mut archive = tar::Archive::new(reader);
                    extract_reader(&manifest, &mut archive, &stage, &seen)?;
                    io::copy(&mut archive.into_inner(), &mut io::sink())?;
                    Ok::<_, Error>(())
                });
                let transfers = futures_util::stream::iter(jobs)
                    .map(|(chunk, sender, completion, permit)| {
                        let client = client.clone();
                        let limit = limit.clone();
                        let state = state.clone();
                        async move {
                            *permit.lock().map_err(|_| Error::Task)? =
                                Some(limit.acquire_owned().await.map_err(|_| Error::Task)?);
                            transfer(
                                &client,
                                &ticket.urls[&chunk.name],
                                &chunk,
                                sender,
                                &state,
                                started,
                            )
                            .await?;
                            // Keep the slot and permit until the decoder consumes this piece. Even
                            // small completed chunks therefore cannot accumulate a whole model.
                            completion
                                .await
                                .map_err(|_| Error::Transfer("model extraction stopped".into()))?;
                            Ok::<_, Error>(())
                        }
                    })
                    .buffer_unordered(ticket.concurrency)
                    .try_collect::<Vec<_>>();
                tokio::try_join!(transfers, async {
                    extraction.await.map_err(|_| Error::Task)?
                })?;
                Ok::<_, Error>(())
            }
        })
        .buffer_unordered(ticket.concurrency.min(20))
        .try_collect::<Vec<_>>()
        .await?;
    if seen.lock().map_err(|_| Error::Task)?.len() != ticket.manifest.files.len() {
        return Err(Error::Manifest("archive is missing expected files".into()));
    }
    use std::os::unix::fs::OpenOptionsExt;
    let mut receipt = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(stage.join(RECEIPT))?;
    receipt.write_all(ticket.manifest.sha256.as_bytes())?;
    drop(receipt);
    real_parents(root)?;
    let dest = root.join("model");
    if fs::symlink_metadata(&dest).is_ok() {
        return Err(Error::Manifest(
            "model destination appeared during hydration".into(),
        ));
    }
    // Only this awaiting owner publishes; workers have no destination path or rename authority.
    fs::rename(stage, dest)?;
    state.write().await.hydration_seconds = started.elapsed().as_secs_f64();
    Ok(())
}

async fn transfer(
    client: &reqwest::Client,
    url: &str,
    chunk: &Chunk,
    sender: mpsc::Sender<Vec<u8>>,
    state: &RwLock<State>,
    started: Instant,
) -> Result<()> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|_| Error::Transfer(format!("{} transport", chunk.name)))?;
    if !response.status().is_success() {
        return Err(Error::Transfer(format!(
            "{} HTTP {}",
            chunk.name,
            response.status().as_u16()
        )));
    }
    let mut stream = response.bytes_stream();
    let mut total = 0_u64;
    while let Some(piece) = stream.next().await {
        let piece = piece.map_err(|_| Error::Transfer(format!("{} interrupted", chunk.name)))?;
        total = total
            .checked_add(piece.len() as u64)
            .ok_or_else(|| Error::Transfer("response size overflow".into()))?;
        if total > chunk.size {
            return Err(Error::Transfer(format!("{} size exceeded", chunk.name)));
        }
        {
            let mut st = state.write().await;
            st.bytes_downloaded += piece.len() as u64;
            sample_rate(&mut st, started);
        }
        for bytes in piece.chunks(PIECE_BYTES) {
            sender
                .send(bytes.to_vec())
                .await
                .map_err(|_| Error::Transfer("model extraction stopped".into()))?;
        }
    }
    if total != chunk.size {
        return Err(Error::Transfer(format!("{} size mismatch", chunk.name)));
    }
    // Dropping sender delivers EOF. Partial streaming attempts fail rather than replaying TAR.
    Ok(())
}

struct ChannelReader {
    receiver: mpsc::Receiver<Vec<u8>>,
    pending: io::Cursor<Vec<u8>>,
    cancelled: Arc<AtomicBool>,
}
impl Read for ChannelReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            if self.cancelled.load(Ordering::Relaxed) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "model hydration cancelled",
                ));
            }
            let size = self.pending.read(buffer)?;
            if size != 0 {
                return Ok(size);
            }
            match self.receiver.blocking_recv() {
                Some(piece) => self.pending = io::Cursor::new(piece),
                None => return Ok(0),
            }
        }
    }
}
struct LazyDecoder {
    source: Option<ChannelReader>,
    decoded: Option<zstd::stream::read::Decoder<'static, io::BufReader<ChannelReader>>>,
}
impl Read for LazyDecoder {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.decoded.is_none() {
            let mut decoder = zstd::stream::read::Decoder::new(self.source.take().unwrap())?;
            decoder.window_log_max(27)?;
            self.decoded = Some(decoder);
        }
        self.decoded.as_mut().unwrap().read(buffer)
    }
}
struct DecodedPiece {
    inner: ExactReader<Box<dyn Read + Send>>,
    consumed: Option<oneshot::Sender<()>>,
    // The blocking reader owns the permit even if its async downloader is cancelled.
    permit: Arc<Mutex<Option<OwnedSemaphorePermit>>>,
}
impl Read for DecodedPiece {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let size = self.inner.read(buffer)?;
        if size == 0 && !buffer.is_empty() {
            self.permit
                .lock()
                .map_err(|_| io::Error::other("model transfer permit lock"))?
                .take();
            if let Some(consumed) = self.consumed.take() {
                let _ = consumed.send(());
            }
        }
        Ok(size)
    }
}
struct PieceReaders {
    readers: std::vec::IntoIter<Box<dyn Read + Send>>,
    current: Option<Box<dyn Read + Send>>,
}
impl Read for PieceReaders {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            if let Some(reader) = &mut self.current {
                let size = reader.read(buffer)?;
                if size != 0 {
                    return Ok(size);
                }
            }
            self.current = self.readers.next();
            if self.current.is_none() {
                return Ok(0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn cancelled_downloader_keeps_permit_with_decoding_worker() {
        struct Paused(std::sync::mpsc::Receiver<()>);
        impl Read for Paused {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                self.0.recv().unwrap();
                Ok(0)
            }
        }
        let limit = Arc::new(Semaphore::new(1));
        let permit = Arc::new(Mutex::new(Some(
            limit.clone().acquire_owned().await.unwrap(),
        )));
        let (release, waiting) = std::sync::mpsc::channel();
        let (started, ready) = oneshot::channel();
        let (consumed, _completion) = oneshot::channel();
        let mut reader = DecodedPiece {
            inner: ExactReader {
                inner: Box::new(Paused(waiting)),
                remaining: 0,
                checked_end: false,
            },
            consumed: Some(consumed),
            permit: permit.clone(),
        };
        let decoding = tokio::task::spawn_blocking(move || {
            let _ = started.send(());
            assert_eq!(reader.read(&mut [0]).unwrap(), 0);
        });
        ready.await.unwrap();
        // Drop the async downloader's ownership while the decoder is still blocked.
        drop(permit);
        decoding.abort();
        assert_eq!(limit.available_permits(), 0);
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while limit.available_permits() != 1 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }
}
