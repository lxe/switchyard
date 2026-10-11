//! Streaming legacy TAR pieces with bounded channels and exclusively owned publication.
use super::diagnostics::{Boundary, ModelDiagnostics, ShardTrace, TransferTrace};
use super::{ExactReader, extract_observed, sample_rate};
use crate::{Chunk, Error, Phase, Result, State, model::LegacyTicket};
use futures_util::{FutureExt, StreamExt, TryStreamExt, future::BoxFuture};
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
pub(super) fn cached(ticket: &LegacyTicket<'_>, root: &Path) -> Result<bool> {
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
    ticket: &LegacyTicket<'_>,
    root: &Path,
    state: Arc<RwLock<State>>,
    limit: Arc<Semaphore>,
) -> Result<()> {
    let allocation = state.read().await.model_allocation.clone();
    let allocated_files = ticket
        .manifest
        .files
        .iter()
        .map(|(path, info)| (PathBuf::from(path), info.size))
        .collect::<Vec<_>>();
    if cached(ticket, root)? {
        let published = root.join("model");
        allocation
            .register(move || crate::allocation::allocated_files(&published, &allocated_files));
        state.write().await.model_cached = true;
        return Ok(());
    }
    let workspace = Arc::new(Workspace::new(root)?);
    let stage = workspace.path.join("model");
    fs::create_dir(&stage)?;
    let owned_workspace = Arc::downgrade(&workspace);
    let allocation_path = Arc::new(Mutex::new((stage.clone(), false)));
    let observed_path = allocation_path.clone();
    allocation.register(move || {
        let owner = owned_workspace.upgrade();
        let path = observed_path.lock().unwrap();
        if owner.is_none() && !path.1 {
            return Ok(0);
        }
        crate::allocation::allocated_files(&path.0, &allocated_files)
    });
    let cancellation = Cancellation(Arc::new(AtomicBool::new(false)));
    let seen = Arc::new(Mutex::new(BTreeSet::new()));
    let client = client(ticket)?;
    let mut shards: BTreeMap<usize, Vec<Chunk>> = BTreeMap::new();
    for chunk in &ticket.manifest.chunks {
        shards.entry(chunk.shard).or_default().push(chunk.clone());
    }
    for chunks in shards.values_mut() {
        chunks.sort_by_key(|chunk| chunk.seq);
    }
    let started = Instant::now();
    let diagnostics = ModelDiagnostics::new(ticket.manifest, &limit, started, CHANNEL_PIECES);
    {
        let mut st = state.write().await;
        st.model_diagnostics = Some(diagnostics.clone());
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
            let diagnostics = diagnostics.clone();
            async move {
                let shard_trace = diagnostics.shard(chunks[0].shard);
                let mut jobs = Vec::new();
                let mut readers: Vec<Box<dyn Read + Send>> = Vec::new();
                for chunk in chunks {
                    let trace = diagnostics.transfer(&chunk.name);
                    let (sender, receiver) = mpsc::channel(CHANNEL_PIECES);
                    let (consumed, completion) = oneshot::channel();
                    let permit = Arc::new(Mutex::new(None));
                    let reader = ChannelReader {
                        receiver,
                        pending: io::Cursor::new(Vec::new()),
                        cancelled: cancelled.clone(),
                        trace: trace.clone(),
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
                        trace,
                        shard_trace: shard_trace.clone(),
                        seq: chunk.seq,
                    }));
                    jobs.push(TransferJob {
                        url: ticket.urls[&chunk.name].clone(),
                        chunk,
                        sender,
                        completion,
                        permit,
                    });
                }
                let manifest = ticket.manifest.clone();
                let extraction = tokio::task::spawn_blocking(move || {
                    let _ownership = ownership;
                    let reader = PieceReaders {
                        readers: readers.into_iter(),
                        current: None,
                    };
                    let mut archive = tar::Archive::new(reader);
                    shard_trace.boundary(Boundary::Archive);
                    let result = extract_observed(
                        &manifest,
                        &mut archive,
                        &stage,
                        &seen,
                        Some(&shard_trace),
                    );
                    if result.is_err() {
                        shard_trace.boundary(Boundary::ExtractionError);
                    }
                    result?;
                    io::copy(&mut archive.into_inner(), &mut io::sink())?;
                    shard_trace.boundary(Boundary::Complete);
                    Ok::<_, Error>(())
                });
                let stall = Duration::from_secs(ticket.stall_s);
                let transfers = futures_util::stream::iter(jobs)
                    .then(move |job| {
                        scheduled_transfer(
                            job,
                            client.clone(),
                            state.clone(),
                            limit.clone(),
                            diagnostics.clone(),
                            stall,
                        )
                    })
                    .boxed()
                    .try_buffer_unordered(ticket.concurrency)
                    .try_collect::<Vec<_>>();
                tokio::try_join!(transfers, async {
                    extraction.await.map_err(|_| Error::Task)?
                })?;
                Ok::<_, Error>(())
            }
            .boxed()
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
    {
        let mut allocation = allocation_path.lock().unwrap();
        fs::rename(stage, &dest)?;
        *allocation = (dest, true);
    }
    state.write().await.hydration_seconds = started.elapsed().as_secs_f64();
    Ok(())
}

type Transfer = BoxFuture<'static, Result<()>>;
struct TransferJob {
    url: String,
    chunk: Chunk,
    sender: mpsc::Sender<Vec<u8>>,
    completion: oneshot::Receiver<()>,
    permit: Arc<Mutex<Option<OwnedSemaphorePermit>>>,
}
fn scheduled_transfer(
    job: TransferJob,
    client: reqwest::Client,
    state: Arc<RwLock<State>>,
    limit: Arc<Semaphore>,
    diagnostics: ModelDiagnostics,
    stall: Duration,
) -> BoxFuture<'static, Result<Transfer>> {
    async move {
        // Acquire in TAR sequence before launching concurrently. A later piece
        // cannot hold a slot while its required predecessor waits.
        let trace = diagnostics.transfer(&job.chunk.name);
        trace.boundary(Boundary::PermitWait);
        trace.update(|t, now| t.permit_wait_started_s = Some(now));
        let acquired = limit.acquire_owned().await.map_err(|_| Error::Task)?;
        trace.update(|t, now| t.permit_acquired_s = Some(now));
        *job.permit.lock().map_err(|_| Error::Task)? = Some(acquired);
        Ok(async move {
            transfer(
                &client,
                &job.url,
                &job.chunk,
                job.sender,
                &state,
                &diagnostics,
                stall,
            )
            .await?;
            // Keep the slot and permit until the decoder consumes this piece. Even
            // small completed chunks therefore cannot accumulate a whole model.
            trace.boundary(Boundary::DecoderCompletion);
            job.completion
                .await
                .map_err(|_| Error::Transfer("model extraction stopped".into()))?;
            trace.boundary(Boundary::Complete);
            Ok(())
        }
        .boxed())
    }
    .boxed()
}

fn client(ticket: &LegacyTicket<'_>) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent("Switchyard/0.1")
        .connect_timeout(Duration::from_secs(15))
        // Reqwest's read timer runs during consumer backpressure. Only time an
        // actively awaited network operation below; keep the total request bound.
        .timeout(Duration::from_secs(ticket.lifetime_s.clamp(1, 600)))
        .build()
        .map_err(|_| Error::Transfer("stream client initialization".into()))
}

async fn transfer(
    client: &reqwest::Client,
    url: &str,
    chunk: &Chunk,
    sender: mpsc::Sender<Vec<u8>>,
    state: &RwLock<State>,
    diagnostics: &ModelDiagnostics,
    stall: Duration,
) -> Result<()> {
    let trace = diagnostics.transfer(&chunk.name);
    let result = transfer_body(client, url, chunk, &sender, state, diagnostics, stall).await;
    if result.is_err() {
        let class = trace
            .error_class()
            .unwrap_or_else(|| "transfer_error".into());
        diagnostics.failure(&chunk.name, &class);
        trace.boundary(Boundary::Failed);
    }
    result
}

async fn transfer_body(
    client: &reqwest::Client,
    url: &str,
    chunk: &Chunk,
    sender: &mpsc::Sender<Vec<u8>>,
    state: &RwLock<State>,
    diagnostics: &ModelDiagnostics,
    stall: Duration,
) -> Result<()> {
    let trace = diagnostics.transfer(&chunk.name);
    trace.boundary(Boundary::HeadersWait);
    trace.update(|t, now| t.request_s = Some(now));
    let response = tokio::time::timeout(stall, client.get(url).send())
        .await
        .map_err(|_| {
            trace.error("headers_wait_deadline");
            Error::Transfer(format!("{} headers stalled", chunk.name))
        })?
        .map_err(|error| {
            let class = trace.request_error(&error);
            diagnostics.failure(&chunk.name, class);
            Error::Transfer(format!("{} transport", chunk.name))
        })?;
    trace.update(|t, now| {
        t.headers_s = Some(now);
        t.http_status = Some(response.status().as_u16());
        t.http_version = Some(format!("{:?}", response.version()));
        t.declared_length = response.content_length();
        t.remote_socket = response.remote_addr().map(|a| a.to_string());
    });
    if !response.status().is_success() {
        trace.error("http_status");
        return Err(Error::Transfer(format!(
            "{} HTTP {}",
            chunk.name,
            response.status().as_u16()
        )));
    }
    let mut stream = response.bytes_stream();
    let mut total = 0_u64;
    loop {
        trace.boundary(Boundary::BodyWait);
        let wait = Instant::now();
        let next = tokio::time::timeout(stall, stream.next()).await;
        let waited = wait.elapsed().as_secs_f64();
        trace.update(|t, _| {
            t.body_wait_s += waited;
            t.longest_body_wait_s = t.longest_body_wait_s.max(waited);
        });
        let next = match next {
            Ok(next) => next,
            Err(_) => {
                trace.error("body_wait_deadline");
                // Sample while the response stream still owns its connection.
                diagnostics.failure(&chunk.name, "body_wait_deadline");
                return Err(Error::Transfer(format!("{} stalled", chunk.name)));
            }
        };
        let Some(piece) = next else {
            trace.update(|t, now| t.body_eof_s = Some(now));
            break;
        };
        let piece = piece.map_err(|error| {
            let class = trace.request_error(&error);
            // Freeze typed transport causes and completed host/socket evidence while
            // this response and the outer transfer's sender are still owned.
            diagnostics.failure(&chunk.name, class);
            Error::Transfer(format!("{} interrupted", chunk.name))
        })?;
        trace.update(|t, now| {
            t.first_body_s.get_or_insert(now);
            t.last_body_s = Some(now);
            t.received_bytes += piece.len() as u64;
        });
        total = total.checked_add(piece.len() as u64).ok_or_else(|| {
            trace.error("size_overflow");
            Error::Transfer("response size overflow".into())
        })?;
        if total > chunk.size {
            trace.error("size_exceeded");
            return Err(Error::Transfer(format!("{} size exceeded", chunk.name)));
        }
        trace.boundary(Boundary::ProgressState);
        {
            let mut st = state.write().await;
            st.bytes_downloaded += piece.len() as u64;
            sample_rate(&mut st, trace.origin());
        }
        for bytes in piece.chunks(PIECE_BYTES) {
            trace.boundary(Boundary::ChannelSend);
            trace.update(|t, _| t.channel_available_slots = sender.capacity());
            let wait = Instant::now();
            let result = sender.send(bytes.to_vec()).await;
            trace.update(|t, _| {
                t.channel_wait_s += wait.elapsed().as_secs_f64();
                t.channel_available_slots = sender.capacity();
                if result.is_ok() {
                    t.delivered_bytes += bytes.len() as u64;
                }
            });
            result.map_err(|_| {
                trace.error("consumer_closed");
                Error::Transfer("model extraction stopped".into())
            })?;
        }
    }
    if total != chunk.size {
        trace.error("size_mismatch");
        return Err(Error::Transfer(format!("{} size mismatch", chunk.name)));
    }
    trace.boundary(Boundary::BodyComplete);
    // The outer transfer drops its sender after success or complete failure capture.
    // Partial streaming attempts fail rather than replaying TAR.
    Ok(())
}

struct ChannelReader {
    receiver: mpsc::Receiver<Vec<u8>>,
    pending: io::Cursor<Vec<u8>>,
    cancelled: Arc<AtomicBool>,
    trace: TransferTrace,
}
impl Read for ChannelReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            if self.cancelled.load(Ordering::Relaxed) {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "model hydration cancelled",
                ));
            }
            let size = self.pending.read(buffer)?;
            self.trace.update(|t, _| t.consumed_bytes += size as u64);
            if size != 0 {
                return Ok(size);
            }
            self.trace.consumer(Boundary::ChannelReceive);
            let next = self.receiver.blocking_recv();
            self.trace.consumer(Boundary::Decoding);
            self.trace
                .update(|t, _| t.channel_available_slots = self.receiver.capacity());
            match next {
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
    trace: TransferTrace,
    shard_trace: ShardTrace,
    seq: usize,
}
impl Read for DecodedPiece {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.shard_trace.update(|s, now| {
            s.active_seq = Some(self.seq);
            s.boundary = Boundary::PieceRead;
            s.boundary_since_s = now;
        });
        self.trace.consumer(Boundary::Decoding);
        let result = self.inner.read(buffer);
        self.trace.consumer(if result.is_ok() {
            Boundary::Archive
        } else {
            Boundary::ReadError
        });
        let size = result?;
        self.trace.update(|t, _| t.decoded_bytes += size as u64);
        self.shard_trace.update(|s, now| {
            s.decoded_bytes += size as u64;
            s.boundary = Boundary::Archive;
            s.boundary_since_s = now;
        });
        if size == 0 && !buffer.is_empty() {
            self.permit
                .lock()
                .map_err(|_| io::Error::other("model transfer permit lock"))?
                .take();
            self.trace.update(|t, now| {
                t.permit_released_s = Some(now);
                t.consumed_s = Some(now);
            });
            self.trace.consumer(Boundary::Complete);
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
mod tests;
