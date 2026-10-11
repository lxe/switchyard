use crate::{
    ArtifactTicket, Chunk, Error, Result, extract,
    pack::{digest_file, set_mode},
    paths, verify,
};
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::io::AsyncWriteExt;

#[path = "buffered.rs"]
mod buffered;
#[path = "streaming.rs"]
mod streaming;
pub use buffered::{ReceiveBufferOptions, ReceptionProgress};

/// Choose complete verification or streaming installation of caller-trusted artifacts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HydrationMode {
    #[default]
    Verified,
    /// No content hashes, compressed cache, file fsyncs, or final tree scan.
    /// The caller must trust the pinned manifest and its transport/storage.
    TrustedStreaming,
}

#[derive(Debug, Clone)]
pub struct HydrationOptions {
    /// Verified by default; trusted streaming relies on caller-pinned artifacts.
    pub mode: HydrationMode,
    pub cache_dir: PathBuf,
    pub download_concurrency: usize,
    /// Optional shared transfer budget across concurrent hydration calls.
    /// A permit covers one HTTP attempt and its streaming decoder, not backoff.
    pub download_limit: Option<Arc<tokio::sync::Semaphore>>,
    pub extraction_concurrency: usize,
    /// Optional physically reserved spool for independent reception and extraction.
    pub receive_buffer: Option<ReceiveBufferOptions>,
    /// Additional attempts after a failed request; partial transfers resume when supported.
    pub retries: usize,
    /// Total deadline for each HTTP attempt, including its streamed response body.
    pub request_timeout_s: u64,
    /// Deadline between response body chunks and while waiting for initial response headers.
    pub stall_timeout_s: u64,
    pub max_compressed_bytes: u64,
    pub max_unpacked_bytes: u64,
}
impl Default for HydrationOptions {
    fn default() -> Self {
        Self {
            mode: HydrationMode::Verified,
            cache_dir: std::env::temp_dir().join("switchyard-artifacts-cache"),
            download_concurrency: 8,
            download_limit: None,
            extraction_concurrency: 4,
            receive_buffer: None,
            retries: 2,
            request_timeout_s: 120,
            stall_timeout_s: 30,
            max_compressed_bytes: 64 * 1024 * 1024 * 1024,
            max_unpacked_bytes: 256 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Downloading,
    Extracting,
    Verifying,
    Ready,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    pub phase: Phase,
    /// Response body bytes received, including rejected data and repeated retry bytes.
    /// HTTP headers and transport framing are excluded.
    pub downloaded_bytes: u64,
    /// Compressed bytes of all verified chunks, including cache hits; bounded by total_bytes.
    pub verified_bytes: u64,
    /// Verified chunk bytes obtained over the network this call, excluding cache hits and retry duplication.
    pub network_verified_bytes: u64,
    pub total_bytes: u64,
    pub verified_chunks: usize,
    pub total_chunks: usize,
    pub unpacked_bytes: u64,
    pub total_unpacked_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reception: Option<ReceptionProgress>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydrationReport {
    pub downloaded_bytes: u64,
    pub verified_bytes: u64,
    pub network_verified_bytes: u64,
    pub cache_hits: usize,
    pub unpacked_bytes: u64,
    pub already_present: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reception: Option<ReceptionProgress>,
}

struct Tracker<F> {
    callback: F,
    downloaded: AtomicU64,
    verified: AtomicUsize,
    verified_bytes: AtomicU64,
    network_verified: AtomicU64,
    total_bytes: u64,
    total_chunks: usize,
    total_unpacked: u64,
}
impl<F: Fn(Progress)> Tracker<F> {
    fn emit(&self, phase: Phase, unpacked_bytes: u64) {
        (self.callback)(Progress {
            phase,
            downloaded_bytes: self.downloaded.load(Ordering::Relaxed),
            verified_bytes: self.verified_bytes.load(Ordering::Relaxed),
            network_verified_bytes: self.network_verified.load(Ordering::Relaxed),
            total_bytes: self.total_bytes,
            verified_chunks: self.verified.load(Ordering::Relaxed),
            total_chunks: self.total_chunks,
            unpacked_bytes,
            total_unpacked_bytes: self.total_unpacked,
            reception: None,
        });
    }
}

/// Hydrate into a new directory with the configured verification policy.
/// Verified mode reuses checked chunks and completely revalidates matching trees;
/// trusted streaming installs HTTP bytes directly and trusts matching receipts.
/// Callbacks may run concurrently on extraction workers. Dropping the returned future
/// prevents publication, while verified immutable chunks remain reusable.
pub async fn hydrate<F>(
    ticket: ArtifactTicket,
    destination: &Path,
    options: HydrationOptions,
    on_progress: F,
) -> Result<HydrationReport>
where
    F: Fn(Progress) + Send + Sync + 'static,
{
    match options.mode {
        HydrationMode::Verified => ticket.manifest.validate()?,
        HydrationMode::TrustedStreaming => ticket.manifest.validate_structure()?,
    }
    let total_bytes = ticket.manifest.chunks.iter().map(|c| c.size).sum::<u64>();
    let total_unpacked = ticket
        .manifest
        .chunks
        .iter()
        .map(|c| c.unpacked_size)
        .sum::<u64>();
    if options.download_concurrency == 0
        || options.download_concurrency > 64
        || options.extraction_concurrency == 0
        || options.extraction_concurrency > 64
        || options.retries > 10
        || options.request_timeout_s == 0
        || options.request_timeout_s > 86400
        || options.stall_timeout_s == 0
        || options.stall_timeout_s > 86400
        || total_bytes > options.max_compressed_bytes
        || total_unpacked > options.max_unpacked_bytes
    {
        return Err(Error::Invalid("hydration limits".into()));
    }
    let tracker = Arc::new(Tracker {
        callback: on_progress,
        downloaded: AtomicU64::new(0),
        verified: AtomicUsize::new(0),
        verified_bytes: AtomicU64::new(0),
        network_verified: AtomicU64::new(0),
        total_bytes,
        total_chunks: ticket.manifest.chunks.len(),
        total_unpacked,
    });
    let destination = destination.to_owned();
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    paths::directory(parent)?;
    paths::ancestors(parent)?;
    if fs::symlink_metadata(&destination).is_ok() {
        let target = destination.clone();
        let manifest = ticket.manifest.clone();
        let mode = options.mode;
        let matching = tokio::task::spawn_blocking(move || {
            if mode == HydrationMode::Verified {
                verify::verify(&target, &manifest)?;
            } else {
                paths::ancestors(&target)?;
                if !fs::symlink_metadata(&target)?.is_dir() {
                    return Err(Error::DestinationExists);
                }
            }
            verify::check_receipt(&target, &manifest)
        })
        .await
        .map_err(|_| Error::Worker)?;
        matching.map_err(|_| Error::DestinationExists)?;
        let verified_bytes = if mode == HydrationMode::Verified {
            total_bytes
        } else {
            0
        };
        if mode == HydrationMode::Verified {
            tracker
                .verified
                .store(tracker.total_chunks, Ordering::Relaxed);
            tracker.verified_bytes.store(total_bytes, Ordering::Relaxed);
        }
        tracker.emit(Phase::Ready, total_unpacked);
        return Ok(HydrationReport {
            downloaded_bytes: 0,
            verified_bytes,
            network_verified_bytes: 0,
            cache_hits: if mode == HydrationMode::Verified {
                tracker.total_chunks
            } else {
                0
            },
            unpacked_bytes: total_unpacked,
            already_present: true,
            reception: None,
        });
    }
    for chunk in &ticket.manifest.chunks {
        let url = ticket
            .urls
            .get(&chunk.name)
            .ok_or_else(|| Error::Invalid("missing chunk URL".into()))?;
        let parsed =
            reqwest::Url::parse(url).map_err(|_| Error::Invalid("invalid chunk URL".into()))?;
        if !matches!(parsed.scheme(), "http" | "https")
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.fragment().is_some()
        {
            return Err(Error::Invalid("unsupported chunk URL".into()));
        }
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(options.request_timeout_s))
        .connect_timeout(Duration::from_secs(options.stall_timeout_s))
        .build()
        .map_err(|_| Error::Transfer)?;
    if options.mode == HydrationMode::TrustedStreaming {
        if options.receive_buffer.is_some() {
            return buffered::hydrate(ticket, &destination, &options, tracker, client).await;
        }
        return streaming::hydrate(ticket, &destination, &options, tracker, client).await;
    }
    paths::directory(&options.cache_dir)?;
    tracker.emit(Phase::Downloading, 0);
    let chunks = ticket
        .manifest
        .chunks
        .iter()
        .cloned()
        .map(|chunk| {
            let url = ticket.urls[&chunk.name].clone();
            (chunk, url)
        })
        .collect::<Vec<_>>();
    let mut downloads = stream::iter(chunks)
        .map(|(chunk, url)| {
            let tracker = tracker.clone();
            let client = client.clone();
            let options = options.clone();
            async move { download(&client, &chunk, &url, &options, tracker).await }
        })
        .buffer_unordered(options.download_concurrency);
    let mut cache_hits = 0;
    while let Some(result) = downloads.next().await {
        cache_hits += usize::from(result?);
    }
    drop(downloads);
    let staging = parent.join(format!(".artifact-{}.staging", paths::nonce()?));
    fs::create_dir(&staging)?;
    set_mode(&staging, 0o700)?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let staging_guard = TemporaryDirectory {
        path: staging.clone(),
        cancelled: cancelled.clone(),
    };
    let manifest = ticket.manifest;
    let cache = options.cache_dir;
    let working = staging.clone();
    let worker_tracker = tracker.clone();
    let extraction_concurrency = options.extraction_concurrency;
    tracker.emit(Phase::Extracting, 0);
    tokio::task::spawn_blocking(move || {
        extract::extract(
            &cache,
            &working,
            &manifest,
            extraction_concurrency,
            &cancelled,
            |bytes| worker_tracker.emit(Phase::Extracting, bytes),
        )?;
        verify::write_receipt(&working, &manifest)?;
        worker_tracker.emit(Phase::Verifying, worker_tracker.total_unpacked);
        verify::verify(&working, &manifest)
    })
    .await
    .map_err(|_| Error::Worker)??;
    paths::ancestors(parent)?;
    if fs::symlink_metadata(&destination).is_ok() {
        return Err(Error::DestinationExists);
    }
    fs::rename(&staging, &destination)?;
    drop(staging_guard);
    tracker.emit(Phase::Ready, total_unpacked);
    Ok(HydrationReport {
        downloaded_bytes: tracker.downloaded.load(Ordering::Relaxed),
        verified_bytes: tracker.verified_bytes.load(Ordering::Relaxed),
        network_verified_bytes: tracker.network_verified.load(Ordering::Relaxed),
        cache_hits,
        unpacked_bytes: total_unpacked,
        already_present: false,
        reception: None,
    })
}

async fn download<F: Fn(Progress) + Send + Sync + 'static>(
    client: &reqwest::Client,
    chunk: &Chunk,
    url: &str,
    options: &HydrationOptions,
    tracker: Arc<Tracker<F>>,
) -> Result<bool> {
    let final_path = options.cache_dir.join(&chunk.name);
    if let Ok(meta) = fs::symlink_metadata(&final_path) {
        if !meta.is_file() {
            return Err(Error::Invalid("cached chunk is not a regular file".into()));
        }
        if check_chunk(final_path.clone(), chunk.clone()).await.is_ok() {
            tracker.verified.fetch_add(1, Ordering::Relaxed);
            tracker
                .verified_bytes
                .fetch_add(chunk.size, Ordering::Relaxed);
            tracker.emit(Phase::Downloading, 0);
            return Ok(true);
        }
        fs::remove_file(&final_path)?;
    }
    let temporary =
        options
            .cache_dir
            .join(format!(".{}.{}.partial", chunk.sha256, paths::nonce()?));
    let temporary_guard = TemporaryFile(temporary.clone());
    let mut file = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .await?;
    set_mode(&temporary, 0o600)?;
    let mut failure = Error::Transfer;
    for attempt in 0..=options.retries {
        let offset = file.metadata().await?.len();
        if offset > chunk.size {
            file.set_len(0).await?;
        }
        let offset = file.metadata().await?.len();
        if offset < chunk.size {
            let _permit = match &options.download_limit {
                Some(limit) => Some(
                    limit
                        .clone()
                        .acquire_owned()
                        .await
                        .map_err(|_| Error::Invalid("download limiter closed".into()))?,
                ),
                None => None,
            };
            let mut request = client.get(url);
            if offset > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
            }
            match tokio::time::timeout(Duration::from_secs(options.stall_timeout_s), request.send())
                .await
            {
                Ok(Ok(response)) => {
                    let status = response.status();
                    if !status.is_success() {
                        failure = Error::Http(status.as_u16());
                    } else {
                        let base = if offset > 0 && status == reqwest::StatusCode::OK {
                            file.set_len(0).await?;
                            use tokio::io::AsyncSeekExt;
                            file.seek(std::io::SeekFrom::Start(0)).await?;
                            0
                        } else {
                            offset
                        };
                        if (base > 0 && !valid_range(&response, base, chunk.size))
                            || (base == 0 && status != reqwest::StatusCode::OK)
                        {
                            failure = Error::Integrity("invalid partial response".into());
                        } else {
                            match receive(
                                response,
                                &mut file,
                                base,
                                chunk.size,
                                options.stall_timeout_s,
                                &tracker,
                            )
                            .await
                            {
                                Ok(()) => {}
                                Err(error) => {
                                    if matches!(error, Error::Integrity(_)) {
                                        file.set_len(0).await?;
                                        use tokio::io::AsyncSeekExt;
                                        file.seek(std::io::SeekFrom::Start(0)).await?;
                                    }
                                    failure = error;
                                }
                            }
                        }
                    }
                }
                _ => failure = Error::Transfer,
            }
        }
        file.flush().await?;
        if file.metadata().await?.len() == chunk.size {
            if check_chunk(temporary.clone(), chunk.clone()).await.is_ok() {
                file.sync_all().await?;
                match fs::hard_link(&temporary, &final_path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        check_chunk(final_path.clone(), chunk.clone()).await?;
                    }
                    Err(error) => return Err(error.into()),
                }
                drop(file);
                drop(temporary_guard);
                tracker.verified.fetch_add(1, Ordering::Relaxed);
                tracker
                    .verified_bytes
                    .fetch_add(chunk.size, Ordering::Relaxed);
                tracker
                    .network_verified
                    .fetch_add(chunk.size, Ordering::Relaxed);
                tracker.emit(Phase::Downloading, 0);
                return Ok(false);
            }
            failure = Error::Integrity("downloaded chunk digest".into());
            file.set_len(0).await?;
            use tokio::io::AsyncSeekExt;
            file.seek(std::io::SeekFrom::Start(0)).await?;
        }
        if attempt < options.retries {
            tokio::time::sleep(Duration::from_millis(100 * (attempt as u64 + 1))).await;
        }
    }
    Err(failure)
}

async fn receive<F: Fn(Progress)>(
    response: reqwest::Response,
    file: &mut tokio::fs::File,
    mut written: u64,
    size: u64,
    stall_timeout_s: u64,
    tracker: &Tracker<F>,
) -> Result<()> {
    let mut body = response.bytes_stream();
    while let Some(bytes) = tokio::time::timeout(Duration::from_secs(stall_timeout_s), body.next())
        .await
        .map_err(|_| Error::Transfer)?
    {
        let bytes = bytes.map_err(|_| Error::Transfer)?;
        tracker
            .downloaded
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        tracker.emit(Phase::Downloading, 0);
        written = written
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::Integrity("download size overflow".into()))?;
        if written > size {
            return Err(Error::Integrity("download exceeds declared size".into()));
        }
        file.write_all(&bytes).await?;
    }
    if written != size {
        return Err(Error::Transfer);
    }
    Ok(())
}

fn valid_range(response: &reqwest::Response, offset: u64, size: u64) -> bool {
    response.status() == reqwest::StatusCode::PARTIAL_CONTENT
        && response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            == Some(format!("bytes {offset}-{}/{size}", size - 1).as_str())
}

async fn check_chunk(path: PathBuf, chunk: Chunk) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_file() || metadata.len() != chunk.size {
            return Err(Error::Integrity("chunk size or type".into()));
        }
        if digest_file(&path)? != (chunk.sha256, chunk.size) {
            return Err(Error::Integrity("chunk digest".into()));
        }
        Ok(())
    })
    .await
    .map_err(|_| Error::Worker)?
}

struct TemporaryFile(PathBuf);
impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
struct TemporaryDirectory {
    path: PathBuf,
    cancelled: Arc<AtomicBool>,
}
impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        // Artifact directories may deliberately be read-only. Restore owner access only
        // within this private staging tree so cancellation can remove its contents.
        let _ = make_removable(&self.path);
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(crate) fn make_removable(path: &Path) -> Result<()> {
    if fs::symlink_metadata(path)?.is_dir() {
        set_mode(path, 0o700)?;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                make_removable(&entry.path())?;
            }
        }
    }
    Ok(())
}
