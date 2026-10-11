//! Physically reserved compressed backing, with independent receiver and decoder budgets.
use super::{HydrationOptions, HydrationReport, Phase, Progress, Tracker, make_removable};
use crate::{ArtifactTicket, Entry, Error, Result, extract, pack::set_mode, paths, verify};
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{Notify, Semaphore};

/// Reserve the entire compressed workload before opening any HTTP response.
/// Backing must support physical allocation and consumed-prefix hole punching.
#[derive(Debug, Clone)]
pub struct ReceiveBufferOptions {
    pub directory: PathBuf,
    pub max_reserved_bytes: u64,
    /// Other simultaneous allocations on the output filesystem, such as a model.
    pub additional_output_bytes: u64,
    /// Optional decoder budget independent of HTTP permits; useful across artifacts.
    pub extraction_limit: Option<Arc<Semaphore>>,
    pub observer: Option<crate::transport::TransferObserver>,
}
impl ReceiveBufferOptions {
    pub fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            max_reserved_bytes: 64 << 30,
            additional_output_bytes: 0,
            extraction_limit: None,
            observer: None,
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReceptionProgress {
    pub reserved_bytes: u64,
    #[serde(default)]
    pub reservation_seconds: f64,
    pub network_received_bytes: u64,
    pub storage_written_bytes: u64,
    pub compressed_consumed_bytes: u64,
    pub buffered_bytes: u64,
    pub buffer_high_water_bytes: u64,
    pub reclaimed_bytes: u64,
    pub decoded_bytes: u64,
    pub file_written_bytes: u64,
    /// Sum of waits across workers, not elapsed wall time.
    pub receiver_body_wait_seconds: f64,
    pub receiver_storage_write_seconds: f64,
    pub consumer_wait_seconds: f64,
    pub reception_complete_seconds: Option<f64>,
    pub extraction_complete_seconds: Option<f64>,
}
#[derive(Default)]
struct Metrics {
    received: AtomicU64,
    written: AtomicU64,
    consumed: AtomicU64,
    high_water: AtomicU64,
    reclaimed: AtomicU64,
    decoded: AtomicU64,
    files: AtomicU64,
    body_ns: AtomicU64,
    storage_ns: AtomicU64,
    consumer_ns: AtomicU64,
    reservation_ns: AtomicU64,
    received_done_ns: AtomicU64,
    extracted_done_ns: AtomicU64,
}
impl Metrics {
    fn snapshot(&self, reserved: u64) -> ReceptionProgress {
        let written = self.written.load(Ordering::Relaxed);
        let consumed = self.consumed.load(Ordering::Relaxed);
        ReceptionProgress {
            reserved_bytes: reserved,
            reservation_seconds: self.reservation_ns.load(Ordering::Relaxed) as f64 / 1e9,
            reception_complete_seconds: Some(self.received_done_ns.load(Ordering::Relaxed))
                .filter(|n| *n != 0)
                .map(|n| n as f64 / 1e9),
            extraction_complete_seconds: Some(self.extracted_done_ns.load(Ordering::Relaxed))
                .filter(|n| *n != 0)
                .map(|n| n as f64 / 1e9),
            network_received_bytes: self.received.load(Ordering::Relaxed),
            storage_written_bytes: written,
            compressed_consumed_bytes: consumed,
            buffered_bytes: written.saturating_sub(consumed),
            buffer_high_water_bytes: self.high_water.load(Ordering::Relaxed),
            reclaimed_bytes: self.reclaimed.load(Ordering::Relaxed),
            decoded_bytes: self.decoded.load(Ordering::Relaxed),
            file_written_bytes: self.files.load(Ordering::Relaxed),
            receiver_body_wait_seconds: self.body_ns.load(Ordering::Relaxed) as f64 / 1e9,
            receiver_storage_write_seconds: self.storage_ns.load(Ordering::Relaxed) as f64 / 1e9,
            consumer_wait_seconds: self.consumer_ns.load(Ordering::Relaxed) as f64 / 1e9,
        }
    }
    fn emit<F: Fn(Progress)>(&self, tracker: &Tracker<F>, reserved: u64) {
        (tracker.callback)(Progress {
            phase: Phase::Extracting,
            downloaded_bytes: tracker.downloaded.load(Ordering::Relaxed),
            verified_bytes: 0,
            network_verified_bytes: 0,
            total_bytes: tracker.total_bytes,
            verified_chunks: 0,
            total_chunks: tracker.total_chunks,
            unpacked_bytes: self.decoded.load(Ordering::Relaxed),
            total_unpacked_bytes: tracker.total_unpacked,
            reception: Some(self.snapshot(reserved)),
        });
    }
}
#[derive(Default)]
struct Prefix {
    state: Mutex<PrefixState>,
    wake: Condvar,
}
#[derive(Default)]
struct PrefixState {
    written: u64,
    eof: bool,
    failed: bool,
}
#[derive(Default)]
struct Control {
    cancelled: AtomicBool,
    wake: Notify,
    prefixes: Mutex<Vec<Arc<Prefix>>>,
}
impl Control {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.wake.notify_waiters();
        if let Ok(prefixes) = self.prefixes.lock() {
            for p in prefixes.iter() {
                // Taking the condition's mutex closes the check-to-sleep race.
                if let Ok(_state) = p.state.lock() {
                    p.wake.notify_all();
                }
            }
        }
    }
    async fn permit(&self, limit: Arc<Semaphore>) -> Result<tokio::sync::OwnedSemaphorePermit> {
        let wake = self.wake.notified();
        tokio::pin!(wake);
        wake.as_mut().enable();
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        tokio::select! { biased; _=&mut wake=>Err(Error::Cancelled), p=limit.acquire_owned()=>p.map_err(|_|Error::Invalid("closed reception budget".into())) }
    }
}
struct Cancel(Arc<Control>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
struct Workspace {
    staging: PathBuf,
    spool: PathBuf,
    control: Arc<Control>,
}
impl Drop for Workspace {
    fn drop(&mut self) {
        self.control.cancel();
        let _ = make_removable(&self.staging);
        let _ = fs::remove_dir_all(&self.staging);
        let _ = fs::remove_dir_all(&self.spool);
    }
}
struct Piece {
    file: File,
    size: u64,
    reserved: u64,
    prefix: Arc<Prefix>,
    owner: Arc<Workspace>,
}
#[cfg(target_os = "linux")]
fn allocate(file: &File, size: u64) -> Result<()> {
    let size =
        i64::try_from(size).map_err(|_| Error::Invalid("reception capacity overflow".into()))?;
    nix::fcntl::fallocate(file, nix::fcntl::FallocateFlags::empty(), 0, size)
        .map_err(|e| Error::Io(io::Error::from_raw_os_error(e as i32)))
}
#[cfg(not(target_os = "linux"))]
fn allocate(_: &File, _: u64) -> Result<()> {
    Err(Error::Invalid(
        "reserved reception requires Linux allocation".into(),
    ))
}
#[cfg(target_os = "linux")]
fn reclaim(file: &File, from: u64, length: u64) -> io::Result<()> {
    nix::fcntl::fallocate(
        file,
        nix::fcntl::FallocateFlags::FALLOC_FL_PUNCH_HOLE
            | nix::fcntl::FallocateFlags::FALLOC_FL_KEEP_SIZE,
        from as i64,
        length as i64,
    )
    .map_err(|e| io::Error::from_raw_os_error(e as i32))
}
#[cfg(not(target_os = "linux"))]
fn reclaim(_: &File, _: u64, _: u64) -> io::Result<()> {
    Err(io::Error::other("unsupported reserved reception"))
}
#[cfg(unix)]
fn available(path: &Path) -> Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let stat = nix::sys::statvfs::statvfs(path)
        .map_err(|e| Error::Io(io::Error::from_raw_os_error(e as i32)))?;
    Ok((
        stat.blocks_available().saturating_mul(stat.fragment_size()),
        fs::metadata(path)?.dev(),
    ))
}
#[cfg(not(unix))]
fn available(_: &Path) -> Result<(u64, u64)> {
    Err(Error::Invalid(
        "reserved reception requires Unix storage".into(),
    ))
}

fn reserve(
    owner: Arc<Workspace>,
    sizes: &[u64],
    output_bytes: u64,
    cfg: &ReceiveBufferOptions,
) -> Result<(Vec<Arc<Piece>>, u64)> {
    if owner.control.cancelled.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    paths::directory(&cfg.directory)?;
    paths::ancestors(&cfg.directory)?;
    let reserved = sizes
        .iter()
        .try_fold(0_u64, |sum, size| {
            sum.checked_add(size.checked_add(4095)? / 4096 * 4096)
        })
        .ok_or_else(|| Error::Invalid("reception capacity overflow".into()))?;
    if reserved > cfg.max_reserved_bytes {
        return Err(Error::Invalid(
            "reception capacity exceeds reserved budget".into(),
        ));
    }
    let output = output_bytes
        .checked_add(cfg.additional_output_bytes)
        .ok_or_else(|| Error::Invalid("output capacity overflow".into()))?;
    let (spool_free, spool_device) = available(&cfg.directory)?;
    let (output_free, output_device) = available(owner.staging.parent().unwrap())?;
    let needed = if spool_device == output_device {
        reserved
            .checked_add(output)
            .ok_or_else(|| Error::Invalid("storage capacity overflow".into()))?
    } else {
        reserved
    };
    if spool_free < needed || output_free < output {
        return Err(Error::Invalid(
            "insufficient reception and output capacity".into(),
        ));
    }
    fs::create_dir(&owner.spool)?;
    set_mode(&owner.spool, 0o700)?;
    let mut pieces = Vec::with_capacity(sizes.len());
    for (index, size) in sizes.iter().copied().enumerate() {
        if owner.control.cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let file = File::options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(owner.spool.join(index.to_string()))?;
        let physical = size.div_ceil(4096) * 4096;
        set_mode(&owner.spool.join(index.to_string()), 0o600)?;
        allocate(&file, physical)?;
        // Probe required reclamation support without discarding allocated payload capacity.
        allocate(&file, physical + 4096)?;
        reclaim(&file, physical, 4096)?;
        file.set_len(size)?;
        let prefix = Arc::new(Prefix::default());
        owner
            .control
            .prefixes
            .lock()
            .map_err(|_| Error::Worker)?
            .push(prefix.clone());
        pieces.push(Arc::new(Piece {
            file,
            size,
            reserved: physical,
            prefix,
            owner: owner.clone(),
        }));
    }
    Ok((pieces, reserved))
}
struct PrefixReader {
    piece: Arc<Piece>,
    metrics: Arc<Metrics>,
    offset: u64,
    reclaimed: u64,
}
impl Read for PrefixReader {
    fn read(&mut self, target: &mut [u8]) -> io::Result<usize> {
        use std::os::unix::fs::FileExt;
        if target.is_empty() {
            return Ok(0);
        }
        let mut state = self
            .piece
            .prefix
            .state
            .lock()
            .map_err(|_| io::Error::other("reception state poisoned"))?;
        while state.written == self.offset
            && !state.eof
            && !state.failed
            && !self.piece.owner.control.cancelled.load(Ordering::Relaxed)
        {
            let began = Instant::now();
            state = self
                .piece
                .prefix
                .wake
                .wait(state)
                .map_err(|_| io::Error::other("reception state poisoned"))?;
            self.metrics.consumer_ns.fetch_add(
                began.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
        if self.piece.owner.control.cancelled.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "reception cancelled",
            ));
        }
        if state.failed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "reception failed",
            ));
        }
        let count = (state.written - self.offset).min(target.len() as u64) as usize;
        if count == 0 {
            return Ok(0);
        }
        drop(state);
        self.piece
            .file
            .read_exact_at(&mut target[..count], self.offset)?;
        self.offset += count as u64;
        self.metrics
            .consumed
            .fetch_add(count as u64, Ordering::Relaxed);
        let through = self.offset / 4096 * 4096;
        if through - self.reclaimed >= 1 << 20 || self.offset == self.piece.size {
            let through = if self.offset == self.piece.size {
                self.piece.reserved
            } else {
                through
            };
            if through > self.reclaimed {
                reclaim(&self.piece.file, self.reclaimed, through - self.reclaimed)?;
                self.metrics
                    .reclaimed
                    .fetch_add(through - self.reclaimed, Ordering::Relaxed);
                self.reclaimed = through;
            }
        }
        Ok(count)
    }
}
#[allow(clippy::too_many_arguments)]
async fn receive<F: Fn(Progress) + Send + Sync + 'static>(
    client: reqwest::Client,
    url: String,
    object: String,
    piece: Arc<Piece>,
    options: &HydrationOptions,
    metrics: Arc<Metrics>,
    tracker: Arc<Tracker<F>>,
    reserved: u64,
) -> Result<()> {
    let trace = crate::transport::Trace::new(
        object,
        0,
        piece.size,
        options
            .receive_buffer
            .as_ref()
            .and_then(|o| o.observer.clone()),
    );
    trace.boundary(crate::transport::TransferBoundary::Headers);
    let control = &piece.owner.control;
    let wake = control.wake.notified();
    tokio::pin!(wake);
    wake.as_mut().enable();
    if control.cancelled.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    let began = Instant::now();
    let response = tokio::select! { biased; _=&mut wake=>return Err(Error::Cancelled), r=tokio::time::timeout(Duration::from_secs(options.stall_timeout_s),client.get(url).send())=>r.map_err(|_|{let mut f=crate::transport::TransportFailure::local("headers_deadline");f.timeout=true;trace.failure(f);Error::Transfer})?.map_err(|error|{trace.failure(crate::transport::TransportFailure::request("headers",&error));Error::Transfer})? };
    metrics.body_ns.fetch_add(
        began.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
    if response.status() != reqwest::StatusCode::OK {
        let mut failure = crate::transport::TransportFailure::local("status");
        failure.status = Some(response.status().as_u16());
        trace.failure(failure);
        return Err(Error::Http(response.status().as_u16()));
    }
    if response.content_length().is_some_and(|n| n != piece.size) {
        trace.failure(crate::transport::TransportFailure::local("content_length"));
        return Err(Error::Integrity("reception Content-Length".into()));
    }
    let mut body = response.bytes_stream();
    let mut seen = 0_u64;
    loop {
        let wake = control.wake.notified();
        tokio::pin!(wake);
        wake.as_mut().enable();
        if control.cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let began = Instant::now();
        let next = tokio::select! {biased; _=&mut wake=>return Err(Error::Cancelled), b=tokio::time::timeout(Duration::from_secs(options.stall_timeout_s),body.next())=>b.map_err(|_|{let mut f=crate::transport::TransportFailure::local("body_deadline");f.timeout=true;trace.failure(f);Error::Transfer})?};
        metrics.body_ns.fetch_add(
            began.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        let Some(bytes) = next else { break };
        let bytes = bytes.map_err(|error| {
            trace.failure(crate::transport::TransportFailure::request("body", &error));
            Error::Transfer
        })?;
        trace.body(bytes.len() as u64, began.elapsed().as_secs_f64());
        metrics
            .received
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        tracker
            .downloaded
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        let after = seen
            .checked_add(bytes.len() as u64)
            .filter(|n| *n <= piece.size)
            .ok_or_else(|| Error::Integrity("reception exceeds declared size".into()))?;
        let write_trace = trace.clone();
        let queued_at = Instant::now();
        let local = piece.clone();
        let measured = metrics.clone();
        let offset = seen;
        tokio::task::spawn_blocking(move || -> Result<()> {
            use std::os::unix::fs::FileExt;
            if local.owner.control.cancelled.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            let began = queued_at;
            local
                .file
                .write_all_at(&bytes, offset)
                .inspect_err(|error| {
                    write_trace.failure(crate::transport::TransportFailure::io(
                        "storage_write",
                        error,
                    ));
                })?;
            write_trace.write(bytes.len() as u64, began.elapsed().as_secs_f64());
            measured.storage_ns.fetch_add(
                began.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            let written = measured
                .written
                .fetch_add(bytes.len() as u64, Ordering::Relaxed)
                + bytes.len() as u64;
            measured.high_water.fetch_max(
                written.saturating_sub(measured.consumed.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
            let mut state = local.prefix.state.lock().map_err(|_| Error::Worker)?;
            state.written = after;
            local.prefix.wake.notify_all();
            Ok(())
        })
        .await
        .map_err(|_| Error::Worker)??;
        seen = after;
        metrics.emit(&tracker, reserved);
    }
    if seen != piece.size {
        trace.failure(crate::transport::TransportFailure::local("truncated_body"));
        return Err(Error::Integrity(
            "reception shorter than declared size".into(),
        ));
    }
    trace.boundary(crate::transport::TransferBoundary::Complete);
    let mut state = piece.prefix.state.lock().map_err(|_| Error::Worker)?;
    state.eof = true;
    piece.prefix.wake.notify_all();
    Ok(())
}

pub(super) async fn hydrate<F: Fn(Progress) + Send + Sync + 'static>(
    ticket: ArtifactTicket,
    destination: &Path,
    options: &HydrationOptions,
    tracker: Arc<Tracker<F>>,
    client: reqwest::Client,
) -> Result<HydrationReport> {
    let epoch = Instant::now();
    if options.retries != 0 {
        return Err(Error::Invalid(
            "progressive reserved reception requires zero retries".into(),
        ));
    }
    let cfg = options
        .receive_buffer
        .as_ref()
        .ok_or_else(|| Error::Invalid("missing reception reservation".into()))?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let control = Arc::new(Control::default());
    let owner = Arc::new(Workspace {
        staging: parent.join(format!(".artifact-{}.staging", paths::nonce()?)),
        spool: cfg.directory.join(format!(".receive-{}", paths::nonce()?)),
        control: control.clone(),
    });
    let _cancel = Cancel(control.clone());
    let sizes = ticket
        .manifest
        .chunks
        .iter()
        .map(|c| c.size)
        .collect::<Vec<_>>();
    let output = ticket.manifest.entries.iter().map(Entry::size).sum::<u64>();
    let reserving = owner.clone();
    let configuration = cfg.clone();
    let metrics = Arc::new(Metrics::default());
    metrics.emit(&tracker, 0);
    let reservation =
        tokio::task::spawn_blocking(move || reserve(reserving, &sizes, output, &configuration))
            .await
            .map_err(|_| Error::Worker)?;
    metrics.reservation_ns.store(
        epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
    let (pieces, reserved) = match reservation {
        Ok(r) => r,
        Err(error) => {
            metrics.emit(&tracker, 0);
            return Err(error);
        }
    };
    metrics.emit(&tracker, reserved);
    fs::create_dir(&owner.staging)?;
    set_mode(&owner.staging, 0o700)?;
    let manifest = Arc::new(ticket.manifest);
    let work = owner.clone();
    let expected = manifest.clone();
    tokio::task::spawn_blocking(move || {
        extract::prepare_entries(&work.staging, &expected, &work.control.cancelled, false)
    })
    .await
    .map_err(|_| Error::Worker)??;
    metrics.reservation_ns.store(
        epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
    metrics.emit(&tracker, reserved);
    let http_limit = options
        .download_limit
        .clone()
        .unwrap_or_else(|| Arc::new(Semaphore::new(options.download_concurrency)));
    let decode_limit = cfg
        .extraction_limit
        .clone()
        .unwrap_or_else(|| Arc::new(Semaphore::new(options.extraction_concurrency)));
    let reception = async {
        let mut work = stream::iter(
            manifest
                .chunks
                .iter()
                .cloned()
                .zip(pieces.iter().cloned())
                .collect::<Vec<_>>(),
        )
        .map(|(chunk, piece)| {
            let piece = piece.clone();
            let url = ticket.urls[&chunk.name].clone();
            let client = client.clone();
            let metrics = metrics.clone();
            let tracker = tracker.clone();
            let limit = http_limit.clone();
            async move {
                let _permit = piece.owner.control.permit(limit).await?;
                let result = receive(
                    client,
                    url,
                    chunk.name.clone(),
                    piece.clone(),
                    options,
                    metrics,
                    tracker,
                    reserved,
                )
                .await;
                if result.is_err()
                    && let Ok(mut state) = piece.prefix.state.lock()
                {
                    state.failed = true;
                    piece.prefix.wake.notify_all();
                }
                result
            }
        })
        .buffer_unordered(options.download_concurrency);
        while let Some(result) = work.next().await {
            result?;
        }
        metrics.received_done_ns.store(
            epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        metrics.emit(&tracker, reserved);
        Ok::<_, Error>(epoch.elapsed().as_secs_f64())
    };
    let extraction = async {
        let mut work = stream::iter(
            manifest
                .chunks
                .iter()
                .cloned()
                .zip(pieces.iter().cloned())
                .collect::<Vec<_>>(),
        )
        .map(|(chunk, piece)| {
            let piece = piece.clone();
            let chunk = chunk.clone();
            let manifest = manifest.clone();
            let metrics = metrics.clone();
            let tracker = tracker.clone();
            let limit = decode_limit.clone();
            async move {
                let permit = piece.owner.control.permit(limit).await?;
                tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    let reader = PrefixReader {
                        piece: piece.clone(),
                        metrics: metrics.clone(),
                        offset: 0,
                        reclaimed: 0,
                    };
                    extract::extract_stream_observed(
                        reader,
                        &piece.owner.staging,
                        &manifest,
                        &chunk,
                        &piece.owner.control.cancelled,
                        false,
                        |n| {
                            metrics.decoded.fetch_add(n, Ordering::Relaxed);
                            metrics.emit(&tracker, reserved);
                        },
                        |n| {
                            metrics.files.fetch_add(n, Ordering::Relaxed);
                        },
                    )
                })
                .await
                .map_err(|_| Error::Worker)?
            }
        })
        .buffer_unordered(options.extraction_concurrency);
        while let Some(result) = work.next().await {
            result?;
        }
        metrics.extracted_done_ns.store(
            epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        metrics.emit(&tracker, reserved);
        Ok::<_, Error>(epoch.elapsed().as_secs_f64())
    };
    let (received_at, extracted_at) = tokio::try_join!(reception, extraction)?;
    let work = owner.clone();
    let expected = manifest.clone();
    tokio::task::spawn_blocking(move || {
        extract::finish_entries(&work.staging, &expected, &work.control.cancelled)?;
        verify::write_trusted_receipt(&work.staging, &expected)
    })
    .await
    .map_err(|_| Error::Worker)??;
    paths::ancestors(parent)?;
    if fs::symlink_metadata(destination).is_ok() {
        return Err(Error::DestinationExists);
    }
    fs::rename(&owner.staging, destination)?;
    let mut progress = metrics.snapshot(reserved);

    progress.reception_complete_seconds = Some(received_at);
    progress.extraction_complete_seconds = Some(extracted_at);
    (tracker.callback)(Progress {
        phase: Phase::Ready,
        downloaded_bytes: tracker.downloaded.load(Ordering::Relaxed),
        verified_bytes: 0,
        network_verified_bytes: 0,
        total_bytes: tracker.total_bytes,
        verified_chunks: 0,
        total_chunks: tracker.total_chunks,
        unpacked_bytes: tracker.total_unpacked,
        total_unpacked_bytes: tracker.total_unpacked,
        reception: Some(progress.clone()),
    });
    Ok(HydrationReport {
        downloaded_bytes: tracker.downloaded.load(Ordering::Relaxed),
        verified_bytes: 0,
        network_verified_bytes: 0,
        cache_hits: 0,
        unpacked_bytes: tracker.total_unpacked,
        already_present: false,
        reception: Some(progress),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_reservation_does_not_create_backing_or_output() {
        let root = std::env::temp_dir().join(format!(
            "switchyard-cancelled-reserve-{}",
            paths::nonce().unwrap()
        ));
        let control = Arc::new(Control::default());
        control.cancel();
        let workspace = Arc::new(Workspace {
            staging: root.join("output"),
            spool: root.join("spool"),
            control,
        });
        let options = ReceiveBufferOptions::new(root.clone());
        assert!(matches!(
            reserve(workspace, &[4096; 24], 4096, &options),
            Err(Error::Cancelled)
        ));
        assert!(!root.exists());
    }
}
