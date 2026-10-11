//! Trusted startup: HTTP bytes feed bounded blocking zstd/tar workers directly.
use super::{HydrationOptions, HydrationReport, Phase, Tracker, make_removable};
use crate::{
    ArtifactTicket, Chunk, Entry, Error, Manifest, Result, extract, pack::set_mode, paths, verify,
};
use futures_util::{StreamExt, stream};
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::Notify;
use tokio_util::io::{StreamReader, SyncIoBridge};

#[derive(Default)]
struct Control {
    cancelled: AtomicBool,
    wake: Notify,
}
impl Control {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.wake.notify_waiters();
    }
}
struct CancelOnDrop(Arc<Control>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
// Blocking workers retain ownership after an async future is dropped. Cleanup
// waits for the last owner, so a worker cannot recreate files after removal.
struct Staging {
    path: PathBuf,
    control: Arc<Control>,
}
impl Drop for Staging {
    fn drop(&mut self) {
        self.control.cancel();
        let _ = make_removable(&self.path);
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(super) async fn hydrate<F>(
    ticket: ArtifactTicket,
    destination: &Path,
    options: &HydrationOptions,
    tracker: Arc<Tracker<F>>,
    client: reqwest::Client,
) -> Result<HydrationReport>
where
    F: Fn(super::Progress) + Send + Sync + 'static,
{
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let control = Arc::new(Control::default());
    let staging = Arc::new(Staging {
        path: parent.join(format!(".artifact-{}.staging", paths::nonce()?)),
        control: control.clone(),
    });
    // Cancellation guard is separate from cleanup ownership and is dropped first.
    let _cancel = CancelOnDrop(control.clone());
    fs::create_dir(&staging.path)?;
    set_mode(&staging.path, 0o700)?;
    let manifest = Arc::new(ticket.manifest);
    let working = staging.clone();
    let prepared = manifest.clone();
    tokio::task::spawn_blocking(move || {
        extract::prepare_entries(&working.path, &prepared, &working.control.cancelled, false)
    })
    .await
    .map_err(|_| Error::Worker)??;
    let unpacked = Arc::new(AtomicU64::new(0));
    tracker.emit(Phase::Extracting, 0);
    let mut operations = stream::iter(manifest.chunks.iter().cloned())
        .map(|chunk| {
            let url = ticket.urls[&chunk.name].clone();
            let manifest = manifest.clone();
            let staging = staging.clone();
            let tracker = tracker.clone();
            let unpacked = unpacked.clone();
            let client = client.clone();
            async move {
                stream_chunk(
                    &client, chunk, url, manifest, staging, options, tracker, unpacked,
                )
                .await
            }
        })
        .buffer_unordered(
            options
                .download_concurrency
                .min(options.extraction_concurrency),
        );
    while let Some(result) = operations.next().await {
        result?;
    }
    drop(operations);
    let working = staging.clone();
    let completed = manifest.clone();
    tokio::task::spawn_blocking(move || {
        extract::finish_entries(&working.path, &completed, &working.control.cancelled)?;
        verify::write_trusted_receipt(&working.path, &completed)
    })
    .await
    .map_err(|_| Error::Worker)??;
    paths::ancestors(parent)?;
    if fs::symlink_metadata(destination).is_ok() {
        return Err(Error::DestinationExists);
    }
    // No publication occurs inside detached blocking tasks.
    fs::rename(&staging.path, destination)?;
    tracker.emit(Phase::Ready, tracker.total_unpacked);
    Ok(HydrationReport {
        downloaded_bytes: tracker.downloaded.load(Ordering::Relaxed),
        verified_bytes: 0,
        network_verified_bytes: 0,
        cache_hits: 0,
        unpacked_bytes: tracker.total_unpacked,
        already_present: false,
        reception: None,
    })
}

#[allow(clippy::too_many_arguments)]
async fn stream_chunk<F>(
    client: &reqwest::Client,
    chunk: Chunk,
    url: String,
    manifest: Arc<Manifest>,
    staging: Arc<Staging>,
    options: &HydrationOptions,
    tracker: Arc<Tracker<F>>,
    unpacked: Arc<AtomicU64>,
) -> Result<()>
where
    F: Fn(super::Progress) + Send + Sync + 'static,
{
    let mut failure = Error::Transfer;
    for attempt in 0..=options.retries {
        if staging.control.cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let mut permit = match &options.download_limit {
            Some(limit) => Some(
                limit
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| Error::Invalid("download limiter closed".into()))?,
            ),
            None => None,
        };
        let response = tokio::time::timeout(
            Duration::from_secs(options.stall_timeout_s),
            client.get(&url).send(),
        )
        .await;
        let response = match response {
            Ok(Ok(response)) if response.status() == reqwest::StatusCode::OK => Some(response),
            Ok(Ok(response)) => {
                failure = Error::Http(response.status().as_u16());
                None
            }
            _ => None,
        };
        if let Some(response) = response {
            let progress = Arc::new(AtomicU64::new(0));
            let control = staging.control.clone();
            let received_tracker = tracker.clone();
            let received_unpacked = unpacked.clone();
            let size = chunk.size;
            let stall = Duration::from_secs(options.stall_timeout_s);
            let body = stream::try_unfold(
                (response.bytes_stream(), 0_u64),
                move |(mut body, seen)| {
                    let control = control.clone();
                    let tracker = received_tracker.clone();
                    let unpacked = received_unpacked.clone();
                    async move {
                        let notified = control.wake.notified();
                        tokio::pin!(notified);
                        // Register before inspecting the flag, closing the notification race.
                        notified.as_mut().enable();
                        if control.cancelled.load(Ordering::Relaxed) {
                            return Err(io::Error::new(
                                io::ErrorKind::Interrupted,
                                "hydration cancelled",
                            ));
                        }
                        let part = tokio::select! {
                            biased;
                            _ = notified => return Err(io::Error::new(io::ErrorKind::Interrupted, "hydration cancelled")),
                            part = tokio::time::timeout(stall, body.next()) => part.map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "artifact body stalled"))?,
                        };
                        match part {
                            Some(part) => {
                                let bytes = part.map_err(|_| {
                                    io::Error::new(
                                        io::ErrorKind::UnexpectedEof,
                                        "artifact body interrupted",
                                    )
                                })?;
                                let count = bytes.len() as u64;
                                let before = tracker.downloaded.fetch_add(count, Ordering::Relaxed);
                                if before >> 20 != (before + count) >> 20 {
                                    tracker
                                        .emit(Phase::Extracting, unpacked.load(Ordering::Relaxed));
                                }
                                let next = seen
                                    .checked_add(count)
                                    .filter(|next| *next <= size)
                                    .ok_or_else(|| {
                                        io::Error::new(
                                            io::ErrorKind::InvalidData,
                                            "compressed chunk exceeds declared size",
                                        )
                                    })?;
                                Ok(Some((bytes, (body, next))))
                            }
                            None if seen == size => Ok(None),
                            None => Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "compressed chunk is shorter than declared size",
                            )),
                        }
                    }
                },
            );
            // Capture the runtime handle here; only the synchronous decoder runs
            // on the bounded blocking worker. StreamReader retains one body part.
            let bridge = SyncIoBridge::new(StreamReader::new(Box::pin(body)));
            let working = staging.clone();
            let expected = manifest.clone();
            let shard = chunk.clone();
            let local = progress.clone();
            let total = unpacked.clone();
            let extracted_tracker = tracker.clone();
            let decoder_permit = permit.take();
            let result = tokio::task::spawn_blocking(move || {
                // The worker retains its permit even if its async owner is dropped.
                let _permit = decoder_permit;
                extract::extract_stream(
                    bridge,
                    &working.path,
                    &expected,
                    &shard,
                    &working.control.cancelled,
                    false,
                    |bytes| {
                        local.fetch_add(bytes, Ordering::Relaxed);
                        let before = total.fetch_add(bytes, Ordering::Relaxed);
                        if before >> 20 != (before + bytes) >> 20 {
                            extracted_tracker.emit(Phase::Extracting, before + bytes);
                        }
                    },
                )
            })
            .await
            .map_err(|_| Error::Worker)?;
            match result {
                Ok(()) => {
                    tracker.emit(Phase::Extracting, unpacked.load(Ordering::Relaxed));
                    return Ok(());
                }
                Err(error) => {
                    unpacked.fetch_sub(progress.load(Ordering::Relaxed), Ordering::Relaxed);
                    failure = error;
                }
            }
        }
        // A decoder cannot resume at an arbitrary HTTP offset. After its worker
        // has stopped, remove only this disjoint shard's files and restart it.
        for index in &chunk.entries {
            if let Entry::File { path, .. } = &manifest.entries[*index] {
                match fs::remove_file(staging.path.join(path)) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
        }
        drop(permit);
        if attempt < options.retries {
            tokio::time::sleep(Duration::from_millis(100 * (attempt as u64 + 1))).await;
        }
    }
    Err(failure)
}
