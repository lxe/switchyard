use super::{ExtentManifest, ExtentTicket, manifest::RECEIPT};
use crate::{Entry, Error, HydrationMode, Result, extract, pack::set_mode, paths};
use futures_util::{StreamExt, stream};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::{Notify, Semaphore};

#[path = "transport.rs"]
mod transport;

#[derive(Debug, Clone)]
pub struct ExtentOptions {
    pub mode: HydrationMode,
    pub concurrency: usize,
    pub download_limit: Option<Arc<Semaphore>>,
    /// Fixed copy buffer per active worker, independent of extent/file size.
    pub buffer_bytes: usize,
    /// Complete request/body deadline. There are no retries or stall recovery.
    pub request_timeout: Duration,
    pub max_bytes: u64,
    /// Caller-selected extents to receive first, followed by round-robin files.
    pub priority_extents: Vec<usize>,
    /// Explicitly trust a matching canonical-content receipt in trusted streaming mode.
    pub reuse_existing: bool,
}
impl Default for ExtentOptions {
    fn default() -> Self {
        Self {
            mode: HydrationMode::Verified,
            concurrency: 20,
            download_limit: None,
            buffer_bytes: 64 * 1024,
            request_timeout: Duration::from_secs(120),
            max_bytes: 256 * 1024 * 1024 * 1024,
            priority_extents: vec![],
            reuse_existing: false,
        }
    }
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExtentReport {
    pub downloaded_bytes: u64,
    pub completed_extents: usize,
    pub file_bytes: u64,
    /// Copy buffers only; excludes HTTP/TLS buffers, page cache and consumer staging.
    pub copy_buffer_bound: usize,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExtentProgress {
    pub received_bytes: u64,
    pub completed_bytes: u64,
    pub completed_extents: usize,
}

struct State {
    done: Vec<bool>,
    failed: bool,
    published: bool,
    root: PathBuf,
}
struct Shared {
    manifest: ExtentManifest,
    state: Mutex<State>,
    wake: Notify,
    cancelled: AtomicBool,
    received: AtomicU64,
    started: AtomicBool,
    staging: PathBuf,
}
impl Shared {
    async fn wait_cancelled(&self) {
        loop {
            let notified = self.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.cancelled.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.state.lock().unwrap().failed = true;
        self.wake.notify_waiters();
    }
    fn ready(&self, entry: usize, offset: u64, length: u64) -> bool {
        let end = offset + length; // checked by the public caller
        let state = self.state.lock().unwrap();
        self.manifest
            .extents
            .iter()
            .enumerate()
            .filter(|(_, e)| e.entry == entry && e.offset < end && e.offset + e.length > offset)
            .all(|(i, _)| state.done[i])
    }
}
impl Drop for Shared {
    fn drop(&mut self) {
        // Blocking workers and leases retain this owner until their last access.
        // Published directories are not removed; only this private staging name is owned.
        let _ = super::super::hydrate::make_removable(&self.staging);
        let _ = fs::remove_dir_all(&self.staging);
    }
}
struct CancelRun(Arc<Shared>);
impl Drop for CancelRun {
    fn drop(&mut self) {
        if !self.0.state.lock().unwrap().published {
            self.0.cancel();
        }
    }
}

/// A range becomes observable only after all its disjoint extents finish writing.
/// Retains the staging owner. Consumers must only map/read the declared byte interval.
#[derive(Clone)]
pub struct CompletedRange {
    shared: Arc<Shared>,
    pub entry: usize,
    pub offset: u64,
    pub length: u64,
}
impl CompletedRange {
    /// Open the backing inode under the publication lock. The caller must restrict
    /// reads/views to this lease's completed interval even though the fd names a file.
    pub fn open_file(&self) -> Result<File> {
        let state = self.shared.state.lock().unwrap();
        Ok(File::open(
            state
                .root
                .join(self.shared.manifest.entries[self.entry].path()),
        )?)
    }
    /// Current path, including after atomic publication. Opening/map creation must
    /// occur under caller-controlled publication ordering if another process is used.
    pub fn path(&self) -> PathBuf {
        self.shared
            .state
            .lock()
            .unwrap()
            .root
            .join(self.shared.manifest.entries[self.entry].path())
    }
    /// Read only completed bytes; the file is opened while holding publication's path lock.
    pub fn read_into(&self, relative_offset: u64, buffer: &mut [u8]) -> Result<()> {
        if relative_offset
            .checked_add(buffer.len() as u64)
            .is_none_or(|end| end > self.length)
        {
            return Err(Error::Invalid("read outside completed range".into()));
        }
        let mut file = self.open_file()?;
        file.seek(SeekFrom::Start(self.offset + relative_offset))?;
        file.read_exact(buffer)?;
        Ok(())
    }
}

/// Prepare private pre-sized regular files, then run independent offset writes.
/// A separate serving consumer may await completed ranges during `run`.
/// Source, destination and ancestors require exclusive caller ownership.
pub struct ExtentHydration {
    shared: Arc<Shared>,
    ticket: ExtentTicket,
    options: ExtentOptions,
    destination: PathBuf,
}
impl ExtentHydration {
    pub fn progress(&self) -> ExtentProgress {
        let state = self.shared.state.lock().unwrap();
        ExtentProgress {
            received_bytes: self.shared.received.load(Ordering::Relaxed),
            completed_bytes: self
                .manifest()
                .extents
                .iter()
                .zip(&state.done)
                .filter(|(_, done)| **done)
                .map(|(e, _)| e.length)
                .sum(),
            completed_extents: state.done.iter().filter(|done| **done).count(),
        }
    }
    pub fn prepare(
        ticket: ExtentTicket,
        destination: &Path,
        options: ExtentOptions,
    ) -> Result<Self> {
        if options.mode == HydrationMode::Verified {
            ticket.manifest.validate()?;
        } else {
            ticket.manifest.validate_structure()?;
        }
        let priorities = options
            .priority_extents
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        if priorities.len() != options.priority_extents.len()
            || priorities
                .iter()
                .any(|i| *i >= ticket.manifest.extents.len())
        {
            return Err(Error::Invalid("invalid extent priority list".into()));
        }
        if !(1..=128).contains(&options.concurrency)
            || !(4096..=1024 * 1024).contains(&options.buffer_bytes)
            || options.request_timeout.is_zero()
            || ticket.manifest.total_bytes()? > options.max_bytes
        {
            return Err(Error::Invalid("extent hydration options".into()));
        }
        for e in &ticket.manifest.extents {
            let url = ticket
                .urls
                .get(&e.object)
                .ok_or_else(|| Error::Invalid("missing extent URL".into()))?;
            let parsed =
                reqwest::Url::parse(url).map_err(|_| Error::Invalid("extent URL".into()))?;
            if !matches!(parsed.scheme(), "http" | "https")
                || !parsed.username().is_empty()
                || parsed.password().is_some()
                || parsed.fragment().is_some()
            {
                return Err(Error::Invalid("unsupported extent URL".into()));
            }
        }
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        paths::directory(parent)?;
        if let Ok(metadata) = fs::symlink_metadata(destination) {
            if !options.reuse_existing
                || options.mode != HydrationMode::TrustedStreaming
                || !metadata.is_dir()
            {
                return Err(Error::DestinationExists);
            }
            paths::ancestors(destination)?;
            let mut receipt = String::new();
            File::open(destination.join(".switchyard-extents-id"))?
                .take(256)
                .read_to_string(&mut receipt)?;
            if receipt.lines().nth(1) != Some(ticket.manifest.content_id.as_str())
                || receipt.lines().count() != 2
            {
                return Err(Error::DestinationExists);
            }
            let staging = parent.join(format!(".extents-{}.unused", paths::nonce()?));
            let shared = Arc::new(Shared {
                manifest: ticket.manifest.clone(),
                state: Mutex::new(State {
                    done: vec![true; ticket.manifest.extents.len()],
                    failed: false,
                    published: true,
                    root: destination.to_owned(),
                }),
                wake: Notify::new(),
                cancelled: AtomicBool::new(false),
                received: AtomicU64::new(0),
                started: AtomicBool::new(false),
                staging,
            });
            return Ok(Self {
                shared,
                ticket,
                options,
                destination: destination.to_owned(),
            });
        }
        let staging = parent.join(format!(".extents-{}.staging", paths::nonce()?));
        fs::create_dir(&staging)?;
        let shared = Arc::new(Shared {
            manifest: ticket.manifest.clone(),
            state: Mutex::new(State {
                done: vec![false; ticket.manifest.extents.len()],
                failed: false,
                published: false,
                root: staging.clone(),
            }),
            wake: Notify::new(),
            cancelled: AtomicBool::new(false),
            received: AtomicU64::new(0),
            started: AtomicBool::new(false),
            staging,
        });
        set_mode(&shared.staging, 0o700)?;
        extract::prepare_entries(
            &shared.staging,
            &ticket.manifest.filesystem_manifest(),
            &shared.cancelled,
            false,
        )?;
        for entry in &ticket.manifest.entries {
            if let Entry::File { path, size, .. } = entry {
                let file = File::options()
                    .write(true)
                    .create_new(true)
                    .open(shared.staging.join(path))?;
                set_mode(&shared.staging.join(path), 0o600)?;
                file.set_len(*size)?;
            }
        }
        Ok(Self {
            shared,
            ticket,
            options,
            destination: destination.to_owned(),
        })
    }
    pub fn manifest(&self) -> &ExtentManifest {
        &self.shared.manifest
    }
    fn check_range(&self, entry: usize, offset: u64, length: u64) -> Result<()> {
        match self.manifest().entries.get(entry) {
            Some(Entry::File { size, .. })
                if offset.checked_add(length).is_some_and(|end| end <= *size) =>
            {
                Ok(())
            }
            _ => Err(Error::Invalid("completion range boundary".into())),
        }
    }
    pub fn is_complete(&self, entry: usize, offset: u64, length: u64) -> Result<bool> {
        self.check_range(entry, offset, length)?;
        if self.shared.state.lock().unwrap().failed {
            return Err(Error::Cancelled);
        }
        Ok(self.shared.ready(entry, offset, length))
    }
    pub async fn wait_range(
        &self,
        entry: usize,
        offset: u64,
        length: u64,
    ) -> Result<CompletedRange> {
        self.check_range(entry, offset, length)?;
        loop {
            // Register before inspecting state, preventing lost completion/cancellation wakes.
            let notified = self.shared.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_complete(entry, offset, length)? {
                return Ok(CompletedRange {
                    shared: self.shared.clone(),
                    entry,
                    offset,
                    length,
                });
            }
            notified.await;
        }
    }
    /// Await any completed range without a lost wake between inspection and waiting.
    pub async fn wait_any(&self, ranges: &[(usize, u64, u64)]) -> Result<usize> {
        if ranges.is_empty() {
            return Err(Error::Invalid("empty completion selection".into()));
        }
        for &(entry, offset, length) in ranges {
            self.check_range(entry, offset, length)?;
        }
        loop {
            let notified = self.shared.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            for (i, &(entry, offset, length)) in ranges.iter().enumerate() {
                if self.is_complete(entry, offset, length)? {
                    return Ok(i);
                }
            }
            notified.await;
        }
    }
    pub fn cancel(&self) {
        self.shared.cancel();
    }
    pub async fn run(&self) -> Result<ExtentReport> {
        if self.shared.started.swap(true, Ordering::AcqRel) {
            return Err(Error::Invalid("extent hydration already started".into()));
        }
        if self.shared.state.lock().unwrap().published {
            return Ok(ExtentReport {
                downloaded_bytes: 0,
                completed_extents: self.manifest().extents.len(),
                file_bytes: self.manifest().total_bytes()?,
                copy_buffer_bound: 0,
            });
        }
        let _cancel = CancelRun(self.shared.clone());
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(self.options.request_timeout)
            .build()
            .map_err(|_| Error::Transfer)?;
        // Round-robin files: the first extent of every file starts before later
        // extents of a large file. Generic transport knows nothing about headers.
        let mut schedule = Vec::with_capacity(self.manifest().extents.len());
        let mut per_file = std::collections::BTreeMap::new();
        for (i, extent) in self.manifest().extents.iter().cloned().enumerate() {
            let ordinal = per_file.entry(extent.entry).or_insert(0);
            schedule.push((*ordinal, i, extent));
            *ordinal += 1;
        }
        schedule.sort_by_key(|(ordinal, i, _)| {
            match self.options.priority_extents.iter().position(|p| p == i) {
                Some(rank) => (0, rank, 0),
                None => (1, *ordinal, *i),
            }
        });
        let mut operations = stream::iter(schedule.into_iter().map(|(_, i, extent)| (i, extent)))
            .map(|(i, extent)| self.transfer(&client, i, extent))
            .buffer_unordered(self.options.concurrency);
        while let Some(result) = operations.next().await {
            result?;
        }
        drop(operations);
        if self.shared.cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        // Workers have stopped; trusted mode performs no hashes, fsync or final tree walk.
        let working = self.shared.clone();
        let verified = self.options.mode == HydrationMode::Verified;
        tokio::task::spawn_blocking(move || -> Result<()> {
            if verified {
                for entry in &working.manifest.entries {
                    if working.cancelled.load(Ordering::Acquire) {
                        return Err(Error::Cancelled);
                    }
                    if let Entry::File {
                        path, size, sha256, ..
                    } = entry
                    {
                        let full = working.staging.join(path);
                        let (digest, bytes) = crate::pack::digest_file(&full)?;
                        if &digest != sha256 || bytes != *size {
                            return Err(Error::Integrity("extent file digest".into()));
                        }
                        File::options().write(true).open(full)?.sync_all()?;
                    }
                }
            }
            extract::finish_entries(
                &working.staging,
                &working.manifest.filesystem_manifest(),
                &working.cancelled,
            )?;
            fs::write(
                working.staging.join(RECEIPT),
                format!("{}\n{}\n", working.manifest.id, working.manifest.content_id),
            )?;
            set_mode(&working.staging.join(RECEIPT), 0o600)?;
            Ok(())
        })
        .await
        .map_err(|_| Error::Worker)??;
        paths::ancestors(self.destination.parent().unwrap_or(Path::new(".")))?;
        let mut state = self.shared.state.lock().unwrap();
        if self.shared.cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if fs::symlink_metadata(&self.destination).is_ok() {
            return Err(Error::DestinationExists);
        }
        fs::rename(&self.shared.staging, &self.destination)?;
        state.root = self.destination.clone();
        state.published = true;
        drop(state);
        self.shared.wake.notify_waiters();
        Ok(ExtentReport {
            downloaded_bytes: self.shared.received.load(Ordering::Relaxed),
            completed_extents: self.manifest().extents.len(),
            file_bytes: self.manifest().total_bytes()?,
            copy_buffer_bound: self.options.buffer_bytes
                * self.options.concurrency.min(self.manifest().extents.len()),
        })
    }
}
