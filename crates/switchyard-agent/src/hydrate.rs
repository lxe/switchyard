//! Verified, bounded archive hydration. Transfer URLs never appear in errors.

use crate::{Chunk, Error, Manifest, Phase, Result, State, Ticket};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, Read},
    path::{Component, Path},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use switchyard_artifacts::HydrationMode;
use tokio::{io::AsyncWriteExt, sync::RwLock};

pub(crate) mod diagnostics;
mod streaming;
pub use diagnostics::{ModelDiagnostics, ModelSnapshot};

fn relative(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && !path.contains('\0')
        && Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}
fn digest_valid(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Manifest {
    /// Validate chunk conservation, contiguous sequences, and safe expected file paths.
    pub fn validate(&self) -> Result<()> {
        self.validate_structure()?;
        let mut hash = Sha256::new();
        for chunk in &self.chunks {
            hash.update(chunk.sha256.as_bytes());
        }
        if format!("{:x}", hash.finalize()) != self.sha256 {
            return Err(Error::Manifest("archive digest mismatch".into()));
        }
        Ok(())
    }
    /// Check paths, types, byte conservation and sequences without recalculating identity.
    pub fn validate_structure(&self) -> Result<()> {
        if self.format != "shards-v1"
            || !matches!(self.compress.as_str(), "none" | "zstd")
            || self.chunks.is_empty()
            || self.raw_size == 0
            || self.files.is_empty()
            || !digest_valid(&self.sha256)
        {
            return Err(Error::Manifest(
                "requires a nonempty shards-v1 model archive".into(),
            ));
        }
        let mut names = BTreeSet::new();
        let mut by_shard: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let (mut total, mut raw) = (0u64, 0u64);
        for chunk in &self.chunks {
            if !relative(&chunk.name)
                || Path::new(&chunk.name).components().count() != 1
                || !names.insert(&chunk.name)
                || chunk.size == 0
                || chunk.raw == 0
                || !digest_valid(&chunk.sha256)
                || (self.compress == "none" && chunk.raw != chunk.size)
            {
                return Err(Error::Manifest("invalid or duplicate chunk".into()));
            }
            total = total
                .checked_add(chunk.size)
                .ok_or_else(|| Error::Manifest("size overflow".into()))?;
            raw = raw
                .checked_add(chunk.raw)
                .ok_or_else(|| Error::Manifest("size overflow".into()))?;
            by_shard.entry(chunk.shard).or_default().push(chunk.seq);
        }
        if total != self.total_size || raw != self.raw_size {
            return Err(Error::Manifest("archive size or digest mismatch".into()));
        }
        for mut sequence in by_shard.into_values() {
            sequence.sort_unstable();
            if sequence.iter().enumerate().any(|(i, x)| i != *x) {
                return Err(Error::Manifest("noncontiguous shard sequence".into()));
            }
        }
        let mut file_bytes = 0u64;
        for (path, info) in &self.files {
            if !relative(path) || info.sha256.as_deref().is_some_and(|s| !digest_valid(s)) {
                return Err(Error::Manifest("invalid file path or digest".into()));
            }
            for parent in Path::new(path)
                .ancestors()
                .skip(1)
                .filter(|p| !p.as_os_str().is_empty())
            {
                if self.files.contains_key(parent.to_string_lossy().as_ref()) {
                    return Err(Error::Manifest(
                        "a model file cannot also be a directory".into(),
                    ));
                }
            }
            file_bytes = file_bytes
                .checked_add(info.size)
                .ok_or_else(|| Error::Manifest("size overflow".into()))?;
        }
        if file_bytes > raw {
            return Err(Error::Manifest("expected files exceed archive size".into()));
        }
        Ok(())
    }
}

/// Download verified chunks, extract shards with bounded parallelism, then publish the model.
///
/// `bytes_downloaded` measures bytes retained by current attempts, so a retry can reduce
/// progress. The observed transfer rate includes time spent on failed attempts.
pub async fn hydrate(ticket: &Ticket, root: &Path, state: Arc<RwLock<State>>) -> Result<()> {
    preflight(ticket, root, &state).await?;
    hydrate_with_limit(
        ticket,
        root,
        state,
        Arc::new(tokio::sync::Semaphore::new(ticket.concurrency)),
    )
    .await
}

pub(crate) async fn preflight(ticket: &Ticket, root: &Path, state: &RwLock<State>) -> Result<()> {
    ticket.startup.validate()?;
    if ticket.startup.early_launch && ticket.runtime.is_none() {
        return Err(Error::Manifest(
            "early launch requires a hydrated cooperative runtime".into(),
        ));
    }
    if let Some((artifact, loader)) = ticket.source.extent() {
        if ticket.runtime.is_none() {
            return Err(Error::Manifest(
                "extent source requires a selected runtime".into(),
            ));
        }
        loader.validate()?;
        crate::model::validate_extent_caches(ticket)?;
        if ticket.startup.mode == HydrationMode::Verified {
            artifact.manifest.validate()?;
        } else {
            artifact.manifest.validate_structure()?;
        }
        if !(1..=64).contains(&ticket.concurrency) || ticket.lifetime_s == 0 {
            return Err(Error::Manifest("invalid extent transfer bounds".into()));
        }
        let metadata = crate::model::metadata_bytes(&artifact.manifest)?;
        let required = artifact
            .manifest
            .total_bytes()?
            .checked_add(metadata)
            .ok_or_else(|| Error::Manifest("extent capacity overflow".into()))?;
        let observation = crate::startup::observe(root, required)?;
        crate::startup::check_capacity(
            &observation,
            ticket.startup.require_model_ram,
            loader.pinned_bytes as u64 + (64 << 20),
        )?;
        state.write().await.model_storage = Some(observation);
        return Ok(());
    }
    let legacy = ticket.legacy()?;
    let ticket = &legacy;
    if ticket.startup.mode == HydrationMode::Verified {
        ticket.manifest.validate()?;
    } else {
        ticket.manifest.validate_structure()?;
    }
    if !(1..=64).contains(&ticket.concurrency) || ticket.stall_s == 0 {
        return Err(Error::Manifest("invalid transfer bounds".into()));
    }
    if ticket
        .manifest
        .chunks
        .iter()
        .any(|c| !ticket.urls.contains_key(&c.name))
    {
        return Err(Error::Manifest("missing chunk URL".into()));
    }
    let required = ticket
        .manifest
        .files
        .values()
        .try_fold(0_u64, |sum, file| sum.checked_add(file.size))
        .ok_or_else(|| Error::Manifest("model capacity overflow".into()))?;
    let mut observation = crate::startup::observe(root, required)?;
    // A trusted receipt is reusable without another copy or a content scan.
    if ticket.startup.mode == HydrationMode::TrustedStreaming && streaming::cached(ticket, root)? {
        observation.required_bytes = 0;
    }
    state.write().await.model_storage = Some(observation.clone());
    crate::startup::check_capacity(
        &observation,
        ticket.startup.require_model_ram,
        streaming::buffer_reserve(ticket.concurrency),
    )?;
    Ok(())
}

pub(crate) async fn hydrate_with_limit(
    ticket: &Ticket,
    root: &Path,
    state: Arc<RwLock<State>>,
    limit: Arc<tokio::sync::Semaphore>,
) -> Result<()> {
    if let Some((artifact, _)) = ticket.source.extent() {
        let hydration = switchyard_artifacts::extents::ExtentHydration::prepare(
            artifact.clone(),
            &root.join("model"),
            switchyard_artifacts::extents::ExtentOptions {
                mode: ticket.startup.mode,
                concurrency: ticket.concurrency,
                download_limit: Some(limit),
                request_timeout: Duration::from_secs(ticket.lifetime_s),
                ..Default::default()
            },
        )?;
        let started = Instant::now();
        let report = hydration.run().await?;
        let mut st = state.write().await;
        st.bytes_downloaded = report.downloaded_bytes;
        st.hydration_seconds = started.elapsed().as_secs_f64();
        return Ok(());
    }
    let legacy = ticket.legacy()?;
    let ticket = &legacy;
    if ticket.startup.mode == HydrationMode::TrustedStreaming {
        return streaming::hydrate(ticket, root, state, limit).await;
    }
    let dest = root.join("model");
    let marker = dest.join(".switchyard-archive");
    if std::fs::symlink_metadata(&marker).is_ok_and(|m| m.is_file())
        && std::fs::read_to_string(&marker).is_ok_and(|s| s == ticket.manifest.sha256)
    {
        let manifest = ticket.manifest.clone();
        let dir = dest.clone();
        let valid = tokio::task::spawn_blocking(move || verify_files(&manifest, &dir))
            .await
            .map_err(|_| Error::Task)?;
        if valid.is_ok() {
            let mut st = state.write().await;
            st.model_cached = true;
            return Ok(());
        }
        // A marker is only a hint. Corrupted cache files are replaced from verified chunks.
    }
    let workspace = Arc::new(streaming::Workspace::new(root)?);
    let chunks = workspace.path.join("chunks");
    let stage = workspace.path.join("model");
    std::fs::create_dir_all(&chunks)?;
    std::fs::create_dir_all(&stage)?;
    {
        let mut st = state.write().await;
        st.phase = Phase::Hydrating;
        st.bytes_downloaded = 0;
        st.model_cached = false;
        st.volume_bytes_per_s = None;
    }
    let client = reqwest::Client::builder()
        .user_agent("Switchyard/0.1")
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(ticket.stall_s))
        .timeout(Duration::from_secs(ticket.lifetime_s.clamp(1, 600)))
        .build()
        .map_err(|_| Error::Transfer("client initialization".into()))?;
    let started = Instant::now();
    let tasks = futures_util::stream::iter(ticket.manifest.chunks.iter().cloned())
        .map(|chunk| {
            let state = state.clone();
            let client = client.clone();
            let dir = chunks.clone();
            let url = &ticket.urls[&chunk.name];
            let limit = limit.clone();
            async move {
                for attempt in 0..3 {
                    let permit = limit
                        .clone()
                        .acquire_owned()
                        .await
                        .map_err(|_| Error::Task)?;
                    let mut retained = 0;
                    let result = download(
                        &client,
                        url,
                        &chunk,
                        &dir.join(&chunk.name),
                        &state,
                        started,
                        &mut retained,
                    )
                    .await;
                    drop(permit);
                    if result.is_ok() {
                        return result;
                    }
                    // Roll back this attempt's bytes; concurrent transfers remain intact.
                    let mut st = state.write().await;
                    st.bytes_downloaded = st.bytes_downloaded.saturating_sub(retained);
                    sample_rate(&mut st, started);
                    drop(st);
                    if attempt == 2 {
                        return result;
                    }
                    tokio::time::sleep(Duration::from_millis(250 * (attempt + 1))).await;
                }
                unreachable!("the final attempt returns its result")
            }
        })
        .buffer_unordered(ticket.concurrency);
    futures_util::pin_mut!(tasks);
    while let Some(result) = tasks.next().await {
        result?;
    }
    let manifest = ticket.manifest.clone();
    let archive_dir = chunks.clone();
    let target = stage.clone();
    let concurrency = ticket.concurrency;
    let _workspace = tokio::task::spawn_blocking(move || {
        let _ownership = workspace;
        extract(&manifest, &archive_dir, &target, concurrency).map(|()| _ownership)
    })
    .await
    .map_err(|_| Error::Task)??;
    std::fs::write(stage.join(".switchyard-archive"), &ticket.manifest.sha256)?;
    remove_existing(&dest)?;
    std::fs::rename(stage, dest)?;
    std::fs::remove_dir_all(&chunks)?;
    let mut st = state.write().await;
    st.hydration_seconds = started.elapsed().as_secs_f64();
    // Keep the pre-extraction download rate; hydration duration includes both stages.
    Ok(())
}

fn remove_existing(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(path)?,
        Ok(_) => std::fs::remove_file(path)?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
fn sample_rate(state: &mut State, started: Instant) {
    state.hydration_seconds = started.elapsed().as_secs_f64();
    state.volume_bytes_per_s =
        Some(state.bytes_downloaded as f64 / state.hydration_seconds.max(0.001));
}

async fn download(
    client: &reqwest::Client,
    url: &str,
    chunk: &Chunk,
    path: &Path,
    state: &RwLock<State>,
    started: Instant,
    retained: &mut u64,
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
    let mut file = tokio::fs::File::create(path).await?;
    let mut digest = Sha256::new();
    while let Some(piece) = stream.next().await {
        let piece = piece.map_err(|_| Error::Transfer(format!("{} stalled", chunk.name)))?;
        let size = retained
            .checked_add(piece.len() as u64)
            .ok_or_else(|| Error::Transfer("response size overflow".into()))?;
        if size > chunk.size {
            return Err(Error::Transfer(format!("{} size exceeded", chunk.name)));
        }
        file.write_all(&piece).await?;
        digest.update(&piece);
        *retained = size;
        let mut st = state.write().await;
        st.bytes_downloaded += piece.len() as u64;
        sample_rate(&mut st, started);
    }
    file.flush().await?;
    if *retained != chunk.size || format!("{:x}", digest.finalize()) != chunk.sha256 {
        return Err(Error::Transfer(format!(
            "{} digest or size mismatch",
            chunk.name
        )));
    }
    Ok(())
}

/// A chunk must decompress to exactly its declared size, including tar padding.
struct ExactReader<R> {
    inner: R,
    remaining: u64,
    checked_end: bool,
}
impl<R: Read> Read for ExactReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            if !self.checked_end {
                self.checked_end = true;
                if self.inner.read(&mut [0])? != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "chunk exceeds declared raw size",
                    ));
                }
            }
            return Ok(0);
        }
        let limit = self.remaining.min(buf.len() as u64) as usize;
        let n = self.inner.read(&mut buf[..limit])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "chunk is shorter than declared raw size",
            ));
        }
        self.remaining -= n as u64;
        Ok(n)
    }
}

fn extract(manifest: &Manifest, chunks: &Path, dest: &Path, concurrency: usize) -> Result<()> {
    let mut by_shard: BTreeMap<usize, Vec<&Chunk>> = BTreeMap::new();
    for chunk in &manifest.chunks {
        by_shard.entry(chunk.shard).or_default().push(chunk);
    }
    let shards: Vec<_> = by_shard
        .into_values()
        .map(|mut s| {
            s.sort_by_key(|c| c.seq);
            s
        })
        .collect();
    let next = AtomicUsize::new(0);
    let seen = Mutex::new(BTreeSet::new());
    let failures = Mutex::new(Vec::new());
    // Keep extraction threads bounded independently of shard count.
    std::thread::scope(|scope| {
        for _ in 0..concurrency.min(shards.len()).min(8) {
            let (shards, next, seen, failures) = (&shards, &next, &seen, &failures);
            scope.spawn(move || {
                while let Some(shard) = shards.get(next.fetch_add(1, Ordering::Relaxed)) {
                    if let Err(error) = extract_shard(manifest, shard, chunks, dest, seen) {
                        failures
                            .lock()
                            .expect("extraction failure lock")
                            .push(error);
                        break;
                    }
                }
            });
        }
    });
    if let Some(error) = failures
        .into_inner()
        .map_err(|_| Error::Task)?
        .into_iter()
        .next()
    {
        return Err(error);
    }
    if seen.into_inner().map_err(|_| Error::Task)?.len() != manifest.files.len() {
        return Err(Error::Manifest("archive is missing expected files".into()));
    }
    verify_files(manifest, dest)
}

fn extract_shard(
    manifest: &Manifest,
    shard: &[&Chunk],
    chunks: &Path,
    dest: &Path,
    seen: &Mutex<BTreeSet<String>>,
) -> Result<()> {
    let pieces = shard.iter().map(|chunk| {
        let file = std::fs::File::open(chunks.join(&chunk.name))?;
        let piece: Box<dyn Read> = if manifest.compress == "zstd" {
            Box::new(zstd::stream::read::Decoder::new(file)?)
        } else {
            Box::new(file)
        };
        Ok(ExactReader {
            inner: piece,
            remaining: chunk.raw,
            checked_end: false,
        })
    });
    let reader = SequentialReader {
        pieces,
        current: None,
    };
    let mut archive = tar::Archive::new(reader);
    extract_reader(manifest, &mut archive, dest, seen)?;
    io::copy(&mut archive.into_inner(), &mut io::sink())?;
    Ok(())
}

/// Construct each piece only when needed and release its decoder at EOF.
/// Nested `Read::chain` retains every completed reader until the archive ends.
struct SequentialReader<I, R> {
    pieces: I,
    current: Option<R>,
}
impl<I, R> Read for SequentialReader<I, R>
where
    I: Iterator<Item = io::Result<R>>,
    R: Read,
{
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            if let Some(reader) = &mut self.current {
                let n = reader.read(buffer)?;
                if n != 0 {
                    return Ok(n);
                }
                self.current = None;
            }
            self.current = self.pieces.next().transpose()?;
            if self.current.is_none() {
                return Ok(0);
            }
        }
    }
}

fn extract_reader<R: Read>(
    manifest: &Manifest,
    archive: &mut tar::Archive<R>,
    dest: &Path,
    seen: &Mutex<BTreeSet<String>>,
) -> Result<()> {
    extract_observed(manifest, archive, dest, seen, None)
}

fn extract_observed<R: Read>(
    manifest: &Manifest,
    archive: &mut tar::Archive<R>,
    dest: &Path,
    seen: &Mutex<BTreeSet<String>>,
    trace: Option<&diagnostics::ShardTrace>,
) -> Result<()> {
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let name = path
            .to_str()
            .ok_or_else(|| Error::Manifest("archive path is not UTF-8".into()))?;
        if !relative(name)
            || !(entry.header().entry_type().is_file() || entry.header().entry_type().is_dir())
        {
            return Err(Error::Manifest("archive contains an unsafe entry".into()));
        }
        let output = dest.join(&path);
        if entry.header().entry_type().is_dir() {
            if manifest.files.contains_key(name) {
                return Err(Error::Manifest(
                    "archive directory replaces a model file".into(),
                ));
            }
            std::fs::create_dir_all(output)?;
            continue;
        }
        let expected = manifest
            .files
            .get(name)
            .ok_or_else(|| Error::Manifest("archive contains an unexpected file".into()))?;
        if entry.size() != expected.size
            || !seen
                .lock()
                .map_err(|_| Error::Task)?
                .insert(name.to_owned())
        {
            return Err(Error::Manifest(
                "archive contains a duplicate file or an invalid file size".into(),
            ));
        }
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        let written = if let Some(trace) = trace {
            io::copy(
                &mut entry,
                &mut diagnostics::ObservedWriter {
                    inner: &mut file,
                    trace,
                },
            )?
        } else {
            io::copy(&mut entry, &mut file)?
        };
        if written != expected.size {
            return Err(Error::Manifest("archive file was truncated".into()));
        }
    }
    // Tar stops at its end marker. Drain padding to validate every chunk's raw length.
    // Drain the underlying reader separately after returning ownership to the caller.
    Ok(())
}

fn verify_files(manifest: &Manifest, dest: &Path) -> Result<()> {
    if !std::fs::symlink_metadata(dest)?.is_dir() {
        return Err(Error::Manifest("model cache is not a directory".into()));
    }
    for (name, info) in &manifest.files {
        let path = dest.join(name);
        for parent in path.ancestors().skip(1).take_while(|p| *p != dest) {
            if !std::fs::symlink_metadata(parent)?.is_dir() {
                return Err(Error::Manifest(
                    "model cache contains an unsafe directory".into(),
                ));
            }
        }
        let meta = std::fs::symlink_metadata(&path)?;
        if !meta.is_file() || meta.len() != info.size {
            return Err(Error::Manifest("cached file type or size mismatch".into()));
        }
        if let Some(expected) = &info.sha256 {
            let mut file = std::fs::File::open(path)?;
            let mut digest = Sha256::new();
            let mut buffer = vec![0u8; 1 << 20];
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                digest.update(&buffer[..n]);
            }
            if format!("{:x}", digest.finalize()) != *expected {
                return Err(Error::Manifest("cached file digest mismatch".into()));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
