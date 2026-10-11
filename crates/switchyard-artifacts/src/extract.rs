use crate::{
    Chunk, Entry, Error, Manifest, Result,
    pack::{digest_file, set_mode},
    paths,
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

pub(crate) fn extract(
    cache: &Path,
    staging: &Path,
    manifest: &Manifest,
    concurrency: usize,
    cancelled: &AtomicBool,
    progress: impl Fn(u64) + Sync,
) -> Result<()> {
    prepare_entries(staging, manifest, cancelled, true)?;
    let next = AtomicUsize::new(0);
    let completed = AtomicU64::new(0);
    std::thread::scope(|scope| {
        let handles = (0..concurrency.min(manifest.chunks.len()))
            .map(|_| {
                scope.spawn(|| {
                    loop {
                        if cancelled.load(Ordering::Relaxed) {
                            return Err(Error::Cancelled);
                        }
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(chunk) = manifest.chunks.get(index) else {
                            return Ok(());
                        };
                        if let Err(error) =
                            extract_chunk(cache, staging, manifest, chunk, cancelled)
                        {
                            if cancelled.load(Ordering::Relaxed)
                                && matches!(&error, Error::Io(error) if error.kind() == std::io::ErrorKind::ConnectionAborted)
                            {
                                return Err(Error::Cancelled);
                            }
                            cancelled.store(true, Ordering::Relaxed);
                            return Err(error);
                        }
                        let bytes = completed.fetch_add(chunk.unpacked_size, Ordering::Relaxed)
                            + chunk.unpacked_size;
                        progress(bytes);
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut failure = None;
        for handle in handles {
            let result = handle
                .join()
                .map_err(|_| Error::Worker)
                .and_then(|result| result);
            if let Err(error) = result
                && (failure.is_none() || matches!(failure, Some(Error::Cancelled)))
            {
                failure = Some(error);
            }
        }
        failure.map_or(Ok(()), Err)
    })?;
    finish_entries(staging, manifest, cancelled)
}

pub(crate) fn prepare_entries(
    staging: &Path,
    manifest: &Manifest,
    cancelled: &AtomicBool,
    check_ancestors: bool,
) -> Result<()> {
    // Real directories are created before files. Symlinks are created only after all writes.
    for entry in &manifest.entries {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        if let Entry::Directory { path, .. } = entry {
            let path = staging.join(path);
            if check_ancestors {
                paths::ancestors(path.parent().unwrap())?;
            }
            fs::create_dir(&path)?;
            set_mode(&path, 0o700)?;
        }
    }
    Ok(())
}

fn extract_chunk(
    cache: &Path,
    staging: &Path,
    manifest: &Manifest,
    chunk: &Chunk,
    cancelled: &AtomicBool,
) -> Result<()> {
    let chunk_path = cache.join(&chunk.name);
    let metadata = fs::symlink_metadata(&chunk_path)?;
    if !metadata.is_file() || metadata.len() != chunk.size {
        return Err(Error::Integrity("cached chunk size or type changed".into()));
    }
    if digest_file(&chunk_path)? != (chunk.sha256.clone(), chunk.size) {
        return Err(Error::Integrity("cached chunk changed".into()));
    }
    extract_stream(
        File::open(&chunk_path)?,
        staging,
        manifest,
        chunk,
        cancelled,
        true,
        |_| {},
    )
}

pub(crate) fn extract_stream(
    source: impl Read,
    staging: &Path,
    manifest: &Manifest,
    chunk: &Chunk,
    cancelled: &AtomicBool,
    verify_content: bool,
    progress: impl Fn(u64),
) -> Result<()> {
    extract_stream_observed(
        source,
        staging,
        manifest,
        chunk,
        cancelled,
        verify_content,
        progress,
        |_| {},
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_stream_observed(
    source: impl Read,
    staging: &Path,
    manifest: &Manifest,
    chunk: &Chunk,
    cancelled: &AtomicBool,
    verify_content: bool,
    progress: impl Fn(u64),
    file_progress: impl Fn(u64),
) -> Result<()> {
    extract_stream_profiled(
        source,
        staging,
        manifest,
        chunk,
        cancelled,
        verify_content,
        progress,
        file_progress,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_stream_profiled(
    source: impl Read,
    staging: &Path,
    manifest: &Manifest,
    chunk: &Chunk,
    cancelled: &AtomicBool,
    verify_content: bool,
    progress: impl Fn(u64),
    file_progress: impl Fn(u64),
    stats: Option<&std::cell::RefCell<crate::profiling::ExtractStats>>,
) -> Result<()> {
    let source = TimedSource {
        inner: source,
        stats,
    };
    let mut decoder = zstd::stream::read::Decoder::new(source)?;
    decoder.window_log_max(27)?; // At most128MiB decoder history, even for untrusted frames.
    let mut archive = tar::Archive::new(LimitedReader {
        inner: decoder,
        limit: chunk.unpacked_size,
        bytes: 0,
        cancelled,
        progress: &progress,
        stats,
    });
    let mut observed = 0;
    for item in archive.entries()? {
        let mut item = item?;
        let index = *chunk
            .entries
            .get(observed)
            .ok_or_else(|| Error::Integrity("extra tar entry".into()))?;
        let expected = &manifest.entries[index];
        let header = item.header();
        if item.path()?.as_ref() != Path::new(expected.path())
            || header.mode()? != expected.mode()
            || header.size()? != expected.size()
            || header.uid()? != 0
            || header.gid()? != 0
            || header.mtime()? != 0
        {
            return Err(Error::Integrity("tar entry metadata".into()));
        }
        let path = staging.join(expected.path());
        if verify_content {
            paths::ancestors(path.parent().unwrap())?;
        }
        match expected {
            Entry::File { size, sha256, .. } => {
                if !item.header().entry_type().is_file() {
                    return Err(Error::Integrity("tar file type".into()));
                }
                let began = stats.map(|_| std::time::Instant::now());
                let mut file = File::options().create_new(true).write(true).open(&path)?;
                if let (Some(stats), Some(began)) = (stats, began) {
                    let mut stats = stats.borrow_mut();
                    stats.file_create_seconds += began.elapsed().as_secs_f64();
                    stats.files += 1;
                }
                let mut hash = verify_content.then(Sha256::new);
                let mut written = 0_u64;
                let mut buffer = [0_u8; 64 * 1024];
                loop {
                    let count = item.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    written += count as u64;
                    if written > *size {
                        return Err(Error::Integrity("file exceeds declared size".into()));
                    }
                    if let Some(hash) = &mut hash {
                        hash.update(&buffer[..count]);
                    }
                    let began = stats.map(|_| std::time::Instant::now());
                    file.write_all(&buffer[..count])?;
                    if let (Some(stats), Some(began)) = (stats, began) {
                        let mut stats = stats.borrow_mut();
                        stats.file_write_seconds += began.elapsed().as_secs_f64();
                        stats.file_bytes += count as u64;
                    }
                    file_progress(count as u64);
                }
                if verify_content {
                    file.sync_all()?;
                }
                if written != *size
                    || hash.is_some_and(|hash| format!("{:x}", hash.finalize()) != *sha256)
                {
                    return Err(Error::Integrity("extracted file digest".into()));
                }
            }
            Entry::Directory { .. } => {
                if !item.header().entry_type().is_dir() {
                    return Err(Error::Integrity("tar directory type".into()));
                }
            }
            Entry::Symlink { target, .. } => {
                if !item.header().entry_type().is_symlink()
                    || item.link_name()?.as_deref() != Some(Path::new(target))
                {
                    return Err(Error::Integrity("tar symlink target".into()));
                }
            }
        }
        observed += 1;
    }
    if observed != chunk.entries.len() {
        return Err(Error::Integrity("missing tar entries".into()));
    }
    // Consume padding and the complete zstd frame; verify checksum and exact declared length.
    let mut reader = archive.into_inner();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        if buffer[..count].iter().any(|byte| *byte != 0) {
            return Err(Error::Integrity(
                "nonzero bytes after tar terminator".into(),
            ));
        }
    }
    if reader.bytes != chunk.unpacked_size {
        return Err(Error::Integrity("unpacked chunk size".into()));
    }
    Ok(())
}

pub(crate) fn finish_entries(
    staging: &Path,
    manifest: &Manifest,
    cancelled: &AtomicBool,
) -> Result<()> {
    for entry in &manifest.entries {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        if let Entry::Symlink { path, target, .. } = entry {
            let path = staging.join(path);
            paths::ancestors(path.parent().unwrap())?;
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(target, path)?;
            }
            #[cfg(not(unix))]
            {
                let _ = (target, path);
                return Err(Error::Invalid("Unix symlinks required".into()));
            }
        }
    }
    // Set files first, then deepest directories so restrictive directory modes don't block writes.
    for entry in manifest.entries.iter().rev() {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        if !matches!(entry, Entry::Symlink { .. }) {
            set_mode(&staging.join(entry.path()), entry.mode())?;
        }
    }
    Ok(())
}

struct TimedSource<'a, R> {
    inner: R,
    stats: Option<&'a std::cell::RefCell<crate::profiling::ExtractStats>>,
}
impl<R: Read> Read for TimedSource<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let began = self.stats.map(|_| std::time::Instant::now());
        let result = self.inner.read(buffer);
        if let (Some(stats), Some(began)) = (self.stats, began) {
            stats.borrow_mut().source_read_seconds += began.elapsed().as_secs_f64();
        }
        result
    }
}

struct LimitedReader<'a, R> {
    inner: R,
    limit: u64,
    bytes: u64,
    cancelled: &'a AtomicBool,
    progress: &'a dyn Fn(u64),
    stats: Option<&'a std::cell::RefCell<crate::profiling::ExtractStats>>,
}
impl<R: Read> Read for LimitedReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionAborted,
                "artifact hydration cancelled",
            ));
        }
        // One byte beyond the declared size detects an oversized compressed stream.
        let remaining = self.limit.saturating_sub(self.bytes).saturating_add(1);
        let length = buffer
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let began = self.stats.map(|_| std::time::Instant::now());
        let count = self.inner.read(&mut buffer[..length])?;
        if let (Some(stats), Some(began)) = (self.stats, began) {
            stats.borrow_mut().decoder_read_seconds += began.elapsed().as_secs_f64();
        }
        self.bytes += count as u64;
        if self.bytes > self.limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "decompressed chunk exceeds manifest size",
            ));
        }
        let began = self.stats.map(|_| std::time::Instant::now());
        (self.progress)(count as u64);
        if let (Some(stats), Some(began)) = (self.stats, began) {
            stats.borrow_mut().progress_seconds += began.elapsed().as_secs_f64();
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    #[test]
    fn cancelled_archive_reader_is_terminal_for_read_exact() {
        struct Bounded<R> {
            inner: R,
            calls: usize,
        }
        impl<R: Read> Read for Bounded<R> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                self.calls += 1;
                if self.calls > 8 {
                    return Err(io::Error::other("retry limit reached"));
                }
                self.inner.read(buffer)
            }
        }
        let cancellation = AtomicBool::new(true);
        let reader = LimitedReader {
            inner: io::Cursor::new(vec![0; 512]),
            limit: 512,
            bytes: 0,
            cancelled: &cancellation,
            progress: &|_| {},
            stats: None,
        };
        let mut reader = Bounded {
            inner: reader,
            calls: 0,
        };
        assert_eq!(
            reader.read_exact(&mut [0; 512]).unwrap_err().kind(),
            io::ErrorKind::ConnectionAborted
        );
        assert_eq!(reader.calls, 1);
    }
}
