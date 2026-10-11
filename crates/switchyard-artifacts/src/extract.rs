use crate::{
    Chunk, Entry, Error, Manifest, Result,
    manifest::RECEIPT,
    pack::{digest_file, file_mode, set_mode},
    paths,
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

/// Verify all declared entries, permissions and content, and reject undeclared entries.
/// Existing symlink ancestors are refused, including symlinks in the destination path.
pub fn verify(destination: &Path, manifest: &Manifest) -> Result<()> {
    manifest.validate()?;
    paths::ancestors(destination)?;
    if !fs::symlink_metadata(destination)?.is_dir() {
        return Err(Error::Integrity("destination is not a directory".into()));
    }
    let mut expected = BTreeSet::new();
    for entry in &manifest.entries {
        let path = destination.join(entry.path());
        paths::ancestors(path.parent().unwrap())?;
        let meta = fs::symlink_metadata(&path)?;
        if file_mode(&meta) != entry.mode() {
            return Err(Error::Integrity("entry permissions".into()));
        }
        match entry {
            Entry::File { size, sha256, .. } => {
                if !meta.is_file()
                    || meta.len() != *size
                    || digest_file(&path)? != (sha256.clone(), *size)
                {
                    return Err(Error::Integrity("file content".into()));
                }
            }
            Entry::Directory { .. } if !meta.is_dir() => {
                return Err(Error::Integrity("directory type".into()));
            }
            Entry::Symlink { target, .. }
                if !meta.file_type().is_symlink() || fs::read_link(&path)? != Path::new(target) =>
            {
                return Err(Error::Integrity("symlink target".into()));
            }
            _ => {}
        }
        expected.insert(entry.path().to_owned());
    }
    let mut actual = BTreeSet::new();
    walk(destination, destination, &mut actual)?;
    if actual.remove(RECEIPT) {
        check_receipt(destination, manifest)?;
    }
    if actual != expected {
        return Err(Error::Integrity("unexpected filesystem entries".into()));
    }
    Ok(())
}

pub(crate) fn write_receipt(destination: &Path, manifest: &Manifest) -> Result<()> {
    let path = destination.join(RECEIPT);
    let mut file = File::options().create_new(true).write(true).open(&path)?;
    file.write_all(manifest.id.as_bytes())?;
    file.sync_all()?;
    set_mode(&path, 0o600)
}

pub(crate) fn check_receipt(destination: &Path, manifest: &Manifest) -> Result<()> {
    let path = destination.join(RECEIPT);
    let meta = fs::symlink_metadata(&path)?;
    if !meta.is_file()
        || meta.len() != 64
        || file_mode(&meta) != 0o600
        || fs::read_to_string(path)? != manifest.id
    {
        return Err(Error::Integrity("artifact identity receipt".into()));
    }
    Ok(())
}

fn walk(root: &Path, directory: &Path, paths: &mut BTreeSet<String>) -> Result<()> {
    for item in fs::read_dir(directory)? {
        let path = item?.path();
        let name = path
            .strip_prefix(root)
            .map_err(|_| Error::Integrity("tree path".into()))?
            .to_str()
            .ok_or_else(|| Error::Integrity("non-UTF8 tree entry".into()))?
            .to_owned();
        paths.insert(name);
        if fs::symlink_metadata(&path)?.is_dir() {
            walk(root, &path, paths)?;
        }
    }
    Ok(())
}

pub(crate) fn extract(
    cache: &Path,
    staging: &Path,
    manifest: &Manifest,
    concurrency: usize,
    cancelled: &AtomicBool,
    progress: impl Fn(u64) + Sync,
) -> Result<()> {
    // Real directories are created before files. Symlinks are created only after all writes.
    for entry in &manifest.entries {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        if let Entry::Directory { path, .. } = entry {
            let path = staging.join(path);
            paths::ancestors(path.parent().unwrap())?;
            fs::create_dir(&path)?;
            set_mode(&path, 0o700)?;
        }
    }
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
                                && matches!(&error, Error::Io(error) if error.kind() == std::io::ErrorKind::Interrupted)
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
    let mut decoder = zstd::stream::read::Decoder::new(File::open(&chunk_path)?)?;
    decoder.window_log_max(27)?; // At most128MiB decoder history, even for untrusted frames.
    let mut archive = tar::Archive::new(LimitedReader {
        inner: decoder,
        limit: chunk.unpacked_size,
        bytes: 0,
        cancelled,
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
        paths::ancestors(path.parent().unwrap())?;
        match expected {
            Entry::File { size, sha256, .. } => {
                if !item.header().entry_type().is_file() {
                    return Err(Error::Integrity("tar file type".into()));
                }
                let mut file = File::options().create_new(true).write(true).open(&path)?;
                let mut hash = Sha256::new();
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
                    hash.update(&buffer[..count]);
                    file.write_all(&buffer[..count])?;
                }
                file.sync_all()?;
                if written != *size || format!("{:x}", hash.finalize()) != *sha256 {
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

fn finish_entries(staging: &Path, manifest: &Manifest, cancelled: &AtomicBool) -> Result<()> {
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
        if !matches!(entry, Entry::Symlink { .. }) {
            set_mode(&staging.join(entry.path()), entry.mode())?;
        }
    }
    Ok(())
}

struct LimitedReader<'a, R> {
    inner: R,
    limit: u64,
    bytes: u64,
    cancelled: &'a AtomicBool,
}
impl<R: Read> Read for LimitedReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "artifact hydration cancelled",
            ));
        }
        // One byte beyond the declared size detects an oversized compressed stream.
        let remaining = self.limit.saturating_sub(self.bytes).saturating_add(1);
        let length = buffer
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let count = self.inner.read(&mut buffer[..length])?;
        self.bytes += count as u64;
        if self.bytes > self.limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "decompressed chunk exceeds manifest size",
            ));
        }
        Ok(count)
    }
}
