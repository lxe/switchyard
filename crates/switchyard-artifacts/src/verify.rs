//! Check complete artifact topology before hashing its files with bounded workers.
use crate::{
    Entry, Error, Manifest, Result,
    manifest::RECEIPT,
    pack::{digest_file_with_buffer, file_mode, set_mode},
    paths,
};
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

/// Bound the number of concurrent file readers used for complete verification.
#[derive(Debug, Clone, Copy)]
pub struct VerificationOptions {
    pub concurrency: usize,
}
impl Default for VerificationOptions {
    fn default() -> Self {
        Self {
            concurrency: std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(8),
        }
    }
}

/// Verify every declared hash, size, type, permission and link, rejecting extra entries.
/// Ancestors outside the artifact are checked once. A no-follow topology walk checks
/// every internal directory before entering it; all regular files are then hashed.
/// The caller must exclusively own the tree and its ancestors during verification.
pub fn verify(destination: &Path, manifest: &Manifest) -> Result<()> {
    verify_with_options(destination, manifest, VerificationOptions::default())
}

/// Complete verification with an explicit file-reading concurrency between 1 and 64.
/// Each worker reuses a 1 MiB buffer; file contents are always hashed in full.
pub fn verify_with_options(
    destination: &Path,
    manifest: &Manifest,
    options: VerificationOptions,
) -> Result<()> {
    if !(1..=64).contains(&options.concurrency) {
        return Err(Error::Invalid("verification concurrency".into()));
    }
    manifest.validate()?;
    paths::ancestors(destination)?;
    if !fs::symlink_metadata(destination)?.is_dir() {
        return Err(Error::Integrity("destination is not a directory".into()));
    }
    let mut files = Vec::new();
    let mut observed = 0;
    topology(
        destination,
        destination,
        manifest,
        &mut files,
        &mut observed,
    )?;
    if observed != manifest.entries.len() {
        return Err(Error::Integrity("missing filesystem entries".into()));
    }
    // Start the largest files first so one large library does not become a final straggler.
    files.sort_unstable_by_key(|index| std::cmp::Reverse(manifest.entries[*index].size()));
    hash_files(destination, manifest, &files, options.concurrency)
}

fn topology(
    root: &Path,
    directory: &Path,
    manifest: &Manifest,
    files: &mut Vec<usize>,
    observed: &mut usize,
) -> Result<()> {
    for item in fs::read_dir(directory)? {
        let path = item?.path();
        let name = path
            .strip_prefix(root)
            .map_err(|_| Error::Integrity("tree path".into()))?
            .to_str()
            .ok_or_else(|| Error::Integrity("non-UTF8 tree entry".into()))?;
        if name == RECEIPT {
            check_receipt(root, manifest)?;
            continue;
        }
        let index = manifest
            .entries
            .binary_search_by(|entry| entry.path().cmp(name))
            .map_err(|_| Error::Integrity("unexpected filesystem entry".into()))?;
        let entry = &manifest.entries[index];
        let metadata = fs::symlink_metadata(&path)?;
        if file_mode(&metadata) != entry.mode() {
            return Err(Error::Integrity("entry permissions".into()));
        }
        match entry {
            Entry::File { size, .. } => {
                if !metadata.is_file() || metadata.len() != *size {
                    return Err(Error::Integrity("file type or size".into()));
                }
                files.push(index);
            }
            Entry::Directory { .. } => {
                if !metadata.is_dir() {
                    return Err(Error::Integrity("directory type".into()));
                }
                topology(root, &path, manifest, files, observed)?;
            }
            Entry::Symlink { target, .. } => {
                if !metadata.file_type().is_symlink() || fs::read_link(&path)? != Path::new(target)
                {
                    return Err(Error::Integrity("symlink target".into()));
                }
            }
        }
        *observed += 1;
    }
    Ok(())
}

fn hash_files(root: &Path, manifest: &Manifest, files: &[usize], concurrency: usize) -> Result<()> {
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let handles = (0..concurrency.min(files.len()))
            .map(|_| {
                scope.spawn(|| {
                    let mut buffer = vec![0_u8; 1024 * 1024];
                    while !failed.load(Ordering::Relaxed) {
                        let Some(index) = files.get(next.fetch_add(1, Ordering::Relaxed)) else {
                            return Ok(());
                        };
                        let Entry::File {
                            path, size, sha256, ..
                        } = &manifest.entries[*index]
                        else {
                            return Err(Error::Integrity("verification file queue".into()));
                        };
                        let result = digest_file_with_buffer(&root.join(path), &mut buffer);
                        match result {
                            Ok((digest, bytes)) if bytes == *size && digest == *sha256 => {}
                            Ok(_) => {
                                failed.store(true, Ordering::Relaxed);
                                return Err(Error::Integrity("file content".into()));
                            }
                            Err(error) => {
                                failed.store(true, Ordering::Relaxed);
                                return Err(error);
                            }
                        }
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        let mut failure = None;
        for handle in handles {
            if let Err(error) = handle
                .join()
                .map_err(|_| Error::Worker)
                .and_then(|result| result)
                && failure.is_none()
            {
                failure = Some(error);
            }
        }
        failure.map_or(Ok(()), Err)
    })
}

pub(crate) fn write_receipt(destination: &Path, manifest: &Manifest) -> Result<()> {
    write_receipt_inner(destination, manifest, true)
}

pub(crate) fn write_trusted_receipt(destination: &Path, manifest: &Manifest) -> Result<()> {
    write_receipt_inner(destination, manifest, false)
}

fn write_receipt_inner(destination: &Path, manifest: &Manifest, durable: bool) -> Result<()> {
    let path = destination.join(RECEIPT);
    let mut file = File::options().create_new(true).write(true).open(&path)?;
    file.write_all(manifest.id.as_bytes())?;
    if durable {
        file.sync_all()?;
    }
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
