use crate::{Chunk, Entry, Error, Manifest, Metadata, Result, manifest::FORMAT, paths};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
};

#[derive(Debug, Clone)]
pub struct PackOptions {
    /// Soft uncompressed target; files remain intact, including files larger than this target.
    pub target_shard_bytes: u64,
    pub zstd_level: i32,
}
impl Default for PackOptions {
    fn default() -> Self {
        Self {
            target_shard_bytes: 128 * 1024 * 1024,
            zstd_level: 6,
        }
    }
}

/// Pack a directory without following symlinks. Output must be outside the source tree.
/// Each chunk and file is streamed; memory use does not scale with file size.
pub fn pack(
    source: &Path,
    out: &Path,
    metadata: Metadata,
    options: PackOptions,
) -> Result<Manifest> {
    if options.target_shard_bytes == 0 || !(-7..=22).contains(&options.zstd_level) {
        return Err(Error::Invalid("packing options".into()));
    }
    paths::ancestors(source)?;
    let source = fs::canonicalize(source)?;
    paths::directory(out)?;
    let out = fs::canonicalize(out)?;
    if out.starts_with(&source) {
        return Err(Error::Invalid(
            "archive output is inside source tree".into(),
        ));
    }
    let mut entries = vec![];
    collect(&source, &source, &mut entries)?;
    entries.sort_by(|a, b| a.path().cmp(b.path()));
    let mut groups: Vec<Vec<usize>> = vec![];
    let mut group = vec![];
    let mut size = 0_u64;
    for (index, entry) in entries.iter().enumerate() {
        let estimate = entry
            .size()
            .checked_add(1024)
            .ok_or_else(|| Error::Invalid("entry size overflow".into()))?;
        if !group.is_empty() && size.saturating_add(estimate) > options.target_shard_bytes {
            groups.push(std::mem::take(&mut group));
            size = 0;
        }
        group.push(index);
        size = size.saturating_add(estimate);
    }
    if !group.is_empty() {
        groups.push(group);
    }
    let mut manifest = Manifest {
        format: FORMAT.into(),
        id: "0".repeat(64),
        metadata,
        entries,
        chunks: vec![],
    };
    // Validate names, links and permissions before writing any chunk.
    manifest.validate_entries()?;
    for group in groups {
        manifest.chunks.push(write_chunk(
            &source,
            &out,
            &manifest.entries,
            group,
            &options,
        )?);
    }
    manifest.id = manifest.identity()?;
    manifest.validate()?;
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    let final_path = out.join(format!("{}.manifest.json", manifest.id));
    if !final_path.exists() {
        let temporary = out.join(format!(".{}.manifest.partial", paths::nonce()?));
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(temporary, final_path)?;
    }
    Ok(manifest)
}

fn collect(root: &Path, directory: &Path, entries: &mut Vec<Entry>) -> Result<()> {
    for item in fs::read_dir(directory)? {
        let item = item?;
        let path = item.path();
        let relative = path
            .strip_prefix(root)
            .map_err(|_| Error::Invalid("source path".into()))?
            .to_str()
            .ok_or_else(|| Error::Invalid("non-UTF8 source path".into()))?
            .to_owned();
        paths::relative(&relative)?;
        let meta = fs::symlink_metadata(&path)?;
        let mode = file_mode(&meta);
        if meta.is_file() {
            let (sha256, size) = digest_file(&path)?;
            entries.push(Entry::File {
                path: relative,
                mode,
                size,
                sha256,
            });
        } else if meta.is_dir() {
            entries.push(Entry::Directory {
                path: relative,
                mode,
            });
            collect(root, &path, entries)?;
        } else if meta.file_type().is_symlink() {
            let target = fs::read_link(&path)?
                .to_str()
                .ok_or_else(|| Error::Invalid("non-UTF8 symlink target".into()))?
                .to_owned();
            entries.push(Entry::Symlink {
                path: relative,
                mode: 0o777,
                target,
            });
        } else {
            return Err(Error::Invalid("special files cannot be archived".into()));
        }
        if entries.len() > 1_000_000 {
            return Err(Error::Invalid("entry limit".into()));
        }
    }
    Ok(())
}

fn write_chunk(
    source: &Path,
    out: &Path,
    entries: &[Entry],
    indices: Vec<usize>,
    options: &PackOptions,
) -> Result<Chunk> {
    let temporary = out.join(format!(".{}.chunk.partial", paths::nonce()?));
    let result = (|| {
        let file = File::options()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let mut encoder = zstd::stream::write::Encoder::new(file, options.zstd_level)?;
        encoder.include_checksum(true)?;
        encoder.window_log(27)?;
        let mut archive = tar::Builder::new(CountWriter {
            inner: encoder,
            bytes: 0,
        });
        for index in &indices {
            let entry = &entries[*index];
            let mut header = tar::Header::new_gnu();
            header.set_uid(0);
            header.set_gid(0);
            header.set_mtime(0);
            header.set_mode(entry.mode());
            header.set_size(entry.size());
            match entry {
                Entry::File {
                    path, sha256, size, ..
                } => {
                    paths::ancestors(source.join(path).parent().unwrap())?;
                    if !fs::symlink_metadata(source.join(path))?.is_file() {
                        return Err(Error::Integrity(
                            "source file changed during packing".into(),
                        ));
                    }
                    let mut reader = HashReader {
                        inner: File::open(source.join(path))?,
                        hash: Sha256::new(),
                        bytes: 0,
                    };
                    header.set_entry_type(tar::EntryType::Regular);
                    header.set_cksum();
                    archive.append_data(&mut header, path, &mut reader)?;
                    if reader.bytes != *size || format!("{:x}", reader.hash.finalize()) != *sha256 {
                        return Err(Error::Integrity(
                            "source file changed during packing".into(),
                        ));
                    }
                }
                Entry::Directory { path, .. } => {
                    header.set_entry_type(tar::EntryType::Directory);
                    header.set_cksum();
                    archive.append_data(&mut header, path, std::io::empty())?;
                }
                Entry::Symlink { path, target, .. } => {
                    header.set_entry_type(tar::EntryType::Symlink);
                    header.set_cksum();
                    archive.append_link(&mut header, path, target)?;
                }
            }
        }
        archive.finish()?;
        let writer = archive.into_inner()?;
        let unpacked_size = writer.bytes;
        writer.inner.finish()?.sync_all()?;
        let (sha256, size) = digest_file(&temporary)?;
        let name = format!("{sha256}.tar.zst");
        let final_path = out.join(&name);
        if fs::symlink_metadata(&final_path).is_ok() {
            if digest_file(&final_path)? != (sha256.clone(), size) {
                return Err(Error::Integrity("existing chunk content".into()));
            }
            fs::remove_file(&temporary)?;
        } else {
            fs::rename(&temporary, final_path)?;
        }
        Ok(Chunk {
            name,
            sha256,
            size,
            unpacked_size,
            entries: indices,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub(crate) fn digest_file(path: &Path) -> Result<(String, u64)> {
    if !fs::symlink_metadata(path)?.is_file() {
        return Err(Error::Invalid("expected regular file".into()));
    }
    let mut reader = HashReader {
        inner: File::open(path)?,
        hash: Sha256::new(),
        bytes: 0,
    };
    std::io::copy(&mut reader, &mut std::io::sink())?;
    Ok((format!("{:x}", reader.hash.finalize()), reader.bytes))
}

pub(crate) fn file_mode(meta: &fs::Metadata) -> u32 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o7777
    }
    #[cfg(not(unix))]
    {
        if meta.permissions().readonly() {
            0o444
        } else {
            0o644
        }
    }
}

pub(crate) fn set_mode(path: &Path, mode: u32) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
        Err(Error::Invalid("Unix permissions required".into()))
    }
}

struct CountWriter<W> {
    inner: W,
    bytes: u64,
}
impl<W: Write> Write for CountWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let count = self.inner.write(bytes)?;
        self.bytes += count as u64;
        Ok(count)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

struct HashReader<R> {
    inner: R,
    hash: Sha256,
    bytes: u64,
}
impl<R: Read> Read for HashReader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(bytes)?;
        self.hash.update(&bytes[..count]);
        self.bytes += count as u64;
        Ok(count)
    }
}
