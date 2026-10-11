use super::{Extent, ExtentManifest, manifest::FORMAT};
use crate::{Entry, Error, Metadata, Result, pack::collect, paths};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
};

/// Pack regular files into independent raw objects, preserving exact file bytes.
/// Whole-file hashing happens at preparation, never in trusted startup.
/// The caller exclusively owns source and output for the duration of this call.
pub fn pack_extents(
    source: &Path,
    out: &Path,
    metadata: Metadata,
    extent_bytes: u64,
) -> Result<ExtentManifest> {
    if extent_bytes == 0 {
        return Err(Error::Invalid("zero extent size".into()));
    }
    paths::ancestors(source)?;
    let source = fs::canonicalize(source)?;
    paths::directory(out)?;
    let out = fs::canonicalize(out)?;
    if out.starts_with(&source) {
        return Err(Error::Invalid("extent output inside source".into()));
    }
    let mut entries = vec![];
    collect(&source, &source, &mut entries)?;
    entries.sort_by(|a, b| a.path().cmp(b.path()));
    let mut manifest = ExtentManifest {
        format: FORMAT.into(),
        id: "0".repeat(64),
        content_id: "0".repeat(64),
        metadata,
        entries,
        extents: vec![],
    };
    manifest.filesystem_manifest().validate_entries()?;
    if manifest
        .entries
        .iter()
        .any(|e| matches!(e, Entry::Symlink { .. }))
    {
        return Err(Error::Invalid(
            "extent packing does not support symlinks".into(),
        ));
    }
    let mut buffer = vec![0_u8; 64 * 1024];
    for (entry, file) in manifest.entries.iter().enumerate() {
        let Entry::File {
            path, size, sha256, ..
        } = file
        else {
            continue;
        };
        let mut input = File::open(source.join(path))?;
        let mut file_hash = Sha256::new();
        let mut offset = 0;
        while offset < *size {
            let length = extent_bytes.min(size - offset);
            let temporary = out.join(format!(".extent-{}.partial", paths::nonce()?));
            let result = (|| -> Result<Extent> {
                let mut output = File::options()
                    .create_new(true)
                    .write(true)
                    .open(&temporary)?;
                let mut hash = Sha256::new();
                let mut remaining = length;
                while remaining != 0 {
                    let count = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
                    input.read_exact(&mut buffer[..count])?;
                    output.write_all(&buffer[..count])?;
                    hash.update(&buffer[..count]);
                    file_hash.update(&buffer[..count]);
                    remaining -= count as u64;
                }
                output.sync_all()?;
                let digest = format!("{:x}", hash.finalize());
                let object = format!("{digest}.raw");
                let target = out.join(&object);
                // Content-addressed output must be exclusively owned; never overwrite an object.
                match fs::symlink_metadata(&target) {
                    Ok(metadata) if metadata.is_file() => {
                        let (existing, bytes) = crate::pack::digest_file(&target)?;
                        if existing != digest || bytes != length {
                            return Err(Error::Integrity("existing extent object".into()));
                        }
                    }
                    Ok(_) => {
                        return Err(Error::Invalid("extent object is not a regular file".into()));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        fs::rename(&temporary, &target)?
                    }
                    Err(error) => return Err(error.into()),
                }
                Ok(Extent {
                    entry,
                    offset,
                    length,
                    object,
                    object_offset: 0,
                    object_size: length,
                    sha256: digest,
                })
            })();
            let _ = fs::remove_file(&temporary);
            manifest.extents.push(result?);
            offset += length;
        }
        let mut extra = [0_u8; 1];
        if input.read(&mut extra)? != 0 {
            return Err(Error::Integrity("source grew during extent packing".into()));
        }
        if format!("{:x}", file_hash.finalize()) != *sha256 {
            return Err(Error::Integrity(
                "source changed during extent packing".into(),
            ));
        }
    }
    manifest.content_id = manifest.content_identity()?;
    manifest.id = manifest.identity()?;
    manifest.validate()?;
    Ok(manifest)
}
