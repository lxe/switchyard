//! Snapshot only the selected writable tree; generic artifacts own archive semantics.
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use switchyard_artifacts::{Metadata, PackOptions};

const MAX_SOURCE_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const RECEIPT: &str = ".switchyard-cache-seed";

pub(super) struct Output(PathBuf);
impl Output {
    pub(super) fn chunks(&self) -> PathBuf {
        self.0.join("chunks")
    }
    fn create(parent: &Path) -> Result<Self> {
        real_parents(parent)?;
        fs::create_dir_all(parent)?;
        real_parents(parent)?;
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random).map_err(|_| Error::Task)?;
        let nonce = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let path = parent.join(format!(".export-{nonce}"));
        fs::create_dir(&path)?;
        let output = Self(path);
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&output.0, fs::Permissions::from_mode(0o700))?;
        Ok(output)
    }
}
impl Drop for Output {
    fn drop(&mut self) {
        let _ = remove_private_tree(&self.0);
    }
}
fn remove_private_tree(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fn writable_directories(path: &Path) -> std::io::Result<()> {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.is_dir() {
            fs::set_permissions(
                path,
                fs::Permissions::from_mode(metadata.permissions().mode() | 0o700),
            )?;
            for entry in fs::read_dir(path)? {
                writable_directories(&entry?.path())?;
            }
        }
        Ok(())
    }
    writable_directories(path)?;
    fs::remove_dir_all(path)
}
struct Cancellation(Arc<AtomicBool>);
impl Drop for Cancellation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
fn check(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Task);
    }
    Ok(())
}
fn real_parents(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(metadata) if !metadata.is_dir() => {
                return Err(Error::Manifest(
                    "export parent is not a real directory".into(),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

pub(super) async fn pack(
    source: PathBuf,
    scratch: PathBuf,
    compatibility: BTreeMap<String, String>,
) -> Result<(Output, switchyard_artifacts::Manifest)> {
    pack_with(source, scratch, compatibility, |_| {}).await
}

async fn pack_with(
    source: PathBuf,
    scratch: PathBuf,
    compatibility: BTreeMap<String, String>,
    prepared: impl FnOnce(&Path) + Send + 'static,
) -> Result<(Output, switchyard_artifacts::Manifest)> {
    let cancel = Cancellation(Arc::new(AtomicBool::new(false)));
    let cancelled = cancel.0.clone();
    // The worker only returns private output ownership. An aborted waiter cannot publish it.
    tokio::task::spawn_blocking(move || {
        check(&cancelled)?;
        real_parents(&source)?;
        if !fs::symlink_metadata(&source)?.is_dir() {
            return Err(Error::Manifest("export source is not a directory".into()));
        }
        let output = Output::create(&scratch)?;
        let snapshot = output.0.join("snapshot");
        fs::create_dir(&snapshot)?;
        let mut bytes = 0;
        snapshot_tree(&source, &source, &snapshot, &mut bytes, &cancelled)?;
        if bytes == 0 {
            return Err(Error::Manifest(
                "export cache contains no compiled files".into(),
            ));
        }
        check(&cancelled)?;
        let manifest = switchyard_artifacts::pack(
            &snapshot,
            &output.chunks(),
            Metadata {
                kind: "cache".into(),
                compatibility,
                attributes: BTreeMap::new(),
            },
            PackOptions::default(),
        )?;
        switchyard_artifacts::verify(&snapshot, &manifest)?;
        for chunk in &manifest.chunks {
            check(&cancelled)?;
            let mut file = fs::File::open(output.chunks().join(&chunk.name))?;
            let mut digest = Sha256::new();
            let mut buffer = [0_u8; 64 * 1024];
            let mut length = 0;
            loop {
                check(&cancelled)?;
                let read = file.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                length += read as u64;
                digest.update(&buffer[..read]);
            }
            if length != chunk.size || format!("{:x}", digest.finalize()) != chunk.sha256 {
                return Err(Error::Task);
            }
        }
        remove_private_tree(&snapshot)?;
        check(&cancelled)?;
        prepared(&output.0);
        Ok((output, manifest))
    })
    .await
    .map_err(|_| Error::Task)?
}

fn snapshot_tree(
    root: &Path,
    source: &Path,
    destination: &Path,
    bytes: &mut u64,
    cancelled: &AtomicBool,
) -> Result<()> {
    for item in fs::read_dir(source)? {
        check(cancelled)?;
        let item = item?;
        let path = item.path();
        let target = destination.join(item.file_name());
        let meta = fs::symlink_metadata(&path)?;
        if source == root && item.file_name() == RECEIPT {
            if !meta.is_file()
                || meta.len() != 64
                || !fs::read(&path)?.iter().all(u8::is_ascii_hexdigit)
            {
                return Err(Error::Manifest("invalid writable cache receipt".into()));
            }
            continue;
        }
        if meta.is_dir() {
            fs::create_dir(&target)?;
            snapshot_tree(root, &path, &target, bytes, cancelled)?;
            fs::set_permissions(target, meta.permissions())?;
        } else if meta.is_file() {
            let mut input = fs::File::open(&path)?;
            let mut output = fs::File::options()
                .write(true)
                .create_new(true)
                .open(&target)?;
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                check(cancelled)?;
                let read = input.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                *bytes = bytes
                    .checked_add(read as u64)
                    .filter(|size| *size <= MAX_SOURCE_BYTES)
                    .ok_or_else(|| Error::Manifest("cache export size limit exceeded".into()))?;
                output.write_all(&buffer[..read])?;
            }
            fs::set_permissions(target, meta.permissions())?;
        } else if meta.is_symlink() {
            std::os::unix::fs::symlink(fs::read_link(path)?, target)?;
        } else {
            return Err(Error::Manifest(
                "cache export contains a special file".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn cancelled_waiter_discards_completed_private_output() {
        let mut random = [0_u8; 8];
        getrandom::fill(&mut random).unwrap();
        let root = std::env::temp_dir().join(format!(
            "switchyard-export-cancel-{:x}",
            u64::from_le_bytes(random)
        ));
        let source = root.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("kernel.bin"), "kernel").unwrap();
        let scratch = root.join("exports");
        let (prepared, reached) = tokio::sync::oneshot::channel();
        let (release, paused) = std::sync::mpsc::channel();
        let task = tokio::spawn(pack_with(source, scratch, BTreeMap::new(), move |path| {
            let _ = prepared.send(path.to_owned());
            let _ = paused.recv();
        }));
        let output = reached.await.unwrap();
        assert!(output.exists());
        task.abort();
        let _ = task.await;
        release.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while output.exists() {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
