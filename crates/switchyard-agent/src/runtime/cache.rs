//! Publish writable compiler caches from verified immutable artifact seeds.
use crate::{Error, Result};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use switchyard_artifacts::{Entry, Manifest};

const RECEIPT: &str = ".switchyard-cache-seed";
static STAGING_ID: AtomicU64 = AtomicU64::new(0);

struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Cancellation(Arc<AtomicBool>);
impl Drop for Cancellation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Transfer(
            "runtime cache preparation was cancelled".into(),
        ));
    }
    Ok(())
}

fn ancestors(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(meta) if !meta.is_dir() => {
                return Err(Error::Manifest(
                    "cache parent is not a real directory".into(),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

pub(super) fn prepare_root(path: &Path) -> Result<()> {
    ancestors(path)?;
    fs::create_dir_all(path)?;
    ancestors(path)
}

/// Only the awaiting caller publishes a completed tree. A cancelled caller drops its staging
/// ownership, including when the blocking worker finishes after its final cancellation check.
pub(super) async fn writable_copy_mode(
    seed: PathBuf,
    destination: PathBuf,
    manifest: Manifest,
    mode: switchyard_artifacts::HydrationMode,
) -> Result<()> {
    writable_copy_with(seed, destination, manifest, mode, || {}).await
}

async fn writable_copy_with(
    seed: PathBuf,
    destination: PathBuf,
    manifest: Manifest,
    mode: switchyard_artifacts::HydrationMode,
    prepared: impl FnOnce() + Send + 'static,
) -> Result<()> {
    let cancellation = Cancellation(Arc::new(AtomicBool::new(false)));
    let cancelled = cancellation.0.clone();
    let target = destination.clone();
    let staging = tokio::task::spawn_blocking(move || {
        let staging = copy(&seed, &target, &manifest, &cancelled, mode)?;
        prepared();
        Ok::<_, Error>(staging)
    })
    .await
    .map_err(|_| Error::Task)??;
    if let Some(staging) = staging {
        ancestors(
            destination
                .parent()
                .ok_or_else(|| Error::Manifest("invalid cache root".into()))?,
        )?;
        match fs::symlink_metadata(&destination) {
            Ok(_) => {
                return Err(Error::Manifest(
                    "writable cache destination appeared during preparation".into(),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        // There is no await between accepting the worker's staging ownership and publication.
        fs::rename(&staging.0, &destination)?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) async fn writable_copy_paused(
    seed: PathBuf,
    destination: PathBuf,
    manifest: Manifest,
    prepared: tokio::sync::oneshot::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
) -> Result<()> {
    writable_copy_with(
        seed,
        destination,
        manifest,
        switchyard_artifacts::HydrationMode::Verified,
        move || {
            let _ = prepared.send(());
            let _ = release.recv();
        },
    )
    .await
}

fn copy(
    seed: &Path,
    destination: &Path,
    manifest: &Manifest,
    cancelled: &AtomicBool,
    mode: switchyard_artifacts::HydrationMode,
) -> Result<Option<Staging>> {
    check_cancelled(cancelled)?;
    if mode == switchyard_artifacts::HydrationMode::Verified {
        switchyard_artifacts::verify(seed, manifest)?;
    } else {
        // Hydration produced this exclusively owned seed; only containment is rechecked.
        ancestors(seed)?;
    }
    check_cancelled(cancelled)?;
    if manifest.entries.iter().any(|entry| entry.path() == RECEIPT) {
        return Err(Error::Manifest(
            "cache artifact uses a reserved receipt path".into(),
        ));
    }
    ancestors(
        destination
            .parent()
            .ok_or_else(|| Error::Manifest("invalid cache root".into()))?,
    )?;
    match fs::symlink_metadata(destination) {
        Ok(meta) => {
            let receipt = destination.join(RECEIPT);
            if !meta.is_dir()
                || !fs::symlink_metadata(&receipt)?.is_file()
                || fs::metadata(&receipt)?.len() != 64
                || fs::read_to_string(receipt)? != manifest.id
            {
                return Err(Error::Manifest(
                    "writable cache has a different or missing seed identity".into(),
                ));
            }
            // The engine may have added or replaced compiled files since the seed was copied.
            return Ok(None);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let parent = destination.parent().unwrap();
    fs::create_dir_all(parent)?;
    let staging = Staging(parent.join(format!(
        ".cache-staging-{}-{}",
        std::process::id(),
        STAGING_ID.fetch_add(1, Ordering::Relaxed)
    )));
    fs::create_dir(&staging.0)?;
    for entry in &manifest.entries {
        check_cancelled(cancelled)?;
        if let Entry::Directory { path, .. } = entry {
            ancestors(&seed.join(path))?;
            fs::create_dir(staging.0.join(path))?;
        }
    }
    for entry in &manifest.entries {
        check_cancelled(cancelled)?;
        if let Entry::File { path, mode, .. } = entry {
            let source = seed.join(path);
            ancestors(source.parent().unwrap())?;
            if !fs::symlink_metadata(&source)?.is_file() {
                return Err(Error::Manifest(
                    "cache seed file is not a regular file".into(),
                ));
            }
            let copy = staging.0.join(path);
            fs::copy(source, &copy)?;
            writable_mode(&copy, *mode)?;
        }
    }
    for entry in &manifest.entries {
        check_cancelled(cancelled)?;
        if let Entry::Symlink { path, target, .. } = entry {
            ancestors(seed.join(path).parent().unwrap())?;
            if fs::read_link(seed.join(path))? != Path::new(target) {
                return Err(Error::Manifest(
                    "cache seed link target differs from its manifest".into(),
                ));
            }
            #[cfg(unix)]
            std::os::unix::fs::symlink(target, staging.0.join(path))?;
            #[cfg(not(unix))]
            return Err(Error::Manifest(
                "runtime cache symlinks require Unix".into(),
            ));
        }
    }
    for entry in manifest.entries.iter().rev() {
        check_cancelled(cancelled)?;
        if let Entry::Directory { path, mode } = entry {
            writable_mode(&staging.0.join(path), *mode | 0o100)?;
        }
    }
    let receipt = staging.0.join(RECEIPT);
    let mut file = fs::File::options()
        .create_new(true)
        .write(true)
        .open(&receipt)?;
    file.write_all(manifest.id.as_bytes())?;
    if mode == switchyard_artifacts::HydrationMode::Verified {
        file.sync_all()?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(receipt, fs::Permissions::from_mode(0o600))?;
    }
    check_cancelled(cancelled)?;
    Ok(Some(staging))
}

fn writable_mode(path: &Path, mode: u32) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(mode | 0o200))?;
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
    }
    Ok(())
}
