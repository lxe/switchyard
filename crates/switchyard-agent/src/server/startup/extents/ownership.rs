//! Stable loader paths protected by kernel leases and bounded ownership records.
use super::*;
use nix::fcntl::{FcntlArg, FdFlag, OFlag, fcntl};
use serde::{Deserialize, Serialize};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    identity: String,
    boot: String,
    namespace: String,
    group: Option<i32>,
    owner_pid: u32,
    owner_start: u64,
    lease_ready: bool,
}
fn context(identity: String) -> Result<Record> {
    Ok(Record {
        identity,
        boot: fs::read_to_string("/proc/sys/kernel/random/boot_id")?
            .trim()
            .into(),
        namespace: fs::read_link("/proc/self/ns/pid")?
            .to_string_lossy()
            .into_owned(),
        group: None,
        owner_pid: std::process::id(),
        owner_start: process_start(std::process::id())?.ok_or(Error::Task)?,
        lease_ready: false,
    })
}
fn process_start(pid: u32) -> Result<Option<u64>> {
    let stat = match fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    stat.rsplit_once(')')
        .and_then(|(_, rest)| rest.split_whitespace().nth(19))
        .and_then(|field| field.parse().ok())
        .map(Some)
        .ok_or_else(|| Error::Transfer("loader owner process identity is invalid".into()))
}
fn record_path(root: &Path) -> PathBuf {
    root.with_file_name(format!(
        "{}.owner",
        root.file_name().unwrap().to_string_lossy()
    ))
}
fn record(root: &Path) -> Result<Record> {
    let mut bytes = Vec::new();
    fs::File::options()
        .read(true)
        .custom_flags(OFlag::O_NOFOLLOW.bits())
        .open(record_path(root))?
        .take(4097)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 4096 {
        return Err(Error::Transfer(
            "loader ownership record exceeds limit".into(),
        ));
    }
    let record: Record = serde_json::from_slice(&bytes)
        .map_err(|_| Error::Transfer("loader ownership record is invalid".into()))?;
    let boot_valid = record.boot.len() == 36
        && record.boot.bytes().enumerate().all(|(i, b)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        });
    let namespace_valid = record
        .namespace
        .strip_prefix("pid:[")
        .and_then(|s| s.strip_suffix(']'))
        .and_then(|s| s.parse::<u64>().ok())
        .is_some_and(|id| id > 0);
    if !boot_valid
        || !namespace_valid
        || record.identity.len() != 64
        || !record.identity.bytes().all(|b| b.is_ascii_hexdigit())
        || record.owner_pid == 0
        || record.owner_start == 0
        || record.group.is_some_and(|group| group <= 0)
    {
        return Err(Error::Transfer(
            "loader ownership identity is invalid".into(),
        ));
    }
    Ok(record)
}
fn write_record(root: &Path, record: &Record) -> Result<()> {
    let temporary = record_path(root).with_extension("owner.pending");
    let bytes = serde_json::to_vec(record).map_err(|_| Error::Task)?;
    let mut file = fs::File::options()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(OFlag::O_NOFOLLOW.bits())
        .open(&temporary)?;
    file.write_all(&bytes)?;
    fs::rename(temporary, record_path(root))?;
    Ok(())
}
fn inactive(previous: &Record, current: &Record) -> Result<bool> {
    if previous.identity != current.identity {
        return Ok(false);
    }
    if previous.boot != current.boot {
        return Ok(true);
    }
    if previous.namespace != current.namespace {
        // A process outside this PID namespace may still access this directory.
        return Ok(false);
    }
    if !previous.lease_ready
        && let Some(start) = process_start(previous.owner_pid)?
    {
        if start != previous.owner_start {
            return Err(Error::Transfer("loader owner PID was reused".into()));
        }
        // flock conversion can release an exclusive lock before obtaining a
        // shared lock. The live creator protects this short publication gap.
        return Ok(false);
    }
    previous.group.map_or(Ok(true), |group| {
        engine::live_group(nix::unistd::Pid::from_raw(group)).map(|live| !live)
    })
}

pub(super) struct Sidecar {
    pub(super) root: PathBuf,
    lock_path: PathBuf,
    lease: fs::File,
    current: std::sync::Mutex<Record>,
}
impl Sidecar {
    pub(super) fn group(&self, pid: u32) -> Result<()> {
        let mut current = self.current.lock().map_err(|_| Error::Task)?;
        current.group = Some(i32::try_from(pid).map_err(|_| Error::Task)?);
        // Register before releasing the metadata gate, hence before worker creation.
        write_record(&self.root, &current)?;
        Ok(())
    }
    pub(super) fn spawn(&self, command: &mut Command) -> Result<engine::Process> {
        command.env("SWITCHYARD_LOADER_LEASE", &self.lock_path);
        let flags = FdFlag::from_bits_truncate(
            fcntl(&self.lease, FcntlArg::F_GETFD).map_err(std::io::Error::from)?,
        );
        // The initial engine retains the lease even if its agent dies before
        // group registration. The descriptor carries no credentials or data.
        fcntl(&self.lease, FcntlArg::F_SETFD(flags & !FdFlag::FD_CLOEXEC))
            .map_err(std::io::Error::from)?;
        let spawned = engine::Process::spawn(command);
        let restored = fcntl(&self.lease, FcntlArg::F_SETFD(flags)).map_err(std::io::Error::from);
        let process = spawned?;
        self.group(process.child.id().ok_or(Error::Task)?)?;
        restored?;
        Ok(process)
    }
}
impl Drop for Sidecar {
    fn drop(&mut self) {
        // Never unlink a live child's files. Kill-on-drop is asynchronous; an
        // abandoned directory can instead be reclaimed by the next owner.
        let Ok(current) = self.current.lock() else {
            return;
        };
        let group_gone = current
            .group
            .map_or(Ok(true), |group| {
                engine::live_group(nix::unistd::Pid::from_raw(group)).map(|live| !live)
            })
            .unwrap_or(false);
        if group_gone && self.lease.try_lock().is_ok() {
            let _ = fs::remove_dir_all(&self.root);
        }
        // Keep the lock inode stable: unlinking it would split competing leases.
    }
}

pub(super) fn sidecar(
    root: &Path,
    content_id: &str,
    loader: &switchyard_weights::VllmLoaderConfig,
) -> Result<Arc<Sidecar>> {
    use sha2::{Digest, Sha256};
    let identity = format!(
        "{:x}",
        Sha256::digest(format!(
            "switchyard-loader-path-v1:{content_id}:{}",
            loader.identity()?
        ))
    );
    let name = format!(".model-loader-{}", &identity[..24]);
    let lock_path = root.join(format!("{name}.lock"));
    let lease = fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(OFlag::O_NOFOLLOW.bits())
        .open(&lock_path)?;
    lease
        .try_lock()
        .map_err(|_| Error::Transfer("loader path has an active lease".into()))?;
    let root = root.join(name);
    let current = context(identity)?;
    if root.try_exists()? {
        if !root.symlink_metadata()?.is_dir() || !inactive(&record(&root)?, &current)? {
            return Err(Error::Transfer(
                "loader path ownership is live or unknown".into(),
            ));
        }
        fs::remove_dir_all(&root)?;
    }
    // Publish proof before creating the directory, so a crash during initial
    // construction also leaves enough information for safe reclamation.
    write_record(&root, &current)?;
    fs::create_dir(&root)?;
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
    let owner = Arc::new(Sidecar {
        root,
        lock_path,
        lease,
        current: std::sync::Mutex::new(current),
    });
    fs::create_dir(owner.root.join("metadata"))?;
    write_vllm_bundle(&owner.root.join("plugin"))?;
    // Worker processes acquire independent shared leases before importing Torch.
    owner.lease.lock_shared()?;
    {
        let mut current = owner.current.lock().map_err(|_| Error::Task)?;
        current.lease_ready = true;
        write_record(&owner.root, &current)?;
    }
    Ok(owner)
}

#[cfg(test)]
mod tests;
