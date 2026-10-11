//! Startup policy and measured model-storage capacity, independent of cloud providers.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Component, Path, PathBuf},
};
pub use switchyard_artifacts::HydrationMode;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartupPolicy {
    #[serde(default)]
    pub mode: HydrationMode,
    /// Parent of the published model directory. None preserves /work/switchyard.
    #[serde(default)]
    pub model_root: Option<String>,
    #[serde(default)]
    pub require_model_ram: bool,
    /// A cooperative program waits for the agent's fixed model-ready line on stdin.
    #[serde(default)]
    pub early_launch: bool,
}
impl StartupPolicy {
    pub fn validate(&self) -> Result<()> {
        if let Some(root) = &self.model_root {
            let path = Path::new(root);
            if !path.is_absolute()
                || root.contains(['\\', '\0'])
                || !path
                    .components()
                    .all(|part| matches!(part, Component::RootDir | Component::Normal(_)))
                || path
                    .components()
                    .filter(|part| matches!(part, Component::Normal(_)))
                    .count()
                    == 0
            {
                return Err(Error::Manifest("invalid model storage root".into()));
            }
        }
        Ok(())
    }
    pub fn model_directory(&self) -> PathBuf {
        Path::new(self.model_root.as_deref().unwrap_or("/work/switchyard")).join("model")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupEvent {
    ModelMetadataReady,
    TensorLoadingFinished,
    RuntimeHydrationStarted,
    RuntimeHydrationFinished,
    RuntimeBootReady,
    EngineGateReleased,
    ModelHydrationStarted,
    ModelHydrationFinished,
    EngineSpawned,
    ModelGateReleased,
    InferenceVerified,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupMilestone {
    pub event: StartupEvent,
    pub elapsed_s: f64,
}
pub(crate) async fn milestone(
    state: &tokio::sync::RwLock<crate::State>,
    epoch: std::time::Instant,
    event: StartupEvent,
) {
    state
        .write()
        .await
        .startup_milestones
        .push(StartupMilestone {
            event,
            elapsed_s: epoch.elapsed().as_secs_f64(),
        });
}

/// Filesystem and memory observations are measured before transfer, not hardware claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStorageObservation {
    pub root: String,
    pub filesystem: String,
    pub mount_point: Option<String>,
    pub source: Option<String>,
    pub ram_backed: bool,
    pub available_bytes: u64,
    pub required_bytes: u64,
    pub memory_available_bytes: Option<u64>,
    pub cgroup_available_bytes: Option<u64>,
}

pub(crate) fn observe(root: &Path, required_bytes: u64) -> Result<ModelStorageObservation> {
    for path in root.ancestors() {
        match fs::symlink_metadata(path) {
            Ok(metadata) if !metadata.is_dir() => {
                return Err(Error::Manifest(
                    "model root contains an unsafe parent".into(),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    fs::create_dir_all(root)?;
    let stat = nix::sys::statvfs::statvfs(root)
        .map_err(|_| Error::Transfer("model storage capacity probe failed".into()))?;
    let available_bytes = stat
        .blocks_available()
        .checked_mul(stat.fragment_size())
        .ok_or_else(|| Error::Manifest("model storage capacity overflow".into()))?;
    let statfs = nix::sys::statfs::statfs(root)
        .map_err(|_| Error::Transfer("model storage type probe failed".into()))?;
    let ram_backed = matches!(statfs.filesystem_type().0, 0x0102_1994 | 0x8584_58f6);
    let mount = fs::read_to_string("/proc/self/mountinfo")
        .ok()
        .and_then(|text| mount_for(root, &text));
    let memory_available_bytes = fs::read_to_string("/proc/meminfo").ok().and_then(|text| {
        text.lines().find_map(|line| {
            line.strip_prefix("MemAvailable:").and_then(|value| {
                value
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)
            })
        })
    });
    let cgroup_available_bytes =
        remaining_memory("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current").or_else(
            || {
                remaining_memory(
                    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                )
            },
        );
    Ok(ModelStorageObservation {
        root: root.to_string_lossy().into_owned(),
        filesystem: mount.as_ref().map_or_else(
            || format!("magic:{:#x}", statfs.filesystem_type().0),
            |(_, filesystem, _)| filesystem.clone(),
        ),
        mount_point: mount.as_ref().map(|(point, _, _)| point.clone()),
        source: mount.map(|(_, _, source)| source),
        ram_backed,
        available_bytes,
        required_bytes,
        memory_available_bytes,
        cgroup_available_bytes,
    })
}
fn remaining_memory(limit: &str, current: &str) -> Option<u64> {
    let limit = fs::read_to_string(limit).ok()?.trim().parse::<u64>().ok()?;
    let current = fs::read_to_string(current)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    Some(limit.saturating_sub(current))
}
fn unescape_mount(value: &str) -> String {
    value
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}
fn mount_for(root: &Path, text: &str) -> Option<(String, String, String)> {
    text.lines()
        .filter_map(|line| {
            let (before, after) = line.split_once(" - ")?;
            let point = unescape_mount(before.split_whitespace().nth(4)?);
            if !root.starts_with(&point) {
                return None;
            }
            let mut after = after.split_whitespace();
            Some((
                point,
                after.next()?.to_owned(),
                unescape_mount(after.next()?),
            ))
        })
        .max_by_key(|(point, _, _)| point.len())
}

pub(crate) fn check_capacity(
    observation: &ModelStorageObservation,
    require_ram: bool,
    buffer_bytes: u64,
) -> Result<()> {
    if require_ram && !observation.ram_backed {
        return Err(Error::Manifest("model storage must be RAM-backed".into()));
    }
    let required = observation
        .required_bytes
        .checked_add(buffer_bytes)
        .ok_or_else(|| Error::Manifest("model memory requirement overflow".into()))?;
    if observation.available_bytes < observation.required_bytes {
        return Err(Error::Transfer(
            "model storage has insufficient free capacity".into(),
        ));
    }
    if observation.ram_backed
        && [
            observation.memory_available_bytes,
            observation.cgroup_available_bytes,
        ]
        .into_iter()
        .flatten()
        .any(|available| available < required)
    {
        return Err(Error::Transfer(
            "model RAM or cgroup capacity is insufficient".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mount_probe_uses_longest_real_mount_and_reports_source_without_device_claims() {
        let mounts = "1 0 1:1 / / rw - overlay overlay rw\n2 1 0:2 / /dev/shm rw - tmpfs shm rw\n";
        assert_eq!(
            mount_for(Path::new("/dev/shm/switchyard"), mounts),
            Some(("/dev/shm".into(), "tmpfs".into(), "shm".into()))
        );
    }
    #[test]
    fn capacity_rejects_tiny_shm_and_cgroup_limits() {
        let mut observation = ModelStorageObservation {
            root: "test".into(),
            filesystem: "tmpfs".into(),
            mount_point: None,
            source: None,
            ram_backed: true,
            available_bytes: 64,
            required_bytes: 128,
            memory_available_bytes: Some(4096),
            cgroup_available_bytes: None,
        };
        assert!(check_capacity(&observation, true, 0).is_err());
        observation.available_bytes = 4096;
        observation.cgroup_available_bytes = Some(64);
        assert!(check_capacity(&observation, true, 0).is_err());
        observation.cgroup_available_bytes = Some(4096);
        check_capacity(&observation, true, 64).unwrap();
        observation.ram_backed = false;
        assert!(check_capacity(&observation, true, 0).is_err());
    }
}
