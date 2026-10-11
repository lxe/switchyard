//! Observe serving-host properties separately from generic archive verification.
use super::RuntimeTicket;
use crate::{Error, Result};
use std::time::Duration;
use tokio::process::Command;

pub(super) async fn observe_and_check(runtime: &RuntimeTicket) -> Result<()> {
    let needs_gpu =
        requirements(runtime).any(|labels| labels.contains_key("gpu_compute_capability"));
    let capabilities = if needs_gpu {
        let mut command = Command::new("nvidia-smi");
        command
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .kill_on_drop(true);
        super::child_environment(&mut command);
        let output = tokio::time::timeout(Duration::from_secs(5), command.output())
            .await
            .map_err(|_| Error::Manifest("GPU capability probe timed out".into()))??;
        if !output.status.success() || output.stdout.len() > 4096 {
            return Err(Error::Manifest("GPU capability probe failed".into()));
        }
        let text = std::str::from_utf8(&output.stdout)
            .map_err(|_| Error::Manifest("GPU capability probe returned invalid data".into()))?;
        let values = text.lines().map(normalize).collect::<Result<Vec<_>>>()?;
        if values.is_empty() {
            return Err(Error::Manifest("GPU capability probe found no GPUs".into()));
        }
        values
    } else {
        Vec::new()
    };
    check(runtime, std::env::consts::ARCH, &capabilities)
}

pub(super) fn check(
    runtime: &RuntimeTicket,
    architecture: &str,
    capabilities: &[String],
) -> Result<()> {
    for requirements in requirements(runtime) {
        if requirements
            .get("architecture")
            .is_some_and(|value| value != architecture)
        {
            return Err(Error::Manifest(
                "artifact architecture differs from the host".into(),
            ));
        }
        if let Some(value) = requirements.get("gpu_compute_capability") {
            let expected = normalize(value)?;
            if capabilities.is_empty() || capabilities.iter().any(|actual| actual != &expected) {
                return Err(Error::Manifest(
                    "artifact GPU compute capability differs from the host".into(),
                ));
            }
        }
    }
    Ok(())
}

fn requirements(
    runtime: &RuntimeTicket,
) -> impl Iterator<Item = &std::collections::BTreeMap<String, String>> {
    std::iter::once(&runtime.artifact.manifest)
        .chain(runtime.caches.iter().map(|cache| &cache.artifact.manifest))
        .map(|manifest| &manifest.metadata.compatibility)
        .chain(
            runtime
                .cache_export
                .iter()
                .map(|export| &export.compatibility),
        )
}

fn normalize(value: &str) -> Result<String> {
    let (major, minor) = value
        .trim()
        .split_once('.')
        .ok_or_else(|| Error::Manifest("invalid GPU compute capability".into()))?;
    let major = major
        .parse::<u16>()
        .map_err(|_| Error::Manifest("invalid GPU compute capability".into()))?;
    let minor = minor
        .parse::<u16>()
        .map_err(|_| Error::Manifest("invalid GPU compute capability".into()))?;
    Ok(format!("{major}.{minor}"))
}
