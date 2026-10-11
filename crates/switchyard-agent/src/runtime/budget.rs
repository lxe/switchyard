//! Capacity accounting for simultaneous model, runtime, spool and decoder allocations.
use super::RootPaths;
use crate::{Error, Result, Ticket, startup};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, time::Instant};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyBudget {
    pub model_bytes: u64,
    /// New model/sidecar output required by this boot, excluding trusted reuse.
    #[serde(default)]
    pub model_output_required_bytes: u64,
    pub runtime_output_bytes: u64,
    pub compressed_backing_bytes: u64,
    pub decoder_history_bound_bytes: u64,
    pub loader_staging_bytes: u64,
    pub required_ram_bytes: u64,
    pub preflight_seconds: f64,
}
pub(super) fn check(
    ticket: &Ticket,
    roots: &RootPaths,
    model_required: Option<u64>,
) -> Result<DependencyBudget> {
    use std::os::unix::fs::MetadataExt;
    let epoch = Instant::now();
    let model_bytes = match &ticket.source {
        crate::model::ModelSource::Legacy(s) => s.manifest.files.values().map(|f| f.size).sum(),
        crate::model::ModelSource::Extents(_) => {
            let (a, _) = ticket.source.extent().unwrap();
            a.manifest
                .total_bytes()?
                .checked_add(crate::model::metadata_bytes(&a.manifest)?)
                .ok_or_else(|| Error::Manifest("model capacity overflow".into()))?
        }
    };
    let model_output_required_bytes = model_required.unwrap_or(model_bytes);
    let runtime = ticket.runtime.as_ref();
    let file_bytes = |m: &switchyard_artifacts::Manifest| {
        m.entries
            .iter()
            .map(|e| match e {
                switchyard_artifacts::Entry::File { size, .. } => *size,
                _ => 0,
            })
            .try_fold(0_u64, |sum, n| {
                sum.checked_add(n)
                    .ok_or_else(|| Error::Manifest("runtime capacity overflow".into()))
            })
    };
    let boot_bytes = runtime
        .map(|r| file_bytes(&r.artifact.manifest))
        .transpose()?
        .unwrap_or(0);
    let component_bytes = runtime
        .map(|r| {
            r.components.iter().try_fold(0_u64, |sum, part| {
                sum.checked_add(file_bytes(&part.artifact.manifest)?)
                    .ok_or_else(|| Error::Manifest("component capacity overflow".into()))
            })
        })
        .transpose()?
        .unwrap_or(0);
    let runtime_output_bytes = boot_bytes
        .checked_add(component_bytes)
        .ok_or_else(|| Error::Manifest("runtime capacity overflow".into()))?;
    let cache_bytes = runtime
        .map(|r| {
            r.caches.iter().try_fold(0_u64, |sum, c| {
                sum.checked_add(
                    file_bytes(&c.artifact.manifest)?
                        .checked_mul(2)
                        .ok_or_else(|| Error::Manifest("cache capacity overflow".into()))?,
                )
                .ok_or_else(|| Error::Manifest("cache capacity overflow".into()))
            })
        })
        .transpose()?
        .unwrap_or(0);
    // Trusted runtime shards feed decoders directly; only verified hydration
    // retains compressed chunks, including component and cache artifacts.
    let compressed_backing_bytes = runtime
        .filter(|_| ticket.startup.mode == switchyard_artifacts::HydrationMode::Verified)
        .map(|r| {
            std::iter::once(&r.artifact)
                .chain(r.components.iter().map(|c| &c.artifact))
                .chain(r.caches.iter().map(|c| &c.artifact))
                .try_fold(0_u64, |sum, a| {
                    a.manifest.chunks.iter().try_fold(sum, |sum, c| {
                        sum.checked_add(
                            c.size
                                .checked_add(4095)
                                .ok_or_else(|| Error::Manifest("spool capacity overflow".into()))?
                                / 4096
                                * 4096,
                        )
                        .ok_or_else(|| Error::Manifest("spool capacity overflow".into()))
                    })
                })
        })
        .transpose()?
        .unwrap_or(0);
    let decoder_history_bound_bytes = decoder_history_bound(ticket);
    let loader_staging_bytes = ticket
        .source
        .extent()
        .map_or(0, |(_, l)| l.pinned_bytes as u64);
    let mut ram = decoder_history_bound_bytes
        .checked_add(loader_staging_bytes)
        .and_then(|n| n.checked_add(ticket.concurrency as u64 * 64 * 1024 + 512 * 1024 * 1024))
        .ok_or_else(|| Error::Manifest("host memory capacity overflow".into()))?;
    let mut devices = BTreeMap::<u64, (u64, u64)>::new();
    let mut memory: Option<u64> = None;
    // Observing storage creates its directory; never create an unpublished boot destination.
    let boot_storage = if roots.runtime.exists() {
        roots.runtime.as_path()
    } else {
        roots.runtime.parent().unwrap()
    };
    let components_root = roots.components();
    for (path, bytes) in [
        (roots.models.as_path(), model_output_required_bytes),
        (boot_storage, boot_bytes),
        (components_root.as_path(), component_bytes),
        (roots.caches.as_path(), cache_bytes),
        (roots.chunks.as_path(), compressed_backing_bytes),
    ] {
        let observation = startup::observe(path, bytes)?;
        if observation.ram_backed {
            ram = ram
                .checked_add(bytes)
                .ok_or_else(|| Error::Manifest("host memory capacity overflow".into()))?;
        }
        let available = [
            observation.memory_available_bytes,
            observation.cgroup_available_bytes,
        ]
        .into_iter()
        .flatten()
        .min();
        memory = match (memory, available) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let device = fs::metadata(path)?.dev();
        let group = devices
            .entry(device)
            .or_insert((0, observation.available_bytes));
        group.0 = group
            .0
            .checked_add(bytes)
            .ok_or_else(|| Error::Manifest("filesystem capacity overflow".into()))?;
        group.1 = group.1.min(observation.available_bytes);
    }
    if devices
        .values()
        .any(|(required, available)| required > available)
    {
        return Err(Error::Transfer(
            "combined artifact outputs and reception backing exceed filesystem capacity".into(),
        ));
    }
    if memory.is_some_and(|available| ram > available) {
        return Err(Error::Transfer(
            "combined model, runtime, reception and loader memory exceeds host or cgroup capacity"
                .into(),
        ));
    }
    Ok(DependencyBudget {
        model_bytes,
        model_output_required_bytes,
        runtime_output_bytes,
        compressed_backing_bytes,
        decoder_history_bound_bytes,
        loader_staging_bytes,
        required_ram_bytes: ram,
        preflight_seconds: epoch.elapsed().as_secs_f64(),
    })
}

pub(super) fn decoder_history_bound(ticket: &Ticket) -> u64 {
    let runtime_workers = if ticket.runtime.is_some() {
        super::extraction_concurrency(ticket)
    } else {
        0
    };
    let cache_workers = ticket.runtime.as_ref().map_or(0, |r| {
        r.caches
            .iter()
            .map(|c| c.artifact.manifest.chunks.len())
            .max()
            .unwrap_or(0)
            .min(runtime_workers)
    });
    let component_workers = ticket.runtime.as_ref().map_or(0, |r| {
        r.components
            .iter()
            .map(|c| c.artifact.manifest.chunks.len())
            .max()
            .unwrap_or(0)
            .min(runtime_workers)
    });
    let model_workers = match &ticket.source {
        crate::model::ModelSource::Legacy(s) if s.manifest.compress == "zstd" => ticket
            .concurrency
            .min(
                s.manifest
                    .chunks
                    .iter()
                    .map(|c| c.shard)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
            )
            .min(
                if ticket.startup.mode == switchyard_artifacts::HydrationMode::TrustedStreaming {
                    20
                } else {
                    8
                },
            ),
        _ => 0,
    };
    (runtime_workers + component_workers + cache_workers + model_workers) as u64 * 128 * 1024 * 1024
}
