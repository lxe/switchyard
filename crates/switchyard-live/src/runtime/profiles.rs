//! Load immutable model inputs and construct explicitly assumed hardware profiles.
use crate::model::ModelManifest;
use crate::{
    Error, Result,
    config::{ExecutionConfig, ModelConfig},
    planner::Planner,
};
use std::{collections::BTreeMap, path::Path, sync::Arc};
use switchyard_core::{CapacityPoint, HostProfile, Profile, Startup, Workload};
use switchyard_provider::{Offering, Provider, storage::Storage};

pub(super) struct Prepared {
    pub(super) offerings: Vec<Offering>,
    pub(super) manifests: BTreeMap<String, ModelManifest>,
    pub(super) planners: BTreeMap<String, Arc<Planner>>,
    pub(super) profiles: Vec<Profile>,
    pub(super) profile_offerings: Vec<Offering>,
    pub(super) runtime_manifest: Option<switchyard_artifacts::Manifest>,
    pub(super) component_manifests: BTreeMap<String, switchyard_artifacts::Manifest>,
    pub(super) cache_manifests: BTreeMap<String, switchyard_artifacts::Manifest>,
}

pub(super) async fn prepare<P: Provider>(
    provider: &P,
    storage: &Storage,
    config: &ExecutionConfig,
    out: &Path,
) -> Result<Prepared> {
    let quoted = provider.offers().await?;
    let offerings = admit_offerings(quoted, config.limits.max_usd_per_hour, out)?;
    let mut manifests = BTreeMap::new();
    let mut planners = BTreeMap::new();
    let mut profiles = vec![];
    let mut profile_offerings = vec![];
    let runtime_manifest = if let Some(runtime) = &config.bootstrap.runtime {
        let manifest = runtime
            .artifact
            .resolve(storage, &runtime.compatibility)
            .await?;
        runtime.validate_manifest(&manifest)?;
        Some(manifest)
    } else {
        None
    };
    let mut component_manifests = BTreeMap::new();
    if let Some(runtime) = &config.bootstrap.runtime {
        for part in &runtime.components {
            let manifest = part.artifact.resolve(storage, &part.compatibility).await?;
            component_manifests.insert(part.name.clone(), manifest);
        }
    }
    let mut cache_manifests = BTreeMap::new();
    for model in &config.models {
        let manifest = model.artifact.resolve(storage).await?;
        let planner = Planner::load(model, &out.join("tokenizers")).await?;
        manifests.insert(model.id.clone(), manifest.clone());
        planners.insert(model.id.clone(), Arc::new(planner));
        for cache in &model.caches {
            let runtime = config
                .bootstrap
                .runtime
                .as_ref()
                .expect("validated runtime");
            let requirements = cache.requirements(runtime, model)?;
            let manifest = cache.artifact.resolve(storage, &requirements).await?;
            cache_manifests.insert(cache.artifact.id.clone(), manifest);
        }
        let first_profile = profiles.len();
        for offer in &offerings {
            if model.engine.native.as_ref().is_some_and(|bounds| {
                !native_hardware_admitted(offer, bounds.minimum_gpu_memory_bytes)
            }) {
                continue;
            }
            if !model.caches.is_empty() && !model.caches.iter().any(|c| c.gpu == offer.gpu) {
                continue;
            }
            let mut artifact_bytes = runtime_manifest
                .as_ref()
                .map_or(0, |m| m.chunks.iter().map(|c| c.size).sum::<u64>());
            for manifest in component_manifests.values() {
                artifact_bytes = artifact_bytes
                    .checked_add(manifest.chunks.iter().map(|c| c.size).sum::<u64>())
                    .ok_or_else(|| Error::Config("component size overflow".into()))?;
            }
            for cache in model.caches.iter().filter(|c| c.gpu == offer.gpu) {
                let bytes: u64 = cache_manifests[&cache.artifact.id]
                    .chunks
                    .iter()
                    .map(|c| c.size)
                    .sum();
                artifact_bytes = artifact_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| Error::Config("artifact size overflow".into()))?;
            }
            let mut prepared = profile(model, &manifest, offer, artifact_bytes);
            if let Some(measured) = model
                .prior
                .measured
                .iter()
                .find(|m| m.profile.id == prepared.id)
            {
                measured
                    .apply(
                        &mut prepared,
                        &crate::calibration::binding(config, model, &offer.gpu)?,
                    )
                    .map_err(Error::Config)?;
            }
            profiles.push(prepared);
            profile_offerings.push(offer.clone());
        }
        if profiles.len() == first_profile {
            return Err(Error::Config(format!(
                "model {} has no available GPU with its configured serving caches",
                model.id
            )));
        }
    }
    if profiles.is_empty() {
        return Err(Error::Config(
            "no available GPU has the configured serving caches".into(),
        ));
    }
    let bindings = profiles
        .iter()
        .map(|p| {
            let model = config
                .models
                .iter()
                .find(|m| m.id == p.model)
                .expect("prepared model");
            Ok((
                p.id.clone(),
                crate::calibration::binding(config, model, &p.gpu)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    std::fs::write(
        out.join("profile-bindings.json"),
        serde_json::to_vec_pretty(&bindings)?,
    )?;
    Ok(Prepared {
        offerings,
        manifests,
        planners,
        profiles,
        profile_offerings,
        runtime_manifest,
        component_manifests,
        cache_manifests,
    })
}

/// Persist admission evidence before returning an early capacity failure.
fn admit_offerings(
    quoted: Vec<Offering>,
    max_hourly_usd: f64,
    out: &Path,
) -> Result<Vec<Offering>> {
    let admitted: Vec<_> = quoted
        .iter()
        .filter(|o| o.total_usd_per_hour() <= max_hourly_usd)
        .cloned()
        .collect();
    std::fs::write(
        out.join("offering-admission.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "quoted": quoted,
            "admitted": admitted,
            "max_hourly_usd": max_hourly_usd,
            "observed_at_unix_s": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
            "scope": "Provider offerings and hourly admission; no allocation performed"
        }))?,
    )?;
    if admitted.is_empty() {
        return Err(switchyard_provider::Error::NoCapacity.into());
    }
    Ok(admitted)
}

/// Use the configured priors without presenting them as measured GPU performance.
fn profile(
    model: &ModelConfig,
    manifest: &ModelManifest,
    offering: &Offering,
    artifact_bytes: u64,
) -> Profile {
    let prior = &model.prior;
    let mut curve = vec![CapacityPoint {
        in_flight: 1,
        work_units_per_s: prior.single_output_tps,
    }];
    if model.engine.max_num_seqs > 1 {
        curve.push(CapacityPoint {
            in_flight: model.engine.max_num_seqs,
            work_units_per_s: prior.aggregate_output_tps,
        });
    }
    Profile {
        id: format!("{}:{}:{}", model.id, offering.gpu, offering.data_center),
        model: model.id.clone(),
        workload: Workload::Text,
        gpu: offering.gpu.clone(),
        usd_per_hour: offering.total_usd_per_hour(),
        startup: Startup {
            allocation_s: prior.startup_s * 0.2,
            container_s: prior.startup_s * 0.4,
            weights_s: 0.0,
            engine_s: prior.startup_s * 0.4,
            jitter: 0.25,
            restore_samples_s: vec![],
            build_samples_s: vec![],
        },
        network_s: 0.05,
        network_bytes_per_s: 100_000_000.0,
        max_in_flight: model.engine.max_num_seqs,
        kv_tokens: prior.kv_tokens,
        input_work_per_token: prior.input_work_per_token,
        image_work_per_megapixel: 0.0,
        adapter_load_s: 0.0,
        multi_adapter_factor: 1.0,
        adapter_cache_slots: 0,
        minimum_bill_s: 1.0,
        curve,
        evidence: "assumed capacity and startup; timestamped provider catalog quote with configured disk rent".into(),
        source: "deployment configuration; live receipts preserve provider quotes and actual outcomes".into(),
        host: HostProfile {
            model_bytes: manifest.total_size().saturating_add(artifact_bytes),
            // Host image residency is unobserved; the container agent cannot measure it.
            container_image_bytes: 0,
            volume_bandwidth_bytes_per_s: vec![50_000_000.0, 200_000_000.0],
            model_cache_probability: 0.0,
            image_cache_probability: 0.0,
            probe_duration_s: 1.0,
            measurement_noise: 0.0,
        },
    }
}

/// Provider capacity is an admission check; it does not size model or KV allocations.
fn native_hardware_admitted(offer: &Offering, required: u64) -> bool {
    offer
        .gpu_memory_gb
        .checked_mul(1024 * 1024 * 1024)
        .is_some_and(|bytes| bytes >= required)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn early_capacity_failure_retains_empty_and_price_rejected_quotes() {
        let out = tempfile::tempdir().unwrap();
        assert!(matches!(
            admit_offerings(vec![], 1.0, out.path()),
            Err(Error::Provider(switchyard_provider::Error::NoCapacity))
        ));
        let read = || -> serde_json::Value {
            serde_json::from_slice(
                &std::fs::read(out.path().join("offering-admission.json")).unwrap(),
            )
            .unwrap()
        };
        assert_eq!(read()["quoted"], serde_json::json!([]));
        let offer = Offering {
            gpu: "fixture".into(),
            gpu_memory_gb: 24,
            data_center: "fixture".into(),
            availability: "LOW".into(),
            gpu_usd_per_hour: 1.0,
            disk_usd_per_hour: 0.01,
            quoted_at_unix_s: 123,
        };
        assert!(admit_offerings(vec![offer.clone()], 1.0, out.path()).is_err());
        assert_eq!(read()["quoted"][0]["quoted_at_unix_s"], 123);
        assert_eq!(read()["admitted"], serde_json::json!([]));
        assert_eq!(
            admit_offerings(vec![offer], 1.01, out.path())
                .unwrap()
                .len(),
            1
        );
        assert_eq!(read()["admitted"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn native_admission_rejects_insufficient_and_overflowing_catalog_capacity() {
        let mut offer = Offering {
            gpu: "fixture".into(),
            gpu_memory_gb: 8,
            data_center: "fixture".into(),
            availability: "available".into(),
            gpu_usd_per_hour: 0.2,
            disk_usd_per_hour: 0.01,
            quoted_at_unix_s: 0,
        };
        assert!(native_hardware_admitted(&offer, 8 << 30));
        assert!(!native_hardware_admitted(&offer, (8 << 30) + 1));
        offer.gpu_memory_gb = u64::MAX;
        assert!(!native_hardware_admitted(&offer, 1));
    }
}
