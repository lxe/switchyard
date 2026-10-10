//! Load immutable model inputs and construct explicitly assumed hardware profiles.
use crate::{
    Error, Result,
    config::{ExecutionConfig, ModelConfig},
    planner::Planner,
};
use std::{collections::BTreeMap, path::Path, sync::Arc};
use switchyard_agent::Manifest;
use switchyard_core::{CapacityPoint, HostProfile, Profile, Startup, Workload};
use switchyard_provider::{Offering, Provider, storage::Storage};

pub(super) struct Prepared {
    pub(super) offerings: Vec<Offering>,
    pub(super) manifests: BTreeMap<String, Manifest>,
    pub(super) planners: BTreeMap<String, Arc<Planner>>,
    pub(super) profiles: Vec<Profile>,
}

pub(super) async fn prepare<P: Provider>(
    provider: &P,
    storage: &Storage,
    config: &ExecutionConfig,
    out: &Path,
) -> Result<Prepared> {
    let offerings = provider
        .offers()
        .await?
        .into_iter()
        .filter(|o| o.total_usd_per_hour() <= config.limits.max_usd_per_hour)
        .collect::<Vec<_>>();
    if offerings.is_empty() {
        return Err(switchyard_provider::Error::NoCapacity.into());
    }
    let mut manifests = BTreeMap::new();
    let mut planners = BTreeMap::new();
    let mut profiles = vec![];
    for model in &config.models {
        let manifest: Manifest = storage
            .get_json(&format!("{}/manifest.json", model.archive_prefix))
            .await?;
        manifest
            .validate()
            .map_err(|e| Error::Config(e.to_string()))?;
        if manifest.sha256 != model.archive_sha256 {
            return Err(Error::Config(
                "model archive identity differs from the pinned fleet configuration".into(),
            ));
        }
        let planner = Planner::load(model, &out.join("tokenizers")).await?;
        manifests.insert(model.id.clone(), manifest.clone());
        planners.insert(model.id.clone(), Arc::new(planner));
        for offer in &offerings {
            profiles.push(profile(model, &manifest, offer));
        }
    }
    Ok(Prepared {
        offerings,
        manifests,
        planners,
        profiles,
    })
}

/// Use the configured priors without presenting them as measured GPU performance.
fn profile(model: &ModelConfig, manifest: &Manifest, offering: &Offering) -> Profile {
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
            model_bytes: manifest.total_size,
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
