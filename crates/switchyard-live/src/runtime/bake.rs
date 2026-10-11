//! One owned host prepares model/GPU caches; publication happens after cleanup.
use super::LiveRuntime;
use crate::{
    Error, Result,
    artifacts::{ArtifactRef, ServingCacheConfig, cache_requirements},
    bake::CacheBakePlan,
};
use serde::Serialize;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use switchyard_agent::export::{CacheExportConfig, ExportPhase};
use switchyard_core::Policy;
use switchyard_provider::Provider;

#[derive(Debug, Serialize)]
pub struct PreparedCache {
    pub directory: PathBuf,
    pub manifest: switchyard_artifacts::Manifest,
    pub configuration: ServingCacheConfig,
}

impl<P: Provider> LiveRuntime<P> {
    /// Warm a single model on a selected GPU, stop the serving process, and fetch
    /// its verified cache artifact. Callers always clean up before publishing it.
    pub async fn bake_cache(
        &mut self,
        plan: &CacheBakePlan,
        namespace: &str,
    ) -> Result<PreparedCache> {
        plan.validate(&self.config)?;
        if !switchyard_provider::config::valid_key_prefix(namespace)
            || !self.nodes.is_empty()
            || self.config.models.len() != 1
            || self.config.limits.max_instances != 1
        {
            return Err(Error::Config(
                "cache preparation requires a fresh single-model, single-host runtime".into(),
            ));
        }
        let model = &self.config.models[0];
        let runtime = self
            .config
            .bootstrap
            .runtime
            .as_ref()
            .expect("validated runtime");
        let requirements = cache_requirements(&plan.compatibility, &plan.gpu, runtime, model)?;
        if self
            .runtime_manifest
            .as_ref()
            .and_then(|manifest| manifest.metadata.compatibility.get("architecture"))
            != requirements.get("architecture")
        {
            return Err(Error::Config(
                "cache preparation architecture differs from runtime artifact".into(),
            ));
        }
        for case in &plan.warmup.cases {
            for id in 0..case.calls {
                let call = case.request(&plan.model, id);
                let input = self.plan_input(&plan.model, &call.messages)?;
                if input
                    .checked_add(call.max_output)
                    .is_none_or(|tokens| tokens > model.engine.max_model_len)
                    || serde_json::to_vec(&crate::serving::completion_payload(
                        &call.model,
                        &call.messages,
                        call.max_output,
                    ))?
                    .len()
                        > 2 << 20
                {
                    return Err(Error::Config(
                        "cache warmup exceeds model context or HTTP request limit".into(),
                    ));
                }
            }
        }
        self.cache_export = Some(CacheExportConfig {
            name: plan.name.clone(),
            compatibility: requirements.clone(),
            drain_timeout_s: 60,
            shutdown_timeout_s: 30,
        });
        self.save("cache-bake-plan.json", plan)?;
        let candidates = self
            .profile_offerings
            .iter()
            .enumerate()
            .filter(|(_, offer)| offer.gpu == plan.gpu)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(switchyard_provider::Error::NoCapacity.into());
        }
        let mut launched = false;
        for profile in candidates {
            match self.provision(profile).await {
                Ok(true) => {
                    launched = true;
                    break;
                }
                Err(Error::Provider(switchyard_provider::Error::NoCapacity)) | Ok(false) => {}
                Err(error) => return Err(error),
            }
        }
        if !launched {
            return Err(switchyard_provider::Error::NoCapacity.into());
        }
        while !self.nodes[0].observed.is_ready {
            self.limits()?;
            if self.now() - self.nodes[0].owned.allocated_s
                > self.config.limits.startup_timeout_s as f64
            {
                return Err(Error::Limit("cache preparation startup timeout".into()));
            }
            self.observe().await?;
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        // A dedicated preparation host stays pinned between warmup arrivals.
        self.nodes[0].observed.warm = true;
        self.set_policy(Policy::WarmOnly);
        for case in &plan.warmup.cases {
            self.capture_metrics(&case.name, "before").await?;
            let report = self
                .run_workload(
                    &case.name,
                    &case.requests(&plan.model),
                    case.desired_output_tps,
                    case.completion_budget_s,
                )
                .await?;
            self.capture_metrics(&case.name, "after").await?;
            if report.failed != 0
                || report.unfinished != 0
                || report.budget_exceeded
                || report.invalid_actions != 0
            {
                return Err(Error::Request(
                    "cache preparation workload failed; receipt saved".into(),
                ));
            }
        }
        let began = Instant::now();
        let endpoint = self.nodes[0].endpoint.clone();
        let mut status = endpoint.start_cache_export().await?;
        let mut observations = Vec::new();
        let manifest = loop {
            self.limits()?;
            observations.push(serde_json::json!({"at_s":self.now(),"status":status}));
            self.save("cache-export-observations.json", &observations)?;
            self.save("cache-export-status.json", &status)?;
            match status.phase {
                ExportPhase::Ready => {
                    break status.manifest.ok_or_else(|| {
                        Error::Request("ready cache export has no manifest".into())
                    })?;
                }
                ExportPhase::Failed => {
                    return Err(Error::Request(
                        "cache export failed; status receipt saved".into(),
                    ));
                }
                _ if began.elapsed().as_secs() >= plan.export_timeout_s => {
                    return Err(Error::Limit("cache export preparation timeout".into()));
                }
                _ => {}
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
            status = endpoint.cache_export_status().await?;
        };
        let directory = self.out.join("cache-artifact");
        let remaining = Duration::from_secs(plan.export_timeout_s).saturating_sub(began.elapsed());
        {
            let download = tokio::time::timeout(
                remaining,
                endpoint.download_cache_export(
                    &manifest,
                    &requirements,
                    &directory,
                    plan.max_artifact_bytes,
                    plan.download_concurrency,
                ),
            );
            tokio::pin!(download);
            let mut checks = tokio::time::interval(Duration::from_millis(500));
            loop {
                tokio::select! {
                    result = &mut download => {
                        result.map_err(|_| Error::Limit("cache export download timeout".into()))??;
                        break;
                    }
                    _ = checks.tick() => self.limits()?,
                }
            }
        }
        self.limits()?;
        Ok(PreparedCache {
            configuration: ServingCacheConfig {
                name: plan.name.clone(),
                gpu: plan.gpu.clone(),
                artifact: ArtifactRef {
                    prefix: format!("{namespace}/{}", manifest.id),
                    id: manifest.id.clone(),
                },
                compatibility: requirements,
            },
            directory,
            manifest,
        })
    }
}
