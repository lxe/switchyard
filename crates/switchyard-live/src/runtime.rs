//! Live observations and requests connected to the shared scheduler contract.
//!
//! Cloud lifecycle actions, profile preparation, and workload execution are separate
//! modules so a provider does not need to understand model serving or routing policy.

mod bake;
mod lifecycle;
mod native_cache;
mod profiles;
mod report;
mod routing;
mod telemetry;
mod workload;

use crate::model::ModelManifest;
use crate::{
    Error, Result,
    config::ExecutionConfig,
    planner::{Message, Planner},
    serving::{Completion, Endpoint},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use switchyard_core::{Policy, Profile, Scheduler, Scheduling, scheduler::ObservedNode};
use switchyard_provider::{Offering, Pod, Provider, storage::Storage};
use switchyard_scheduler::BuiltinScheduler;
use tokio::task::JoinSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkCall {
    pub model: String,
    pub messages: Vec<Message>,
    pub planned_output: u64,
    pub max_output: u64,
    pub arrival_delay_s: f64,
    pub class: String,
    #[serde(default)]
    pub depends_on: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct WorkloadReport {
    pub name: String,
    pub policy: String,
    pub requests: usize,
    pub completed: usize,
    pub failed: usize,
    pub unfinished: usize,
    pub target_output_tokens_per_s: f64,
    pub budget_exceeded: bool,
    pub output_tokens: u64,
    pub output_tokens_per_s: f64,
    pub ttft_p50_s: Option<f64>,
    pub ttft_p95_s: Option<f64>,
    pub response_p95_s: Option<f64>,
    pub elapsed_s: f64,
    pub estimated_rental_usd: f64,
    pub invalid_actions: usize,
    pub launches: usize,
    pub probes: usize,
    pub releases: usize,
    pub controller: serde_json::Value,
    pub completions: Vec<Completion>,
    pub failures: BTreeMap<usize, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnedPod {
    pub pod: Pod,
    pub allocated_s: f64,
    pub ended_s: Option<f64>,
    pub profile: usize,
    pub quoted: Offering,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingLaunch {
    pub name: String,
    pub ticket_key: String,
    #[serde(default)]
    pub resolved: bool,
}
#[derive(Debug, Serialize)]
struct AllocationRejection {
    at_s: f64,
    profile: usize,
    name: String,
    status: u16,
    retry_at_s: f64,
}
struct Node {
    owned: OwnedPod,
    endpoint: Endpoint,
    observed: ObservedNode,
    ticket_key: String,
    checked_tokenizer: bool,
    bandwidth: switchyard_core::probe::BandwidthWindow,
    native_cache_binding: Option<crate::native_cache_registry::CacheBinding>,
    successful_calls: u64,
    cache_collection_attempted: bool,
    cache_quiescence_known: bool,
}

/// Live execution translates observations and provider actions through the simulator's scheduling contract.
/// Only pods recorded in this runtime's ownership journal can be terminated.
pub struct LiveRuntime<P: Provider> {
    provider: P,
    storage: Storage,
    config: ExecutionConfig,
    offerings: Vec<Offering>,
    profile_offerings: Vec<Offering>,
    pub profiles: Vec<Profile>,
    nodes: Vec<Node>,
    pending: Vec<PendingLaunch>,
    provisioning_cooldowns: BTreeMap<usize, f64>,
    allocation_rejections: Vec<AllocationRejection>,
    observations: Vec<serde_json::Value>,
    manifests: BTreeMap<String, ModelManifest>,
    runtime_manifest: Option<switchyard_artifacts::Manifest>,
    component_manifests: BTreeMap<String, switchyard_artifacts::Manifest>,
    cache_manifests: BTreeMap<String, switchyard_artifacts::Manifest>,
    cache_export: Option<switchyard_agent::export::CacheExportConfig>,
    collected_native_caches: Vec<(usize, crate::native_cache_registry::CollectedCache)>,
    planners: BTreeMap<String, Arc<Planner>>,
    epoch: Instant,
    run_id: String,
    out: PathBuf,
    controller: Box<dyn Scheduler + Send>,
    scheduling: Scheduling,
    tasks: JoinSet<()>,
    launches: usize,
    probes: usize,
    releases: usize,
    invalid: usize,
}

pub fn nonce() -> Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|_| Error::Config("system random source failed".into()))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

impl<P: Provider> LiveRuntime<P> {
    pub(crate) async fn capture_metrics(&self, name: &str, stage: &str) -> Result<()> {
        std::fs::create_dir_all(self.out.join("workloads"))?;
        let mut observations = vec![];
        for node in &self.nodes {
            if node.owned.ended_s.is_none() && node.observed.is_ready {
                let value = match node.endpoint.metrics().await {
                    Ok(metrics) => {
                        serde_json::json!({"pod_id":node.owned.pod.id,"metrics":metrics})
                    }
                    Err(error) => {
                        serde_json::json!({"pod_id":node.owned.pod.id,"error":error.to_string()})
                    }
                };
                observations.push(value);
            }
        }
        self.save(
            &format!("workloads/{name}-metrics-{stage}.json"),
            &observations,
        )
    }
    pub async fn new(
        provider: P,
        storage: Storage,
        config: ExecutionConfig,
        policy: Policy,
        out: &Path,
    ) -> Result<Self> {
        config.validate()?;
        std::fs::create_dir_all(out)?;
        let profiles::Prepared {
            offerings,
            manifests,
            planners,
            profiles,
            profile_offerings,
            runtime_manifest,
            component_manifests,
            cache_manifests,
        } = profiles::prepare(&provider, &storage, &config, out).await?;
        let scheduling = config.scheduling.clone();
        if out.join("pending-launches.json").exists() {
            return Err(Error::Config("output directory already contains an ownership journal; recover it or choose a new directory".into()));
        }
        let runtime = Self {
            provider,
            storage,
            config,
            offerings,
            profile_offerings,
            profiles,
            nodes: vec![],
            pending: vec![],
            provisioning_cooldowns: BTreeMap::new(),
            allocation_rejections: vec![],
            observations: vec![],
            manifests,
            runtime_manifest,
            component_manifests,
            cache_manifests,
            cache_export: None,
            collected_native_caches: vec![],
            planners,
            epoch: Instant::now(),
            run_id: nonce()?[..16].into(),
            out: out.into(),
            controller: Box::new(BuiltinScheduler::new(policy)),
            scheduling,
            tasks: JoinSet::new(),
            launches: 0,
            probes: 0,
            releases: 0,
            invalid: 0,
        };
        runtime.save("profiles-used.json", &runtime.profiles)?;
        runtime.save("offerings.json", &runtime.offerings)?;
        runtime.save("configuration.json", &runtime.config)?;
        Ok(runtime)
    }
    pub fn with_scheduler(mut self, controller: impl Scheduler + Send + 'static) -> Self {
        self.controller = Box::new(controller);
        self
    }
    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }
    fn save<T: Serialize>(&self, name: &str, value: &T) -> Result<()> {
        let path = self.out.join(name);
        let temp = path.with_extension("partial");
        std::fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
        std::fs::rename(temp, path)?;
        Ok(())
    }
    fn journal(&self) -> Result<()> {
        self.save(
            "owned-pods.json",
            &self.nodes.iter().map(|n| &n.owned).collect::<Vec<_>>(),
        )
    }
    pub fn rental_usd(&self) -> f64 {
        self.nodes
            .iter()
            .map(|n| {
                // The provider's actual pod quote supersedes the catalog GPU price; disk is an explicit separate estimate.
                let rate = rental_rate(&n.owned);
                (n.owned.ended_s.unwrap_or(self.now()) - n.owned.allocated_s).max(1.0) * rate
                    / 3600.0
            })
            .sum()
    }
    fn limits(&self) -> Result<()> {
        if self.now() > self.config.limits.run_timeout_s as f64 {
            return Err(Error::Limit("wall-clock run timeout".into()));
        }
        if self.rental_usd() >= self.config.limits.max_run_usd {
            return Err(Error::Limit("rental cost budget".into()));
        }
        Ok(())
    }

    /// Remove client-side work after cancellation before another workload reuses node IDs.
    async fn abort_requests(&mut self) {
        self.tasks.abort_all();
        while self.tasks.join_next().await.is_some() {}
        let now = self.now();
        for node in &mut self.nodes {
            if !node.observed.active.is_empty() {
                node.cache_quiescence_known = false;
                node.observed.active.clear();
                node.observed.last_used_s = now;
            }
        }
    }
    pub fn set_policy(&mut self, policy: Policy) {
        self.controller = Box::new(BuiltinScheduler::new(policy));
    }
    pub fn pods(&self) -> Vec<OwnedPod> {
        self.nodes.iter().map(|n| n.owned.clone()).collect()
    }
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.config.models.iter().map(|m| m.id.as_str())
    }
    pub fn plan_input(&self, model: &str, messages: &[Message]) -> Result<u64> {
        self.planners
            .get(model)
            .ok_or_else(|| Error::Config("unknown model".into()))?
            .input_tokens(messages)
    }
}

fn rental_rate(owned: &OwnedPod) -> f64 {
    // Conservative until billing establishes whether pod.cost already includes disk.
    owned.pod.cost.max(owned.quoted.gpu_usd_per_hour) + owned.quoted.disk_usd_per_hour
}
