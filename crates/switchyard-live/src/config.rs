use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use switchyard_core::{Scheduling, validate_scheduling};
use switchyard_provider::{
    ComputeConfig, StorageConfig,
    config::{valid_component, valid_key_prefix, valid_text, validate_endpoint},
};

/// Portable deployment settings contain credential variable names, never their values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "DeploymentConfigInput")]
pub struct DeploymentConfig {
    pub compute: ComputeConfig,
    pub storage: StorageConfig,
    pub bootstrap: BootstrapConfig,
    pub models: Vec<ModelConfig>,
    pub limits: Limits,
    pub scheduling: Scheduling,
}

/// Execution settings shared by live backends, independent of any cloud's catalog options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "ExecutionConfigInput")]
pub struct ExecutionConfig {
    pub storage: StorageConfig,
    pub bootstrap: BootstrapConfig,
    pub models: Vec<ModelConfig>,
    pub limits: Limits,
    pub disk_gb: u64,
    pub scheduling: Scheduling,
}

// The omitted-block default depends on the enclosing hard startup limit. Resolve it
// during decoding rather than encoding a sentinel in the shared Scheduling contract.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeploymentConfigInput {
    compute: ComputeConfig,
    storage: StorageConfig,
    bootstrap: BootstrapConfig,
    models: Vec<ModelConfig>,
    limits: Limits,
    #[serde(default)]
    scheduling: Option<Scheduling>,
}

impl From<DeploymentConfigInput> for DeploymentConfig {
    fn from(input: DeploymentConfigInput) -> Self {
        let scheduling = input
            .scheduling
            .unwrap_or_else(|| legacy_scheduling(&input.limits));
        Self {
            compute: input.compute,
            storage: input.storage,
            bootstrap: input.bootstrap,
            models: input.models,
            limits: input.limits,
            scheduling,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionConfigInput {
    storage: StorageConfig,
    bootstrap: BootstrapConfig,
    models: Vec<ModelConfig>,
    limits: Limits,
    disk_gb: u64,
    #[serde(default)]
    scheduling: Option<Scheduling>,
}

impl From<ExecutionConfigInput> for ExecutionConfig {
    fn from(input: ExecutionConfigInput) -> Self {
        let scheduling = input
            .scheduling
            .unwrap_or_else(|| legacy_scheduling(&input.limits));
        Self {
            storage: input.storage,
            bootstrap: input.bootstrap,
            models: input.models,
            limits: input.limits,
            disk_gb: input.disk_gb,
            scheduling,
        }
    }
}

fn legacy_scheduling(limits: &Limits) -> Scheduling {
    Scheduling {
        maximum_startup_s: limits.startup_timeout_s as f64,
        minimum_hydration_bytes_per_s: 5_000_000.0,
        ..Scheduling::default()
    }
}

impl DeploymentConfig {
    pub fn execution(&self) -> ExecutionConfig {
        ExecutionConfig {
            storage: self.storage.clone(),
            bootstrap: self.bootstrap.clone(),
            models: self.models.clone(),
            limits: self.limits.clone(),
            disk_gb: self.compute.disk_gb,
            scheduling: self.scheduling.clone(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.compute.validate()?;
        self.execution().validate()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapConfig {
    pub image: String,
    pub agent_url: String,
    pub agent_sha256: String,
    pub ticket_prefix: String,
    /// Artifact transfer and model placement policy shared with the endpoint agent.
    #[serde(default)]
    pub startup: switchyard_agent::startup::StartupPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<crate::artifacts::ServingRuntimeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "ModelConfigInput")]
pub struct ModelConfig {
    pub id: String,
    pub repository: String,
    pub revision: String,
    #[serde(flatten)]
    pub artifact: crate::model::ModelArtifact,
    pub engine: EngineConfig,
    /// Planning priors remain assumptions until a matching live measurement replaces them.
    pub prior: CapacityPrior,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub caches: Vec<crate::artifacts::ServingCacheConfig>,
}

#[derive(Deserialize)]
struct ModelConfigInput {
    id: String,
    repository: String,
    revision: String,
    engine: EngineConfig,
    prior: CapacityPrior,
    #[serde(default)]
    caches: Vec<crate::artifacts::ServingCacheConfig>,
    #[serde(flatten)]
    artifact: std::collections::BTreeMap<String, serde_json::Value>,
}
impl TryFrom<ModelConfigInput> for ModelConfig {
    type Error = serde_json::Error;
    fn try_from(input: ModelConfigInput) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            id: input.id,
            repository: input.repository,
            revision: input.revision,
            engine: input.engine,
            prior: input.prior,
            caches: input.caches,
            artifact: serde_json::from_value(serde_json::to_value(input.artifact)?)?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    pub max_model_len: u64,
    pub max_num_seqs: usize,
    pub gpu_memory_utilization: f64,
    pub dtype: String,
    pub language_model_only: bool,
    pub extra_args: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityPrior {
    pub single_output_tps: f64,
    pub aggregate_output_tps: f64,
    pub input_work_per_token: f64,
    pub startup_s: f64,
    pub kv_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_instances: usize,
    pub max_usd_per_hour: f64,
    pub max_run_usd: f64,
    /// Total estimated rental budget across every model in a multi-model experiment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_experiment_usd: Option<f64>,
    /// Wall-clock limit shared by every model in a multi-model experiment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment_timeout_s: Option<u64>,
    pub run_timeout_s: u64,
    pub startup_timeout_s: u64,
    pub request_timeout_s: u64,
    pub hydration_concurrency: usize,
    pub hydration_stall_s: u64,
    pub poll_ms: u64,
}

impl ExecutionConfig {
    pub fn validate(&self) -> Result<()> {
        validate_scheduling(&self.scheduling).map_err(Error::Config)?;
        self.bootstrap
            .startup
            .validate()
            .map_err(|error| Error::Config(error.to_string()))?;
        if self.bootstrap.startup.early_launch && self.bootstrap.runtime.is_none() {
            return Err(Error::Config(
                "early engine launch requires a configured runtime prelude".into(),
            ));
        }
        if let Some(runtime) = &self.bootstrap.runtime {
            runtime.validate()?;
        }
        self.storage.validate()?;
        validate_endpoint(&self.bootstrap.agent_url)?;
        let l = &self.limits;
        if self.models.is_empty()
            || self.disk_gb == 0
            || !valid_text(&self.bootstrap.image)
            || !valid_sha256(&self.bootstrap.agent_sha256)
            || !valid_key_prefix(&self.bootstrap.ticket_prefix)
            || l.max_instances == 0
            || l.max_instances > 64
            || l.run_timeout_s == 0
            || l.startup_timeout_s == 0
            || l.request_timeout_s == 0
            || l.hydration_stall_s == 0
            || l.poll_ms < 100
            || l.hydration_concurrency == 0
            || l.hydration_concurrency > 64
            || !l.max_run_usd.is_finite()
            || l.max_run_usd <= 0.0
            || l.max_experiment_usd
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
            || l.experiment_timeout_s == Some(0)
            || !l.max_usd_per_hour.is_finite()
            || l.max_usd_per_hour <= 0.0
            || !(60..=86400).contains(&self.storage.url_lifetime_s)
            || self.storage.url_lifetime_s < l.startup_timeout_s
        {
            return Err(Error::Config(
                "invalid compute, storage, or experiment limits".into(),
            ));
        }
        let mut ids = std::collections::BTreeSet::new();
        for m in &self.models {
            m.artifact.validate()?;
            let mut cache_keys = std::collections::BTreeSet::new();
            for cache in &m.caches {
                cache.validate()?;
                let runtime = self.bootstrap.runtime.as_ref().ok_or_else(|| {
                    Error::Config("serving caches require a hydrated runtime".into())
                })?;
                cache.requirements(runtime, m)?;
                if !cache_keys.insert((&cache.gpu, &cache.name)) {
                    return Err(Error::Config(
                        "duplicate serving cache for GPU and name".into(),
                    ));
                }
            }
            if !ids.insert(&m.id)
                || !valid_component(&m.id)
                || m.repository.split('/').count() != 2
                || !valid_key_prefix(&m.repository)
                || m.revision.len() != 40
                || !m.revision.bytes().all(|b| b.is_ascii_hexdigit())
                || m.engine.max_num_seqs == 0
                || m.engine.max_model_len == 0
                || m.engine.gpu_memory_utilization <= 0.0
                || m.engine.gpu_memory_utilization >= 1.0
                || !m.engine.gpu_memory_utilization.is_finite()
                || !valid_text(&m.engine.dtype)
                || !valid_extra_args(&m.engine.extra_args)
                || [
                    m.prior.single_output_tps,
                    m.prior.aggregate_output_tps,
                    m.prior.startup_s,
                ]
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
                || m.prior.aggregate_output_tps < m.prior.single_output_tps
                || !m.prior.input_work_per_token.is_finite()
                || m.prior.input_work_per_token < 0.0
                || m.prior.kv_tokens < m.engine.max_model_len
            {
                return Err(Error::Config(
                    "invalid model, engine settings, or capacity prior".into(),
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_extra_args(args: &[String]) -> bool {
    const MANAGED: &[&str] = &[
        "--host",
        "--port",
        "--uds",
        "--api-key",
        "--served-model-name",
        "--model",
        "--config",
        "--max-model-len",
        "--max-num-seqs",
        "--gpu-memory-utilization",
        "--dtype",
        "--language-model-only",
    ];
    args.iter().all(|arg| {
        !arg.contains('\0') && !MANAGED.contains(&arg.split('=').next().unwrap_or_default())
    })
}
