//! Validate native serving policy before a paid host is provisioned.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

/// Engine-specific configuration key; agent-owned SWITCHYARD variables remain protected.
pub const VLLM_STARTUP_POLICY_ENV: &str = "VLLM_ENGINE_STARTUP_POLICY";

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StartupStrategy {
    #[default]
    Native,
    Prepared,
}
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerMethod {
    #[default]
    Spawn,
    GuardedFork,
}
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReusePolicy {
    #[default]
    Native,
    Required,
}
fn schema_version() -> u32 {
    1
}
fn telemetry() -> bool {
    true
}

/// Policy shared by configuration preflight and the pinned native Python launcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VllmStartupPolicy {
    #[serde(default = "schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub strategy: StartupStrategy,
    #[serde(default)]
    pub worker_method: WorkerMethod,
    #[serde(default)]
    pub fork_runtime_id: Option<String>,
    #[serde(default)]
    pub compute_capability: Option<String>,
    #[serde(default)]
    pub tokenizer_preload: bool,
    #[serde(default)]
    pub cache_policy: ReusePolicy,
    #[serde(default)]
    pub memory_plan: ReusePolicy,
    #[serde(default)]
    pub serving_profile: Option<String>,
    #[serde(default)]
    pub cache_aliases: Vec<String>,
    #[serde(default = "telemetry")]
    pub telemetry: bool,
}
impl VllmStartupPolicy {
    pub fn parse(value: &str) -> Result<Self> {
        let policy: Self = serde_json::from_str(value)
            .map_err(|_| Error::Invalid("vLLM startup policy JSON or fields".into()))?;
        if policy.schema_version != 1
            || policy.serving_profile.as_ref().is_some_and(|value| {
                let path = Path::new(value);
                path.is_absolute()
                    || value.contains(['\\', '\0'])
                    || path.components().count() < 2
                    || path
                        .components()
                        .any(|p| !matches!(p, std::path::Component::Normal(_)))
                    || path
                        .components()
                        .collect::<std::path::PathBuf>()
                        .as_os_str()
                        != value.as_str()
                    || policy.cache_policy != ReusePolicy::Required
            })
            || policy.cache_aliases.len() > 8
            || policy.cache_aliases.iter().any(|value| {
                let path = Path::new(value);
                !path.is_absolute()
                    || value.contains(['\\', '\0'])
                    || path.components().any(|p| {
                        matches!(
                            p,
                            std::path::Component::ParentDir | std::path::Component::CurDir
                        )
                    })
                    || path
                        .components()
                        .collect::<std::path::PathBuf>()
                        .as_os_str()
                        != value.as_str()
            })
            || policy.compute_capability.as_ref().is_some_and(|value| {
                let Some((major, minor)) = value.split_once('.') else {
                    return true;
                };
                !(1..=2).contains(&major.len())
                    || major.starts_with('0')
                    || !major.bytes().all(|b| b.is_ascii_digit())
                    || minor.len() != 1
                    || !minor.bytes().all(|b| b.is_ascii_digit())
            })
        {
            return Err(Error::Invalid(
                "vLLM startup policy schema, paths, or architecture".into(),
            ));
        }
        Ok(policy)
    }
    /// Validate selected artifact and native guard flags without executing Python.
    pub fn validate_environment(
        &self,
        environment: &BTreeMap<String, String>,
        runtime_id: &str,
    ) -> Result<()> {
        if self.worker_method == WorkerMethod::GuardedFork
            && (self.strategy != StartupStrategy::Prepared
                || self.compute_capability.is_none()
                || self.fork_runtime_id.as_deref() != Some(runtime_id))
        {
            return Err(Error::Invalid(
                "guarded fork requires prepared strategy and the selected runtime and GPU".into(),
            ));
        }
        if self.compute_capability.as_ref().is_some_and(|capability| {
            environment
                .get("FLASHINFER_CUDA_ARCH_LIST")
                .is_some_and(|value| value != capability)
        }) {
            return Err(Error::Invalid(
                "FlashInfer architecture differs from selected GPU".into(),
            ));
        }
        if self.cache_policy == ReusePolicy::Required
            && (self.compute_capability.is_none()
                || [
                    "VLLM_FORCE_AOT_LOAD",
                    "FLASHINFER_DISABLE_JIT",
                    "FLASHINFER_NO_DOWNLOAD",
                ]
                .iter()
                .any(|name| environment.get(*name).map(String::as_str) != Some("1")))
        {
            return Err(Error::Invalid(
                "required cache policy needs native strict flags".into(),
            ));
        }
        if self.memory_plan == ReusePolicy::Required
            && environment
                .get("VLLM_ENABLE_STARTUP_PLAN")
                .map(String::as_str)
                != Some("1")
        {
            return Err(Error::Invalid(
                "required memory plan needs native startup-plan flag".into(),
            ));
        }
        Ok(())
    }
}
