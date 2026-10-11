//! Configuration for a disposable serving host that exports compiler caches.
use crate::{Error, Result, config::ExecutionConfig, workloads::Experiment};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use switchyard_core::Policy;

/// The GPU and engine configuration select cache compatibility; cases describe
/// the request shapes actually exercised. Preparation is separate from routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBakePlan {
    pub model: String,
    pub gpu: String,
    pub name: String,
    #[serde(default)]
    pub compatibility: BTreeMap<String, String>,
    pub warmup: Experiment,
    #[serde(default = "default_export_timeout")]
    pub export_timeout_s: u64,
    #[serde(default = "default_max_bytes")]
    pub max_artifact_bytes: u64,
    #[serde(default = "default_concurrency")]
    pub download_concurrency: usize,
}
fn default_export_timeout() -> u64 {
    180
}
fn default_max_bytes() -> u64 {
    8 << 30
}
fn default_concurrency() -> usize {
    8
}

impl CacheBakePlan {
    pub fn validate(&self, config: &ExecutionConfig) -> Result<()> {
        self.warmup.validate()?;
        if !config.scheduling.classes.contains_key("standard") {
            return Err(Error::Config(
                "cache warmup requires the standard scheduling class".into(),
            ));
        }
        if config.bootstrap.runtime.is_none()
            || !config.models.iter().any(|model| model.id == self.model)
            || self.gpu.is_empty()
            || !switchyard_provider::config::valid_component(&self.name)
            || matches!(self.name.as_str(), "runtime" | ".seeds")
            || !(1..=3600).contains(&self.export_timeout_s)
            || self.max_artifact_bytes == 0
            || !(1..=32).contains(&self.download_concurrency)
            || self
                .compatibility
                .get("architecture")
                .is_none_or(String::is_empty)
            || self.compatibility.iter().any(|(key, value)| {
                key.is_empty()
                    || value.is_empty()
                    || key.chars().chain(value.chars()).any(char::is_control)
            })
            || self
                .compatibility
                .get("gpu_compute_capability")
                .is_some_and(|value| {
                    let mut parts = value.split('.');
                    !parts.next().is_some_and(|part| part.parse::<u16>().is_ok())
                        || !parts.next().is_some_and(|part| part.parse::<u16>().is_ok())
                        || parts.next().is_some()
                })
            || self
                .warmup
                .cases
                .iter()
                .any(|case| case.policy != Policy::WarmOnly)
        {
            return Err(Error::Config(
                "invalid cache preparation model, runtime, GPU labels, or limits".into(),
            ));
        }
        Ok(())
    }
}
