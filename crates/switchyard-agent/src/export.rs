//! Optional, ticket-scoped export of a stopped engine's writable compiler caches.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The ticket selects a single child of its runtime's writable cache directory.
/// HTTP clients cannot select paths, commands, or artifact metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheExportConfig {
    pub name: String,
    pub compatibility: BTreeMap<String, String>,
    #[serde(default = "default_drain_timeout")]
    pub drain_timeout_s: u64,
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout_s: u64,
}
fn default_drain_timeout() -> u64 {
    60
}
fn default_shutdown_timeout() -> u64 {
    30
}
impl CacheExportConfig {
    pub(crate) fn validate(&self, runtime_id: &str) -> Result<()> {
        if self.name.is_empty()
            || matches!(self.name.as_str(), "." | ".." | ".seeds" | "runtime")
            || !self
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
            || !(1..=3600).contains(&self.drain_timeout_s)
            || !(1..=300).contains(&self.shutdown_timeout_s)
            || self.compatibility.get("runtime_id").map(String::as_str) != Some(runtime_id)
            || self.compatibility.get("architecture").map(String::as_str)
                != Some(std::env::consts::ARCH)
        {
            return Err(Error::Manifest("invalid cache export capability".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExportPhase {
    Idle,
    Draining,
    Stopping,
    Packing,
    Ready,
    Failed,
}

/// Ready exports include the generic content-addressed manifest, never local paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportState {
    pub phase: ExportPhase,
    pub manifest: Option<switchyard_artifacts::Manifest>,
    pub error: Option<String>,
}
impl Default for ExportState {
    fn default() -> Self {
        Self {
            phase: ExportPhase::Idle,
            manifest: None,
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_defaults_and_reserved_names_are_validated() {
        let runtime = "a".repeat(64);
        let mut config: CacheExportConfig = serde_json::from_value(serde_json::json!({
            "name": "compiled", "compatibility": {"runtime_id": runtime, "architecture": std::env::consts::ARCH}
        })).unwrap();
        assert_eq!(
            (config.drain_timeout_s, config.shutdown_timeout_s),
            (60, 30)
        );
        config.validate(&runtime).unwrap();
        for name in [
            ".seeds",
            "runtime",
            "..",
            "../compiled",
            "/compiled",
            "compiled/child",
        ] {
            config.name = name.into();
            assert!(config.validate(&runtime).is_err(), "{name}");
        }
        config.name = "compiled".into();
        config
            .compatibility
            .insert("architecture".into(), "foreign-architecture".into());
        assert!(config.validate(&runtime).is_err());
    }
}
