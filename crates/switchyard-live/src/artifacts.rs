//! Resolve immutable serving artifacts into scoped transfer capabilities.
//!
//! This integration owns deployment references and engine/cache compatibility.
//! Packing, transport, and filesystem verification remain in switchyard-artifacts.
use crate::{
    Error, Result,
    config::{EngineConfig, ModelConfig},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use switchyard_artifacts::{ArtifactTicket, Manifest};
use switchyard_provider::{
    config::{valid_component, valid_env_name, valid_key_prefix},
    storage::Storage,
};

/// A manifest and its chunks reside under the immutable object prefix.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub prefix: String,
    pub id: String,
}

/// A hydrated executable environment, independent of the selected GPU provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingRuntimeConfig {
    pub artifact: ArtifactRef,
    #[serde(default)]
    pub compatibility: BTreeMap<String, String>,
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
}

/// Model and GPU-specific cache contents, prepared outside endpoint startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingCacheConfig {
    pub name: String,
    pub gpu: String,
    pub artifact: ArtifactRef,
    #[serde(default)]
    pub compatibility: BTreeMap<String, String>,
}

impl ArtifactRef {
    pub fn validate(&self) -> Result<()> {
        if !valid_key_prefix(&self.prefix) || !crate::config::valid_sha256(&self.id) {
            return Err(Error::Config("invalid serving artifact reference".into()));
        }
        Ok(())
    }
    pub async fn resolve(
        &self,
        storage: &Storage,
        requirements: &BTreeMap<String, String>,
    ) -> Result<Manifest> {
        self.validate()?;
        let manifest: Manifest = storage
            .get_json(&format!("{}/manifest.json", self.prefix))
            .await?;
        manifest
            .validate()
            .map_err(|e| Error::Config(e.to_string()))?;
        if manifest.id != self.id {
            return Err(Error::Config(
                "serving artifact differs from pinned identity".into(),
            ));
        }
        manifest
            .check_compatibility(requirements)
            .map_err(|e| Error::Config(e.to_string()))?;
        Ok(manifest)
    }
    pub fn ticket(&self, storage: &Storage, manifest: Manifest) -> Result<ArtifactTicket> {
        let urls = manifest
            .chunks
            .iter()
            .map(|c| {
                Ok((
                    c.name.clone(),
                    storage.presign(&format!("{}/{}", self.prefix, c.name))?,
                ))
            })
            .collect::<switchyard_provider::Result<_>>()?;
        Ok(ArtifactTicket { manifest, urls })
    }
}

impl ServingRuntimeConfig {
    pub fn validate_manifest(&self, manifest: &Manifest) -> Result<()> {
        self.validate()?;
        manifest
            .check_compatibility(&self.compatibility)
            .map_err(|e| Error::Config(e.to_string()))?;
        let entry = manifest
            .resolve_entry(&self.program)
            .map_err(|error| Error::Config(error.to_string()))?
            .ok_or_else(|| {
                Error::Config("serving executable is absent from runtime artifact".into())
            })?;
        if !matches!(entry, switchyard_artifacts::Entry::File { .. }) || entry.mode() & 0o100 == 0 {
            return Err(Error::Config(
                "serving runtime program is not executable".into(),
            ));
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        self.artifact.validate()?;
        if !valid_key_prefix(&self.program)
            || self.args.iter().any(|a| a.contains('\0'))
            || self.environment.iter().any(|(k, v)| {
                let upper = k.to_ascii_uppercase();
                !valid_env_name(k)
                    || v.contains('\0')
                    || upper.starts_with("SWITCHYARD_")
                    || upper.contains("SECRET")
                    || upper.contains("PASSWORD")
                    || upper.ends_with("_TOKEN")
                    || upper.ends_with("_KEY")
                    || upper.ends_with("_CREDENTIAL")
            })
            || self.compatibility.iter().any(|(k, v)| {
                k.is_empty() || v.is_empty() || k.chars().chain(v.chars()).any(char::is_control)
            })
        {
            return Err(Error::Config(
                "invalid serving runtime program, environment, or compatibility requirements"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl ServingCacheConfig {
    pub fn validate(&self) -> Result<()> {
        self.artifact.validate()?;
        if !valid_component(&self.name)
            || self.name == "runtime"
            || self.name == ".seeds"
            || self.gpu.is_empty()
            || self.gpu.chars().any(char::is_control)
        {
            return Err(Error::Config("invalid serving cache name or GPU".into()));
        }
        Ok(())
    }
    pub fn requirements(
        &self,
        runtime: &ServingRuntimeConfig,
        model: &ModelConfig,
    ) -> Result<BTreeMap<String, String>> {
        cache_requirements(&self.compatibility, &self.gpu, runtime, model)
    }
}

/// Required labels are derived identically for cache preparation and reuse.
/// Explicit labels may add constraints, but cannot change engine or model identity.
pub fn cache_requirements(
    compatibility: &BTreeMap<String, String>,
    gpu: &str,
    runtime: &ServingRuntimeConfig,
    model: &ModelConfig,
) -> Result<BTreeMap<String, String>> {
    let mut requirements = compatibility.clone();
    for (key, value) in [
        ("runtime_id", runtime.artifact.id.clone()),
        ("model_revision", model.revision.clone()),
        ("model_repository", model.repository.clone()),
        ("model_archive_sha256", model.archive_sha256.clone()),
        ("engine_configuration", engine_fingerprint(&model.engine)?),
        (
            "runtime_launch_configuration",
            runtime_launch_fingerprint(runtime)?,
        ),
        ("gpu", gpu.into()),
    ] {
        if requirements
            .insert(key.into(), value.clone())
            .is_some_and(|v| v != value)
        {
            return Err(Error::Config(format!(
                "serving cache overrides required {key}"
            )));
        }
    }
    Ok(requirements)
}

/// Hash the full serving configuration so caches cannot silently cross engine settings.
pub fn engine_fingerprint(engine: &EngineConfig) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(engine)?)))
}

/// Include launch arguments and environment flags that can change compiler-cache behavior.
pub fn runtime_launch_fingerprint(runtime: &ServingRuntimeConfig) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(
            &runtime.program,
            &runtime.args,
            &runtime.environment
        ))?)
    ))
}
