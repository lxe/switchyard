//! Optional compiler-cache families learned from real native inference requests.
//!
//! Registry records are mutable hints; artifacts are immutable. A family binding
//! includes the exact runtime, GPU model, model weights, and launch configuration.
//! Driver-specific directories inside a family must be selected by the native
//! launcher using the observed driver identity. A missing directory uses normal
//! driver JIT. Concurrent publishers are last-writer-wins: an older family may
//! lose a driver entry, causing more JIT, but may never authorize another binding.
use crate::{
    Error, Result,
    artifacts::{ArtifactRef, cache_requirements},
    config::{EngineKind, ExecutionConfig, ModelConfig},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};
use switchyard_artifacts::Manifest;
use switchyard_provider::{
    config::{valid_component, valid_key_prefix},
    storage::Storage,
};

const MAX_RECORD_BYTES: usize = 1 << 20;
const MAX_MANIFEST_BYTES: usize = 8 << 20;

/// Performance-only cache registry. No setting creates hosts or performs warmup.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCacheRegistryConfig {
    pub name: String,
    pub registry_prefix: String,
    pub artifact_prefix: String,
    #[serde(default)]
    pub compatibility: BTreeMap<String, String>,
    #[serde(default = "lookup_timeout")]
    pub lookup_timeout_s: u64,
    #[serde(default = "collection_timeout")]
    pub collection_timeout_s: u64,
    #[serde(default = "publication_timeout")]
    pub publication_timeout_s: u64,
    #[serde(default = "max_bytes")]
    pub max_artifact_bytes: u64,
    #[serde(default = "concurrency")]
    pub transfer_concurrency: usize,
}
fn lookup_timeout() -> u64 {
    3
}
fn collection_timeout() -> u64 {
    30
}
fn publication_timeout() -> u64 {
    60
}
fn max_bytes() -> u64 {
    512 << 20
}
fn concurrency() -> usize {
    4
}

impl NativeCacheRegistryConfig {
    pub(crate) fn validate(&self, config: &ExecutionConfig) -> Result<()> {
        let runtime = config.bootstrap.runtime.as_ref().ok_or_else(|| {
            Error::Config("native cache registry requires a hydrated runtime".into())
        })?;
        if !valid_component(&self.name)
            || matches!(
                self.name.as_str(),
                "." | ".."
                    | "runtime"
                    | ".seeds"
                    | ".switchyard-artifact-id"
                    | ".switchyard-extents-id"
            )
            || !valid_key_prefix(&self.registry_prefix)
            || !valid_key_prefix(&self.artifact_prefix)
            || !(1..=10).contains(&self.lookup_timeout_s)
            || !(1..=30).contains(&self.collection_timeout_s)
            || !(1..=120).contains(&self.publication_timeout_s)
            || self.max_artifact_bytes == 0
            || self.max_artifact_bytes > 1 << 30
            || !(1..=20).contains(&self.transfer_concurrency)
            || self
                .compatibility
                .get("native_driver_cache_layout")
                .map(String::as_str)
                != Some("native-cuda-driver-family-v1")
            || self
                .compatibility
                .keys()
                .chain(runtime.compatibility.keys())
                .any(|key| {
                    matches!(
                        key.as_str(),
                        "cuda_driver" | "driver_identity" | "driver_version"
                    )
                })
            || runtime
                .environment
                .get("NATIVE_ENGINE_CACHE_POLICY")
                .map(String::as_str)
                != Some("opportunistic")
            || runtime
                .compatibility
                .get("architecture")
                .map(String::as_str)
                != Some(std::env::consts::ARCH)
            || runtime.components.iter().any(|part| part.name == self.name)
            || config.models.iter().any(|model| {
                model.engine.kind != EngineKind::Native
                    || model.caches.iter().any(|cache| cache.name == self.name)
            })
            || runtime.components.len()
                + config
                    .models
                    .iter()
                    .map(|model| model.caches.len())
                    .max()
                    .unwrap_or(0)
                >= 64
        {
            return Err(Error::Config(
                "invalid native cache registry bounds, layout, or native deployment".into(),
            ));
        }
        for model in &config.models {
            self.binding(config, model, "validation-gpu")?;
        }
        Ok(())
    }

    pub fn binding(
        &self,
        config: &ExecutionConfig,
        model: &ModelConfig,
        gpu: &str,
    ) -> Result<CacheBinding> {
        let runtime = config
            .bootstrap
            .runtime
            .as_ref()
            .ok_or_else(|| Error::Config("native cache registry requires a runtime".into()))?;
        let mut compatibility = runtime.compatibility.clone();
        for (key, value) in &self.compatibility {
            if value.is_empty()
                || key.is_empty()
                || key.chars().chain(value.chars()).any(char::is_control)
                || compatibility
                    .insert(key.clone(), value.clone())
                    .is_some_and(|old| old != *value)
            {
                return Err(Error::Config(
                    "native cache registry overrides runtime compatibility".into(),
                ));
            }
        }
        // Include the generated native configuration for legacy archives too.
        // Existing strict cache bindings keep their historical derivation.
        let arguments = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(
                &model.engine.args(model, "{model_root}")
            )?)
        );
        if compatibility
            .insert("native_engine_arguments".into(), arguments.clone())
            .is_some_and(|old| old != arguments)
        {
            return Err(Error::Config(
                "native cache registry overrides engine arguments".into(),
            ));
        }
        let compatibility = cache_requirements(&compatibility, gpu, runtime, model)?;
        let identity = format!("{:x}", Sha256::digest(serde_json::to_vec(&compatibility)?));
        Ok(CacheBinding {
            identity,
            compatibility,
        })
    }

    pub(crate) fn key(&self, binding: &CacheBinding) -> String {
        format!("{}/{}.json", self.registry_prefix, binding.identity)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CacheBinding {
    pub identity: String,
    pub compatibility: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheRegistryEntry {
    pub version: u32,
    pub binding: CacheBinding,
    pub artifact: ArtifactRef,
}

impl CacheRegistryEntry {
    fn validate(&self, config: &NativeCacheRegistryConfig, binding: &CacheBinding) -> Result<()> {
        self.artifact.validate()?;
        if self.version != 1
            || &self.binding != binding
            || self.artifact.prefix != format!("{}/{}", config.artifact_prefix, self.artifact.id)
        {
            return Err(Error::Config(
                "native cache registry entry has the wrong binding or namespace".into(),
            ));
        }
        Ok(())
    }
}

/// Invalid records and object failures are treated by the caller as cold JIT.
/// Lookup is bounded in both response bytes and wall-clock time.
pub(crate) async fn lookup(
    storage: &Storage,
    config: &NativeCacheRegistryConfig,
    binding: &CacheBinding,
) -> Result<Option<(ArtifactRef, Manifest)>> {
    match storage
        .get_json_bounded::<CacheRegistryEntry>(&config.key(binding), MAX_RECORD_BYTES)
        .await
    {
        Err(switchyard_provider::Error::Http { status: 404, .. }) => Ok(None),
        Err(error) => Err(error.into()),
        Ok(entry) => {
            entry.validate(config, binding)?;
            let manifest: Manifest = storage
                .get_json_bounded(
                    &format!("{}/manifest.json", entry.artifact.prefix),
                    MAX_MANIFEST_BYTES,
                )
                .await?;
            validate_manifest(config, binding, &entry.artifact.id, &manifest)?;
            Ok(Some((entry.artifact, manifest)))
        }
    }
}

pub(crate) fn validate_manifest(
    config: &NativeCacheRegistryConfig,
    binding: &CacheBinding,
    id: &str,
    manifest: &Manifest,
) -> Result<()> {
    manifest.validate()?;
    manifest.check_compatibility(&binding.compatibility)?;
    let compressed = manifest
        .chunks
        .iter()
        .try_fold(0_u64, |sum, chunk| sum.checked_add(chunk.size));
    let unpacked = manifest
        .chunks
        .iter()
        .try_fold(0_u64, |sum, chunk| sum.checked_add(chunk.unpacked_size));
    if manifest.id != id
        || manifest.metadata.kind != "cache"
        || manifest.metadata.compatibility != binding.compatibility
        || compressed.is_none_or(|bytes| bytes > config.max_artifact_bytes)
        || unpacked.is_none_or(|bytes| bytes > config.max_artifact_bytes)
    {
        return Err(Error::Config(
            "native cache family differs from its binding or exceeds bounds".into(),
        ));
    }
    Ok(())
}

pub(crate) struct CollectedCache {
    pub directory: PathBuf,
    pub manifest: Manifest,
    pub binding: CacheBinding,
}

pub(crate) async fn publish(
    storage: &Storage,
    config: &NativeCacheRegistryConfig,
    collected: &CollectedCache,
) -> Result<switchyard_artifact_tools::PublicationReceipt> {
    validate_manifest(
        config,
        &collected.binding,
        &collected.manifest.id,
        &collected.manifest,
    )?;
    let publication = switchyard_artifact_tools::publish_directory(
        storage,
        &collected.directory,
        &config.artifact_prefix,
        config.transfer_concurrency,
    )
    .await?;
    storage
        .put_json(
            &config.key(&collected.binding),
            &CacheRegistryEntry {
                version: 1,
                binding: collected.binding.clone(),
                artifact: ArtifactRef {
                    id: publication.id.clone(),
                    prefix: publication.prefix.clone(),
                },
            },
        )
        .await?;
    Ok(publication)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> NativeCacheRegistryConfig {
        serde_json::from_value(serde_json::json!({"name":"native-driver","registry_prefix":"registry/v1","artifact_prefix":"cache/v1","compatibility":{"native_driver_cache_layout":"native-cuda-driver-family-v1"}})).unwrap()
    }
    #[test]
    fn wrong_binding_and_namespace_are_never_used() {
        let config = config();
        let binding = CacheBinding {
            identity: "a".repeat(64),
            compatibility: BTreeMap::from([("gpu".into(), "A".into())]),
        };
        let mut entry = CacheRegistryEntry {
            version: 1,
            binding: binding.clone(),
            artifact: ArtifactRef {
                id: "b".repeat(64),
                prefix: format!("cache/v1/{}", "b".repeat(64)),
            },
        };
        entry.validate(&config, &binding).unwrap();
        entry.binding.compatibility.insert("gpu".into(), "B".into());
        assert!(entry.validate(&config, &binding).is_err());
        entry.binding = binding.clone();
        entry.artifact.prefix = "foreign/cache".into();
        assert!(entry.validate(&config, &binding).is_err());
    }
    #[test]
    fn default_collection_is_bounded() {
        let config = config();
        assert_eq!(config.collection_timeout_s, 30);
        assert_eq!(config.max_artifact_bytes, 512 << 20);
        assert_eq!(config.transfer_concurrency, 4);
    }
    #[test]
    fn registry_binding_separates_gpu_weights_and_launch_settings() {
        let mut fleet: crate::config::DeploymentConfig = serde_json::from_str(include_str!(
            "../../../configs/runpod/native-fleet-f32-prefix-32g-candidate.json"
        ))
        .unwrap();
        let registry = config();
        let mut config = fleet.execution();
        let original = registry
            .binding(&config, &config.models[0], "NVIDIA L40")
            .unwrap();
        assert_ne!(
            original.identity,
            registry
                .binding(&config, &config.models[0], "NVIDIA RTX 4090")
                .unwrap()
                .identity
        );
        assert_ne!(
            original.identity,
            registry
                .binding(&config, &config.models[1], "NVIDIA L40")
                .unwrap()
                .identity
        );
        config.models[0].engine.max_model_len += 1;
        assert_ne!(
            original.identity,
            registry
                .binding(&config, &config.models[0], "NVIDIA L40")
                .unwrap()
                .identity
        );
        config = fleet.execution();
        config
            .bootstrap
            .runtime
            .as_mut()
            .unwrap()
            .environment
            .insert("CACHE_TEST_SETTING".into(), "changed".into());
        assert_ne!(
            original.identity,
            registry
                .binding(&config, &config.models[0], "NVIDIA L40")
                .unwrap()
                .identity
        );
        config = fleet.execution();
        config.bootstrap.runtime.as_mut().unwrap().artifact.id = "d".repeat(64);
        assert_ne!(
            original.identity,
            registry
                .binding(&config, &config.models[0], "NVIDIA L40")
                .unwrap()
                .identity
        );
        // Optional registry requires the explicit native fallback launch contract.
        for model in &mut fleet.models {
            model.caches.clear();
        }
        fleet.bootstrap.native_cache_registry = Some(registry);
        assert!(fleet.validate().is_err());
        fleet
            .bootstrap
            .runtime
            .as_mut()
            .unwrap()
            .environment
            .insert("NATIVE_ENGINE_CACHE_POLICY".into(), "opportunistic".into());
        fleet.validate().unwrap();
        for name in ["..", ".switchyard-artifact-id"] {
            let mut invalid = fleet.clone();
            invalid
                .bootstrap
                .native_cache_registry
                .as_mut()
                .unwrap()
                .name = name.into();
            assert!(invalid.validate().is_err());
        }
        for key in ["cuda_driver", "driver_identity", "driver_version"] {
            let mut invalid = fleet.clone();
            invalid
                .bootstrap
                .native_cache_registry
                .as_mut()
                .unwrap()
                .compatibility
                .insert(key.into(), "580.1".into());
            assert!(invalid.validate().is_err());
            let mut invalid = fleet.clone();
            invalid
                .bootstrap
                .runtime
                .as_mut()
                .unwrap()
                .compatibility
                .insert(key.into(), "580.1".into());
            assert!(invalid.validate().is_err());
        }
    }
}
