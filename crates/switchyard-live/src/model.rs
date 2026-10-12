//! Distinct immutable model references for legacy archives and raw byte extents.
use crate::{Error, Result, config::ModelConfig};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use switchyard_agent::model::{ExtentModel, ExtentSource, LegacySource, ModelSource};
use switchyard_artifacts::extents::{ExtentManifest, ExtentTicket};
use switchyard_provider::{config::valid_key_prefix, storage::Storage};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelArtifact {
    Extents(ExtentReference),
    Legacy(LegacyReference),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyReference {
    pub archive_prefix: String,
    pub archive_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtentReference {
    pub model_artifact: ExtentArtifact,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExtentArtifact {
    Files {
        prefix: String,
        id: String,
        content_id: String,
    },
    Extents {
        prefix: String,
        id: String,
        content_id: String,
        loader: switchyard_agent::weights::VllmLoaderConfig,
    },
}
impl ModelArtifact {
    pub fn validate(&self) -> Result<()> {
        let digest = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
        match self {
            Self::Extents(ExtentReference {
                model_artifact:
                    ExtentArtifact::Files {
                        prefix,
                        id,
                        content_id,
                    },
            }) if valid_key_prefix(prefix) && digest(id) && digest(content_id) => Ok(()),
            Self::Legacy(r) if valid_key_prefix(&r.archive_prefix) && digest(&r.archive_sha256) => {
                Ok(())
            }
            Self::Extents(ExtentReference {
                model_artifact:
                    ExtentArtifact::Extents {
                        prefix,
                        id,
                        content_id,
                        loader,
                    },
            }) if valid_key_prefix(prefix) && digest(id) && digest(content_id) => {
                loader.validate().map_err(Into::into)
            }
            _ => Err(Error::Config("invalid immutable model reference".into())),
        }
    }
    pub fn cache_labels(&self) -> Result<BTreeMap<String, String>> {
        Ok(match self {
            Self::Extents(ExtentReference {
                model_artifact: ExtentArtifact::Files { content_id, .. },
            }) => BTreeMap::from([
                ("model_content_id".into(), content_id.clone()),
                ("weight_loader".into(), "published-files-v1".into()),
            ]),
            Self::Legacy(r) => {
                BTreeMap::from([("model_archive_sha256".into(), r.archive_sha256.clone())])
            }
            Self::Extents(ExtentReference {
                model_artifact:
                    ExtentArtifact::Extents {
                        content_id, loader, ..
                    },
            }) => BTreeMap::from([
                ("model_content_id".into(), content_id.clone()),
                ("weight_loader".into(), loader.identity()?),
            ]),
        })
    }
    pub(crate) async fn resolve(&self, storage: &Storage) -> Result<ModelManifest> {
        self.validate()?;
        match self {
            Self::Legacy(r) => {
                let manifest: switchyard_agent::Manifest = storage
                    .get_json(&format!("{}/manifest.json", r.archive_prefix))
                    .await?;
                manifest
                    .validate()
                    .map_err(|e| Error::Config(e.to_string()))?;
                if manifest.sha256 != r.archive_sha256 {
                    return Err(Error::Config(
                        "model archive differs from pinned identity".into(),
                    ));
                }
                Ok(ModelManifest::Legacy(manifest))
            }
            Self::Extents(ExtentReference {
                model_artifact:
                    ExtentArtifact::Files {
                        prefix,
                        id,
                        content_id,
                    }
                    | ExtentArtifact::Extents {
                        prefix,
                        id,
                        content_id,
                        ..
                    },
            }) => {
                let manifest: ExtentManifest =
                    storage.get_json(&format!("{prefix}/manifest.json")).await?;
                manifest.validate()?;
                if manifest.id != *id || manifest.content_id != *content_id {
                    return Err(Error::Config(
                        "model extents differ from pinned layout or content".into(),
                    ));
                }
                Ok(ModelManifest::Extents(manifest))
            }
        }
    }
    pub(crate) fn ticket(&self, storage: &Storage, manifest: ModelManifest) -> Result<ModelSource> {
        match (self, manifest) {
            (Self::Legacy(r), ModelManifest::Legacy(manifest)) => {
                let urls = manifest
                    .chunks
                    .iter()
                    .map(|c| {
                        Ok((
                            c.name.clone(),
                            storage.presign(&format!("{}/{}", r.archive_prefix, c.name))?,
                        ))
                    })
                    .collect::<switchyard_provider::Result<_>>()?;
                Ok(ModelSource::Legacy(LegacySource { manifest, urls }))
            }
            (
                Self::Extents(ExtentReference {
                    model_artifact: ExtentArtifact::Extents { prefix, loader, .. },
                }),
                ModelManifest::Extents(manifest),
            ) => {
                let urls = manifest
                    .extents
                    .iter()
                    .map(|e| e.object.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .map(|name| Ok((name.clone(), storage.presign(&format!("{prefix}/{name}"))?)))
                    .collect::<switchyard_provider::Result<_>>()?;
                Ok(ModelSource::Extents(ExtentSource {
                    model_source: ExtentModel::Extents {
                        artifact: ExtentTicket { manifest, urls },
                        loader: loader.clone(),
                    },
                }))
            }
            (
                Self::Extents(ExtentReference {
                    model_artifact: ExtentArtifact::Files { prefix, .. },
                }),
                ModelManifest::Extents(manifest),
            ) => {
                let urls = manifest
                    .extents
                    .iter()
                    .map(|e| e.object.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .map(|name| Ok((name.clone(), storage.presign(&format!("{prefix}/{name}"))?)))
                    .collect::<switchyard_provider::Result<_>>()?;
                Ok(ModelSource::Extents(ExtentSource {
                    model_source: ExtentModel::Files {
                        artifact: ExtentTicket { manifest, urls },
                    },
                }))
            }
            _ => Err(Error::Config(
                "model reference and manifest types differ".into(),
            )),
        }
    }
    pub(crate) fn vllm_extents(&self) -> bool {
        matches!(
            self,
            Self::Extents(ExtentReference {
                model_artifact: ExtentArtifact::Extents { .. }
            })
        )
    }
}
#[derive(Clone)]
pub(crate) enum ModelManifest {
    Legacy(switchyard_agent::Manifest),
    Extents(ExtentManifest),
}
impl ModelManifest {
    pub(crate) fn total_size(&self) -> u64 {
        match self {
            Self::Legacy(m) => m.total_size,
            Self::Extents(m) => m.extents.iter().map(|e| e.length).sum(),
        }
    }
}

/// Engine arguments are hashed with a fixed model path so cache identity is stable.
pub fn engine_arguments_identity(model: &ModelConfig) -> Result<String> {
    if model.engine.kind != crate::config::EngineKind::Vllm {
        use sha2::{Digest, Sha256};
        return Ok(format!("{:x}",Sha256::digest(serde_json::to_vec(&(
            "switchyard-engine-arguments-v1", model.engine.kind,
            model.engine.args(model, "{model_root}")
        ))?)));
    }
    switchyard_agent::model::engine_arguments_identity(&model.engine.args(model, "{model_root}"))
        .map_err(Into::into)
}
