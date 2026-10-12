//! Typed model inputs; legacy wire tickets remain readable without fictional TAR metadata.
use crate::{Error, Manifest, Result, Ticket};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, ops::Deref};
use switchyard_artifacts::extents::ExtentTicket;

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelSource {
    Extents(ExtentSource),
    Legacy(LegacySource),
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacySource {
    pub manifest: Manifest,
    pub urls: BTreeMap<String, String>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtentSource {
    pub model_source: ExtentModel,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExtentModel {
    /// Raw file ranges are published before the ordinary engine starts.
    Files { artifact: ExtentTicket },
    Extents {
        artifact: ExtentTicket,
        loader: switchyard_weights::VllmLoaderConfig,
    },
}
impl ModelSource {
    pub fn artifact_identity(&self) -> &str {
        match self {
            Self::Legacy(source) => &source.manifest.sha256,
            Self::Extents(source) => match &source.model_source {
                ExtentModel::Files { artifact } | ExtentModel::Extents { artifact, .. } => {
                    &artifact.manifest.content_id
                }
            },
        }
    }
    pub fn transfer_bytes(&self) -> u64 {
        match self {
            Self::Legacy(source) => source.manifest.total_size,
            Self::Extents(ExtentSource {
                model_source:
                    ExtentModel::Extents { artifact, .. } | ExtentModel::Files { artifact },
            }) => artifact
                .manifest
                .extents
                .iter()
                .fold(0_u64, |n, e| n.saturating_add(e.length)),
        }
    }
    pub fn extent(&self) -> Option<(&ExtentTicket, &switchyard_weights::VllmLoaderConfig)> {
        match self {
            Self::Extents(ExtentSource {
                model_source: ExtentModel::Extents { artifact, loader },
            }) => Some((artifact, loader)),
            _ => None,
        }
    }
    pub fn files(&self) -> Option<&ExtentTicket> {
        match self {
            Self::Extents(ExtentSource {
                model_source: ExtentModel::Files { artifact },
            }) => Some(artifact),
            _ => None,
        }
    }
}
/// A borrowed legacy view keeps the legacy transport independent of extent handling.
pub(crate) struct LegacyTicket<'a> {
    ticket: &'a Ticket,
    pub(crate) manifest: &'a Manifest,
    pub(crate) urls: &'a BTreeMap<String, String>,
}
impl Deref for LegacyTicket<'_> {
    type Target = Ticket;
    fn deref(&self) -> &Ticket {
        self.ticket
    }
}
impl Ticket {
    pub(crate) fn legacy(&self) -> Result<LegacyTicket<'_>> {
        match &self.source {
            ModelSource::Legacy(source) => Ok(LegacyTicket {
                ticket: self,
                manifest: &source.manifest,
                urls: &source.urls,
            }),
            _ => Err(Error::Manifest(
                "legacy transport requires a legacy model source".into(),
            )),
        }
    }
    #[cfg(test)]
    pub(crate) fn legacy_source_mut(&mut self) -> &mut LegacySource {
        match &mut self.source {
            ModelSource::Legacy(source) => source,
            _ => panic!("test expected legacy source"),
        }
    }
}

const MAX_METADATA: u64 = 64 * 1024 * 1024;
pub(crate) fn metadata_bytes(h: &switchyard_artifacts::extents::ExtentManifest) -> Result<u64> {
    let mut config = false;
    let bytes = h.entries.iter().try_fold(0_u64, |sum, e| match e {
        switchyard_artifacts::Entry::File { path, size, .. } if !path.ends_with(".safetensors") => {
            config |= path == "config.json";
            sum.checked_add(*size)
                .ok_or_else(|| Error::Manifest("model metadata overflow".into()))
        }
        _ => Ok(sum),
    })?;
    if !config || bytes > MAX_METADATA {
        return Err(Error::Manifest(
            "extent model requires bounded local metadata and config.json".into(),
        ));
    }
    Ok(bytes)
}

/// Hash serving arguments after replacing the transient checkpoint path.
pub fn engine_arguments_identity(args: &[String]) -> switchyard_artifacts::Result<String> {
    use sha2::{Digest, Sha256};
    let mut args = args.to_vec();
    if args.first().map(String::as_str) != Some("serve") || args.len() < 2 {
        return Err(switchyard_artifacts::Error::Invalid(
            "invalid serving arguments".into(),
        ));
    }
    args[1] = "{model_root}".into();
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(
            "switchyard-serving-arguments-v1",
            args
        ))?)
    ))
}
pub(crate) fn validate_extent_caches(ticket: &Ticket) -> Result<()> {
    let Some((artifact, loader)) = ticket.source.extent() else {
        return Ok(());
    };
    let runtime = ticket
        .runtime
        .as_ref()
        .ok_or_else(|| Error::Manifest("missing extent runtime".into()))?;
    let labels = BTreeMap::from([
        ("model_content_id", artifact.manifest.content_id.clone()),
        ("weight_loader", loader.identity()?),
        (
            "engine_arguments",
            engine_arguments_identity(&ticket.engine_args)?,
        ),
    ]);
    for metadata in runtime
        .caches
        .iter()
        .map(|c| &c.artifact.manifest.metadata.compatibility)
        .chain(runtime.cache_export.iter().map(|c| &c.compatibility))
    {
        if metadata.contains_key("model_archive_sha256")
            || labels.iter().any(|(k, v)| metadata.get(*k) != Some(v))
        {
            return Err(Error::Manifest(
                "extent serving cache has incompatible content, loader or engine arguments".into(),
            ));
        }
    }
    Ok(())
}
