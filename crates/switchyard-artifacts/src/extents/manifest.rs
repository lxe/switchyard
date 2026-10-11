use crate::{Entry, Error, Manifest, Metadata, Result, manifest::valid_digest, paths};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const FORMAT: &str = "switchyard-extents-v1";
pub(super) const RECEIPT: &str = ".switchyard-extents-id";

/// One exact write into a regular file. An object may be a raw extent or a full file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extent {
    pub entry: usize,
    pub offset: u64,
    pub length: u64,
    pub object: String,
    pub object_offset: u64,
    pub object_size: u64,
    pub sha256: String,
}

/// Layout identity and stable filesystem-content identity are deliberately distinct.
/// Neither is interchangeable with an artifact-v1 or legacy shards-v1 identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtentManifest {
    pub format: String,
    pub id: String,
    pub content_id: String,
    pub metadata: Metadata,
    pub entries: Vec<Entry>,
    /// Sorted by (entry, offset); each nonempty file has exact contiguous coverage.
    pub extents: Vec<Extent>,
}

/// URLs are integration-owned capabilities, excluded from identity and Debug output.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtentTicket {
    pub manifest: ExtentManifest,
    pub urls: BTreeMap<String, String>,
}

impl ExtentManifest {
    pub fn content_identity(&self) -> Result<String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(
                "switchyard-files-v1",
                &self.metadata,
                &self.entries,
            ))?)
        ))
    }
    pub fn identity(&self) -> Result<String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(
                &self.format,
                &self.content_id,
                &self.extents,
            ))?)
        ))
    }
    pub fn validate(&self) -> Result<()> {
        self.validate_structure()?;
        if self.content_id != self.content_identity()? || self.id != self.identity()? {
            return Err(Error::Invalid("extent manifest identity".into()));
        }
        Ok(())
    }
    /// Pinned startup checks topology and boundaries, without recomputing identities.
    pub fn validate_structure(&self) -> Result<()> {
        if self.format != FORMAT
            || !valid_digest(&self.id)
            || !valid_digest(&self.content_id)
            || self.extents.len() > 1_000_000
        {
            return Err(Error::Invalid("extent format or limits".into()));
        }
        self.filesystem_manifest().validate_entries()?;
        for entry in &self.entries {
            if matches!(entry, Entry::Symlink { .. })
                || entry.path() == RECEIPT
                || entry.path().starts_with(&format!("{RECEIPT}/"))
            {
                return Err(Error::Invalid(
                    "extent entries must be regular files or directories".into(),
                ));
            }
        }
        let mut cursor = 0;
        let mut objects = BTreeMap::new();
        for (entry, file) in self.entries.iter().enumerate() {
            let mut offset = 0_u64;
            while let Some(extent) = self.extents.get(cursor).filter(|e| e.entry == entry) {
                paths::relative(&extent.object)?;
                if !matches!(file, Entry::File { .. })
                    || extent.offset != offset
                    || extent.length == 0
                    || !valid_digest(&extent.sha256)
                    || extent
                        .object_offset
                        .checked_add(extent.length)
                        .is_none_or(|end| end > extent.object_size)
                {
                    return Err(Error::Invalid("extent coverage or object boundary".into()));
                }
                if objects
                    .insert(&extent.object, extent.object_size)
                    .is_some_and(|size| size != extent.object_size)
                {
                    return Err(Error::Invalid("inconsistent object size".into()));
                }
                offset = offset
                    .checked_add(extent.length)
                    .ok_or_else(|| Error::Invalid("extent overflow".into()))?;
                cursor += 1;
            }
            if offset != file.size() {
                return Err(Error::Invalid("incomplete extent file coverage".into()));
            }
        }
        if cursor != self.extents.len() {
            return Err(Error::Invalid("unsorted or invalid extent entry".into()));
        }
        self.total_bytes()?;
        Ok(())
    }
    pub fn total_bytes(&self) -> Result<u64> {
        self.entries.iter().try_fold(0_u64, |sum, e| {
            sum.checked_add(e.size())
                .ok_or_else(|| Error::Invalid("extent total overflow".into()))
        })
    }
    // Reuse artifact-v1's topology rules without changing its serialized format or identity.
    pub(super) fn filesystem_manifest(&self) -> Manifest {
        Manifest {
            format: crate::manifest::FORMAT.into(),
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            entries: self.entries.clone(),
            chunks: vec![],
        }
    }
}
