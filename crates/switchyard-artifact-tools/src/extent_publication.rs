//! Publish raw file extents without assigning archive identities to them.
use crate::{Error, Result, publication::Snapshot};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
};
use switchyard_artifacts::extents::ExtentManifest;
use switchyard_provider::{config::valid_key_prefix, storage::Storage};

/// Immutable layout and content references; no credentials or download capabilities.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExtentPublicationReceipt {
    pub format: String,
    pub id: String,
    pub content_id: String,
    pub prefix: String,
    pub manifest_created: bool,
    pub objects_created: BTreeMap<String, bool>,
}

/// Validate local object bytes, publish each unique object, then the captured manifest.
///
/// Supports whole raw objects and ranges within shared objects. Every declared range
/// digest is checked during preparation, outside endpoint startup. The caller must
/// keep object files stable until publication completes. Conditional storage writes
/// reject different existing contents; cancellation may leave objects but never a
/// manifest describing an incomplete upload.
pub async fn publish_extent_directory(
    storage: &Storage,
    directory: &Path,
    namespace: &str,
    concurrency: usize,
) -> Result<ExtentPublicationReceipt> {
    if !valid_key_prefix(namespace) || !(1..=32).contains(&concurrency) {
        return Err(Error::Config(
            "invalid publication prefix or concurrency".into(),
        ));
    }
    let snapshot = Snapshot::capture(&directory.join("manifest.json"), ExtentManifest::validate)?;
    let manifest = snapshot.manifest.clone();
    let source = directory.to_owned();
    let objects =
        tokio::task::spawn_blocking(move || validate_objects(&source, &manifest)).await??;
    let prefix = format!("{namespace}/{}", snapshot.manifest.id);
    let objects_created = {
        let operations = futures_util::stream::iter(objects)
            .map(|(name, (digest, size))| {
                let storage = storage.clone();
                let path = directory.join(&name);
                let key = format!("{prefix}/{name}");
                async move {
                    Ok::<_, Error>((
                        name,
                        storage
                            .put_file_immutable(&key, &path, &digest, size)
                            .await?,
                    ))
                }
            })
            .buffer_unordered(concurrency);
        tokio::pin!(operations);
        let mut objects_created = BTreeMap::new();
        while let Some(result) = operations.next().await {
            let (name, created) = result?;
            objects_created.insert(name, created);
        }
        objects_created
    };
    let manifest_created = storage
        .put_file_immutable(
            &format!("{prefix}/manifest.json"),
            &snapshot.path,
            &snapshot.sha256,
            snapshot.size,
        )
        .await?;
    Ok(ExtentPublicationReceipt {
        format: snapshot.manifest.format.clone(),
        id: snapshot.manifest.id.clone(),
        content_id: snapshot.manifest.content_id.clone(),
        prefix,
        manifest_created,
        objects_created,
    })
}

fn digest(reader: &mut impl Read) -> Result<String> {
    let mut buffer = [0_u8; 64 * 1024];
    let mut hash = Sha256::new();
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn validate_objects(
    directory: &Path,
    manifest: &ExtentManifest,
) -> Result<BTreeMap<String, (String, u64)>> {
    let mut objects = BTreeMap::new();
    for extent in &manifest.extents {
        let path = directory.join(&extent.object);
        if !objects.contains_key(&extent.object) {
            let metadata = fs::symlink_metadata(&path)?;
            if !metadata.is_file() || metadata.len() != extent.object_size {
                return Err(Error::Config("extent object type or size mismatch".into()));
            }
            objects.insert(
                extent.object.clone(),
                (digest(&mut File::open(&path)?)?, extent.object_size),
            );
        }
        let (object_digest, size) = &objects[&extent.object];
        if *size != extent.object_size {
            return Err(Error::Config("inconsistent extent object size".into()));
        }
        let actual = if extent.object_offset == 0 && extent.length == *size {
            object_digest.clone()
        } else {
            let mut file = File::open(&path)?;
            file.seek(SeekFrom::Start(extent.object_offset))?;
            digest(&mut file.take(extent.length))?
        };
        if actual != extent.sha256 {
            return Err(Error::Config("extent object digest mismatch".into()));
        }
    }
    Ok(objects)
}
