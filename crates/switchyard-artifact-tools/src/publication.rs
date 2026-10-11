//! Publish immutable artifact directories independently of command-line invocation.
use crate::{Error, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};
use switchyard_artifacts::Manifest;
use switchyard_provider::{config::valid_key_prefix, storage::Storage};

/// The immutable reference and whether this publication created each object.
///
/// `prefix` includes the artifact identity and can be used to resolve its
/// `manifest.json` and chunks. False creation flags mean existing object contents
/// were verified to match. Receipts contain no credentials or signed URLs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PublicationReceipt {
    pub id: String,
    pub prefix: String,
    pub manifest_created: bool,
    pub chunks_created: BTreeMap<String, bool>,
}

/// Publish a packed directory through conditional, verified object writes.
///
/// The directory must contain `manifest.json` and its content-addressed chunks.
/// `namespace` is a relative object-key prefix; the validated artifact identity is
/// appended automatically. Concurrency must be between 1 and 32. The original
/// validated manifest bytes are captured before transfer and published last,
/// after every chunk succeeds. No existing different object is overwritten.
/// Failed or cancelled calls may leave reusable chunks, but do not publish a
/// manifest before its chunks have completed. Caller-supplied chunk files must
/// remain stable throughout publication.
pub async fn publish_directory(
    storage: &Storage,
    directory: &Path,
    namespace: &str,
    concurrency: usize,
) -> Result<PublicationReceipt> {
    if !valid_key_prefix(namespace) || !(1..=32).contains(&concurrency) {
        return Err(Error::Config(
            "invalid publication prefix or concurrency".into(),
        ));
    }
    let snapshot = Snapshot::capture(&directory.join("manifest.json"), Manifest::validate)?;
    let manifest = &snapshot.manifest;
    let prefix = format!("{namespace}/{}", manifest.id);
    let chunks_created = {
        let operations = futures_util::stream::iter(manifest.chunks.iter().cloned())
            .map(|chunk| {
                let storage = storage.clone();
                let key = format!("{prefix}/{}", chunk.name);
                let path = directory.join(&chunk.name);
                async move {
                    Ok::<_, Error>((
                        chunk.name,
                        storage
                            .put_file_immutable(&key, &path, &chunk.sha256, chunk.size)
                            .await?,
                    ))
                }
            })
            .buffer_unordered(concurrency);
        tokio::pin!(operations);
        let mut chunks_created = BTreeMap::new();
        while let Some(result) = operations.next().await {
            let (name, created) = result?;
            chunks_created.insert(name, created);
        }
        chunks_created
    };
    let manifest_created = storage
        .put_file_immutable(
            &format!("{prefix}/manifest.json"),
            &snapshot.path,
            &snapshot.sha256,
            snapshot.size,
        )
        .await?;
    Ok(PublicationReceipt {
        id: manifest.id.clone(),
        prefix,
        manifest_created,
        chunks_created,
    })
}

pub(crate) struct Snapshot<M> {
    pub(crate) manifest: M,
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
    pub(crate) size: u64,
}

impl<M: serde::de::DeserializeOwned> Snapshot<M> {
    pub(crate) fn capture(
        source: &Path,
        validate: impl FnOnce(&M) -> switchyard_artifacts::Result<()>,
    ) -> Result<Self> {
        let bytes = fs::read(source)?;
        let manifest: M = serde_json::from_slice(&bytes)?;
        validate(&manifest)?;
        let mut random = [0_u8; 12];
        getrandom::fill(&mut random)
            .map_err(|_| Error::Config("manifest snapshot randomness unavailable".into()))?;
        let nonce = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let path = std::env::temp_dir().join(format!("switchyard-manifest-{nonce}.json"));
        let mut options = File::options();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path)?;
        // Establish cleanup ownership immediately, including failed writes.
        let snapshot = Self {
            manifest,
            path,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            size: bytes.len() as u64,
        };
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(snapshot)
    }
}

impl<M> Drop for Snapshot<M> {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use switchyard_artifacts::{Metadata, PackOptions, pack};

    #[test]
    fn captured_manifest_has_private_permissions_and_is_removed_on_drop() {
        let mut random = [0_u8; 12];
        getrandom::fill(&mut random).unwrap();
        let nonce = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let root = std::env::temp_dir().join(format!("switchyard-snapshot-test-{nonce}"));
        let source = root.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("file"), "contents").unwrap();
        let manifest = pack(
            &source,
            &root.join("packed"),
            Metadata {
                kind: "test".into(),
                compatibility: BTreeMap::new(),
                attributes: BTreeMap::new(),
            },
            PackOptions::default(),
        )
        .unwrap();
        let path = root.join("manifest.json");
        let bytes = serde_json::to_vec(&manifest).unwrap();
        fs::write(&path, &bytes).unwrap();
        let snapshot = Snapshot::capture(&path, Manifest::validate).unwrap();
        let temporary = snapshot.path.clone();
        assert_eq!(fs::read(&temporary).unwrap(), bytes);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&temporary).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        drop(snapshot);
        assert!(!temporary.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
