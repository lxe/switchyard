//! Preserve the validated manifest bytes throughout an immutable publication.
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};
use switchyard_artifacts::Manifest;

pub(crate) struct Snapshot {
    pub(crate) manifest: Manifest,
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
    pub(crate) size: u64,
}

impl Snapshot {
    pub(crate) fn capture(source: &Path) -> Result<Self> {
        let bytes = fs::read(source)?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        manifest.validate()?;
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

impl Drop for Snapshot {
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
        let snapshot = Snapshot::capture(&path).unwrap();
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
