//! Credit only allocated blocks belonging to this boot's model output.
use crate::{Error, Result};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

type AllocationQuery = Arc<dyn Fn() -> Result<u64> + Send + Sync>;

#[derive(Clone, Default)]
pub struct ModelAllocation(Arc<Mutex<Option<AllocationQuery>>>);
impl std::fmt::Debug for ModelAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ModelAllocation")
    }
}
impl ModelAllocation {
    pub(crate) fn register(&self, query: impl Fn() -> Result<u64> + Send + Sync + 'static) {
        *self.0.lock().unwrap() = Some(Arc::new(query));
    }
}

/// The caller supplies an exclusively owned root and validated manifest paths.
/// Missing files contribute no credit; sparse length contributes no credit.
pub(crate) fn allocated_files(root: &Path, files: &[(PathBuf, u64)]) -> Result<u64> {
    use std::os::unix::fs::MetadataExt;
    files.iter().try_fold(0_u64, |sum, (path, size)| {
        let metadata = match fs::symlink_metadata(root.join(path)) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(sum),
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_file() || metadata.len() > *size {
            return Err(Error::Manifest(
                "owned model allocation has invalid type or length".into(),
            ));
        }
        let allocated = metadata
            .blocks()
            .checked_mul(512)
            .ok_or_else(|| Error::Manifest("model allocation overflow".into()))?
            .min(*size);
        sum.checked_add(allocated)
            .ok_or_else(|| Error::Manifest("model allocation overflow".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, os::unix::fs::symlink};
    #[test]
    fn sparse_length_does_not_claim_allocation_and_blocks_are_capped() {
        let root = std::env::temp_dir().join(format!("model-allocation-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let mut file = fs::File::create(root.join("weights")).unwrap();
        let files = vec![(PathBuf::from("weights"), 1024 * 1024)];
        file.set_len(1024 * 1024).unwrap();
        assert_eq!(allocated_files(&root, &files).unwrap(), 0);
        file.write_all(&[1; 4096]).unwrap();
        let allocated = allocated_files(&root, &files).unwrap();
        assert!((4096..=1024 * 1024).contains(&allocated));
        fs::write(root.join("small"), [1]).unwrap();
        assert_eq!(allocated_files(&root, &[("small".into(), 1)]).unwrap(), 1);
        assert!(allocated_files(&root, &[("weights".into(), 1)]).is_err());
        drop(file);
        fs::remove_file(root.join("weights")).unwrap();
        symlink("/etc/passwd", root.join("weights")).unwrap();
        assert!(allocated_files(&root, &files).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
