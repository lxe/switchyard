use crate::partition::Recipe;
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use switchyard_artifacts::{Manifest, Metadata, PackOptions};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub struct Fixture {
    pub root: PathBuf,
    pub source: PathBuf,
}
impl Fixture {
    pub fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "switchyard-layout-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let source = root.join("source");
        for path in ["lib", "compiler/lib", "compiler/bin"] {
            fs::create_dir_all(source.join(path)).unwrap();
        }
        fs::write(source.join("artifact-metadata.json"), "source metadata").unwrap();
        fs::write(source.join("runtime-env.sh"), "source environment\n").unwrap();
        fs::write(
            source.join("compiler/lib/payload"),
            "compiler headers and libraries",
        )
        .unwrap();
        Self { root, source }
    }
    pub fn recipe(&self) -> Recipe {
        serde_json::from_value(json!({"format":"switchyard-runtime-partition-v1","schema_version":1,
            "runtime_root":"/opt/switchyard/runtime","components_root":"/opt/switchyard/components","requires":{"vllm":"fixture"},
            "components":[{"name":"build","moves":[{"source":"compiler","destination":"toolchain"}]}],
            "environment":{},"library_paths":["lib"],"external_libraries":["libc.so.6"],"boot_imports":[],"resource_checks":[]})).unwrap()
    }
    pub fn manifest(&self) -> Manifest {
        let metadata: Metadata = serde_json::from_value(json!({"kind":"runtime","compatibility":{"vllm":"fixture"},"attributes":{"runtime_root":"/opt/switchyard/runtime"}})).unwrap();
        let path = self
            .root
            .join(format!("packed-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
        switchyard_artifacts::pack(&self.source, &path, metadata, PackOptions::default()).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
