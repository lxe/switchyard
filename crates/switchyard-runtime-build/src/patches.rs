//! Source patches applied once during an artifact bake, never at worker startup.

use crate::{Result, package_version, sha256};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    schema_version: u32,
    id: String,
    package: Package,
    target: String,
    before_sha256: String,
    after_sha256: String,
    replacements: Vec<Replacement>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Package {
    name: String,
    version: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Replacement {
    before: String,
    after: String,
}

#[derive(Serialize)]
pub struct Receipt {
    pub id: String,
    pub descriptor_sha256: String,
    package: Package,
    pub target: String,
    pub before_sha256: String,
    pub after_sha256: String,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn relative_file(root: &Path, path: &str) -> Result<std::path::PathBuf> {
    if path.is_empty()
        || Path::new(path)
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("patch path must be a normalized relative file: {path}").into());
    }
    let normalized: std::path::PathBuf = Path::new(path).components().collect();
    if normalized.as_os_str() != path {
        return Err(format!("patch path must use normalized separators: {path}").into());
    }
    let root = root.canonicalize()?;
    let file = root.join(path);
    // Reject symlinks in every component, including aliases within the root.
    let mut prefix = root.clone();
    for component in Path::new(path).components() {
        prefix.push(component);
        if fs::symlink_metadata(&prefix)?.file_type().is_symlink() {
            return Err(format!("patch path traverses a symlink: {path}").into());
        }
    }
    if !file.is_file() || !file.canonicalize()?.starts_with(&root) {
        return Err(format!("patch target is not a regular file within its root: {path}").into());
    }
    Ok(file)
}

/// Validate the entire patch set before modifying any source file.
///
/// An absent `source_patches` field means no patches. Each entry pins a JSON
/// descriptor relative to the build directory. Targets are distinct files
/// relative to the runtime root; patches cannot silently stack on one file.
pub fn apply(build: &Path, root: &Path, lock: &Value, packages: &Value) -> Result<Vec<Receipt>> {
    let pins: Vec<Pin> = match lock.get("source_patches") {
        None => return Ok(Vec::new()),
        Some(value) => serde_json::from_value(value.clone())?,
    };
    let mut ids = HashSet::new();
    let mut targets = HashSet::new();
    let mut pending = Vec::new();
    for pin in pins {
        let descriptor_path = relative_file(build, &pin.path)?;
        let bytes = fs::read(descriptor_path)?;
        if digest(&bytes) != pin.sha256 {
            return Err(format!("patch descriptor hash mismatch: {}", pin.path).into());
        }
        let descriptor: Descriptor = serde_json::from_slice(&bytes)?;
        if descriptor.schema_version != 1
            || descriptor.id.is_empty()
            || !ids.insert(descriptor.id.clone())
            || !targets.insert(descriptor.target.clone())
            || descriptor.replacements.is_empty()
        {
            return Err("invalid or duplicate source patch descriptor".into());
        }
        if package_version(packages, &descriptor.package.name)? != descriptor.package.version {
            return Err(format!("patch package version mismatch: {}", descriptor.id).into());
        }
        let target = relative_file(root, &descriptor.target)?;
        let original = fs::read(&target)?;
        if digest(&original) != descriptor.before_sha256 {
            return Err(format!("patch source hash mismatch: {}", descriptor.id).into());
        }
        let mut output = String::from_utf8(original)?;
        for replacement in descriptor.replacements {
            if replacement.before.is_empty() || output.matches(&replacement.before).count() != 1 {
                return Err(format!(
                    "patch replacement must match exactly once: {}",
                    descriptor.id
                )
                .into());
            }
            output = output.replacen(&replacement.before, &replacement.after, 1);
        }
        if digest(output.as_bytes()) != descriptor.after_sha256 {
            return Err(format!("patch result hash mismatch: {}", descriptor.id).into());
        }
        pending.push((
            target,
            output,
            Receipt {
                id: descriptor.id,
                descriptor_sha256: pin.sha256,
                package: descriptor.package,
                target: descriptor.target,
                before_sha256: descriptor.before_sha256,
                after_sha256: descriptor.after_sha256,
            },
        ));
    }
    let mut receipts = Vec::new();
    for (target, output, receipt) in pending {
        // A failed bake is discarded. Preserve permissions for successful files;
        // no partially validated patch set is ever written.
        fs::write(&target, output)?;
        if sha256(&target)? != receipt.after_sha256 {
            return Err(format!("written patch hash mismatch: {}", receipt.id).into());
        }
        receipts.push(receipt);
    }
    Ok(receipts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "switchyard-patches-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(path.join("build")).unwrap();
            fs::create_dir_all(path.join("runtime")).unwrap();
            Self(path)
        }
        fn pin(&self, name: &str, target: &str, before: &str, after: &str) -> Value {
            let descriptor = json!({"schema_version":1,"id":name,
                "package":{"name":"engine","version":"1.0"},"target":target,
                "before_sha256":digest(before.as_bytes()),"after_sha256":digest(after.as_bytes()),
                "replacements":[{"before":before,"after":after}]});
            let bytes = serde_json::to_vec(&descriptor).unwrap();
            fs::write(self.0.join("build").join(name), &bytes).unwrap();
            json!({"path":name,"sha256":digest(&bytes)})
        }
        fn apply(&self, pins: Value) -> Result<Vec<Receipt>> {
            apply(
                &self.0.join("build"),
                &self.0.join("runtime"),
                &json!({"source_patches":pins}),
                &json!([{"name":"engine","version":"1.0"}]),
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn applies_pinned_source_and_records_identity() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join("runtime/source.py"), "original").unwrap();
        let pin = fixture.pin("patch.json", "source.py", "original", "patched");
        let receipts = fixture.apply(json!([pin])).unwrap();
        assert_eq!(
            fs::read_to_string(fixture.0.join("runtime/source.py")).unwrap(),
            "patched"
        );
        assert_eq!(receipts[0].id, "patch.json");
        assert!(
            fixture
                .apply(json!([fixture.pin(
                    "again.json",
                    "source.py",
                    "original",
                    "patched"
                )]))
                .is_err()
        );
    }

    #[test]
    fn invalid_later_patch_leaves_all_sources_unchanged() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join("runtime/first.py"), "first").unwrap();
        fs::write(fixture.0.join("runtime/second.py"), "second").unwrap();
        let first = fixture.pin("first.json", "first.py", "first", "changed");
        let second = fixture.pin("second.json", "second.py", "wrong", "changed");
        assert!(fixture.apply(json!([first, second])).is_err());
        assert_eq!(
            fs::read_to_string(fixture.0.join("runtime/first.py")).unwrap(),
            "first"
        );
    }

    #[test]
    fn rejects_duplicate_targets_and_path_escape() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join("runtime/source.py"), "original").unwrap();
        let first = fixture.pin("first.json", "source.py", "original", "patched");
        let second = fixture.pin("second.json", "source.py", "original", "patched");
        assert!(fixture.apply(json!([first, second])).is_err());
        let escape = fixture.pin("escape.json", "../build/escape.json", "original", "patched");
        assert!(fixture.apply(json!([escape])).is_err());
    }

    #[test]
    fn rejects_descriptor_version_and_result_drift_without_writes() {
        for kind in ["descriptor", "package", "result", "ambiguous"] {
            let fixture = Fixture::new();
            let original = if kind == "ambiguous" {
                "old old"
            } else {
                "old"
            };
            fs::write(fixture.0.join("runtime/source.py"), original).unwrap();
            let mut pin = fixture.pin("patch.json", "source.py", original, "new");
            let descriptor_path = fixture.0.join("build/patch.json");
            if kind == "descriptor" {
                pin["sha256"] = json!("wrong");
            } else {
                let mut descriptor: Value =
                    serde_json::from_slice(&fs::read(&descriptor_path).unwrap()).unwrap();
                match kind {
                    "package" => descriptor["package"]["version"] = json!("2.0"),
                    "result" => descriptor["after_sha256"] = json!(digest(b"different")),
                    "ambiguous" => descriptor["replacements"][0]["before"] = json!("old"),
                    _ => unreachable!(),
                }
                let bytes = serde_json::to_vec(&descriptor).unwrap();
                fs::write(&descriptor_path, &bytes).unwrap();
                pin["sha256"] = json!(digest(&bytes));
            }
            assert!(fixture.apply(json!([pin])).is_err(), "{kind}");
            assert_eq!(
                fs::read_to_string(fixture.0.join("runtime/source.py")).unwrap(),
                original
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_target() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join("runtime/source.py"), "original").unwrap();
        std::os::unix::fs::symlink("source.py", fixture.0.join("runtime/alias.py")).unwrap();
        let pin = fixture.pin("patch.json", "alias.py", "original", "patched");
        assert!(fixture.apply(json!([pin])).is_err());
    }
}
