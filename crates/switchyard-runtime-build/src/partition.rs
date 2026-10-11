//! Declarative relocation of complete directories without pruning source payload.
use crate::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use switchyard_artifacts::{Entry, Manifest};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub format: String,
    pub schema_version: u32,
    pub runtime_root: String,
    pub components_root: String,
    pub requires: BTreeMap<String, String>,
    pub components: Vec<Component>,
    pub environment: BTreeMap<String, String>,
    pub library_paths: Vec<String>,
    pub external_libraries: BTreeSet<String>,
    pub boot_imports: Vec<String>,
    pub resource_checks: Vec<ResourceCheck>,
    #[serde(default)]
    pub compiler_checks: Vec<CompilerCheck>,
    #[serde(default)]
    pub toolchain_wrappers: Vec<ToolchainWrappers>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub name: String,
    pub moves: Vec<Move>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Move {
    pub source: String,
    pub destination: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolchainWrappers {
    pub component: String,
    pub path: String,
    pub gcc_major: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceCheck {
    pub module: String,
    pub attribute: String,
    pub component: String,
    pub path: String,
    pub loader: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompilerCheck {
    pub name: String,
    pub command: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct PlacedEntry {
    pub source_path: String,
    pub component: String,
    #[serde(flatten)]
    pub entry: Entry,
}
#[derive(Default, Debug, Serialize)]
pub struct Statistics {
    pub entries: u64,
    pub files: u64,
    pub file_bytes: u64,
}
#[derive(Debug, Serialize)]
pub struct Plan {
    pub source_runtime_id: String,
    pub recipe_sha256: String,
    pub entries: Vec<PlacedEntry>,
    pub statistics: BTreeMap<String, Statistics>,
}

pub fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}
pub fn within(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|p| p.starts_with('/'))
}
pub fn relative(path: &str) -> Result<()> {
    if path.is_empty()
        || path.len() > 4096
        || path.contains(['\0', '\\', ':'])
        || path.split('/').any(|p| matches!(p, "" | "." | ".."))
    {
        return Err(format!("invalid normalized relative path: {path}").into());
    }
    Ok(())
}
fn absolute(path: &str) -> Result<()> {
    if !path
        .bytes()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'/' | b'_' | b'-' | b'.'))
    {
        return Err("publication roots must use plain absolute path components".into());
    }
    relative(
        path.strip_prefix('/')
            .ok_or("publication root must be absolute")?,
    )
}
pub fn link_destination(path: &str, target: &str) -> Result<String> {
    if target.is_empty() || target.starts_with('/') || target.contains(['\0', '\\', ':']) {
        return Err("invalid relative link target".into());
    }
    let mut parts: Vec<_> = path.split('/').collect();
    parts.pop();
    for part in target.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop().ok_or("link escapes artifact")?;
            }
            _ => parts.push(part),
        }
    }
    let destination = parts.join("/");
    relative(&destination)?;
    Ok(destination)
}
fn relative_link(path: &str, destination: &str) -> String {
    let mut parent: Vec<_> = path.split('/').collect();
    parent.pop();
    let target: Vec<_> = destination.split('/').collect();
    let shared = parent
        .iter()
        .zip(&target)
        .take_while(|(a, b)| a == b)
        .count();
    let mut result = vec![".."; parent.len() - shared];
    result.extend_from_slice(&target[shared..]);
    result.join("/")
}
fn set_path(entry: &mut Entry, value: String) {
    match entry {
        Entry::File { path, .. } | Entry::Directory { path, .. } | Entry::Symlink { path, .. } => {
            *path = value
        }
    }
}

impl Recipe {
    pub fn expand(&self, value: &str) -> String {
        value
            .replace("{runtime_root}", &self.runtime_root)
            .replace("{components_root}", &self.components_root)
    }
    pub fn plan(&self, manifest: &Manifest) -> Result<Plan> {
        manifest.validate()?;
        if self.format != "switchyard-runtime-partition-v1" || self.schema_version != 1 {
            return Err("unsupported runtime partition recipe".into());
        }
        absolute(&self.runtime_root)?;
        absolute(&self.components_root)?;
        if within(&self.runtime_root, &self.components_root)
            || within(&self.components_root, &self.runtime_root)
            || manifest.metadata.attributes.get("runtime_root") != Some(&self.runtime_root)
        {
            return Err("runtime and components require disjoint pinned publication roots".into());
        }
        for (key, value) in &self.requires {
            if manifest.metadata.compatibility.get(key) != Some(value) {
                return Err(format!("source compatibility mismatch: {key}").into());
            }
        }
        let original: BTreeMap<_, _> = manifest.entries.iter().map(|e| (e.path(), e)).collect();
        for reserved in ["partition-source", "runtime-components.json"] {
            if original.keys().any(|p| within(p, reserved)) {
                return Err(format!("source reserves generated path: {reserved}").into());
            }
        }
        for name in ["artifact-metadata.json", "runtime-env.sh"] {
            if !matches!(original.get(name), Some(Entry::File { .. })) {
                return Err(format!("source lacks regular {name}").into());
            }
        }
        let mut names = BTreeSet::new();
        let mut sources: Vec<&str> = vec![];
        for component in &self.components {
            if component.name.is_empty()
                || component.name == "boot"
                || !component
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
                || !names.insert(&component.name)
                || component.moves.is_empty()
            {
                return Err("invalid or duplicate component name".into());
            }
            let mut destinations: Vec<&str> = vec![];
            for rule in &component.moves {
                relative(&rule.source)?;
                relative(&rule.destination)?;
                if !matches!(
                    original.get(rule.source.as_str()),
                    Some(Entry::Directory { .. })
                ) {
                    return Err("moves require complete source directories".into());
                }
                if sources
                    .iter()
                    .any(|old| within(&rule.source, old) || within(old, &rule.source))
                    || destinations
                        .iter()
                        .any(|old| within(&rule.destination, old) || within(old, &rule.destination))
                    || within("artifact-metadata.json", &rule.destination)
                    || within(&rule.destination, "artifact-metadata.json")
                {
                    return Err("overlapping or reserved relocations".into());
                }
                sources.push(&rule.source);
                destinations.push(&rule.destination);
            }
        }
        if sources.is_empty() {
            return Err("partition requires a supplemental payload".into());
        }
        for check in &self.resource_checks {
            if !names.contains(&check.component) {
                return Err("resource check names an undeclared component".into());
            }
            relative(&check.path)?;
        }
        for wrapper in &self.toolchain_wrappers {
            if !names.contains(&wrapper.component) || wrapper.gcc_major == 0 {
                return Err("invalid toolchain wrapper recipe".into());
            }
            relative(&wrapper.path)?;
        }
        for (key, value) in &self.environment {
            if key.is_empty()
                || key.as_bytes()[0].is_ascii_digit()
                || !key.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
                || value.contains('\0')
                || self.expand(value).contains(['{', '}'])
            {
                return Err("invalid recipe environment".into());
            }
        }
        let place = |path: &str| -> (String, String) {
            for c in &self.components {
                for rule in &c.moves {
                    if within(path, &rule.source) {
                        return (
                            c.name.clone(),
                            format!("{}{}", rule.destination, &path[rule.source.len()..]),
                        );
                    }
                }
            }
            let path = if matches!(path, "artifact-metadata.json" | "runtime-env.sh") {
                format!("partition-source/{path}")
            } else {
                path.into()
            };
            ("boot".into(), path)
        };
        let mut entries = vec![];
        let mut mapped = BTreeMap::new();
        let mut statistics: BTreeMap<String, Statistics> = BTreeMap::new();
        let wrappers: BTreeSet<_> = self
            .toolchain_wrappers
            .iter()
            .flat_map(|w| {
                ["gcc", "g++"].map(|name| (w.component.clone(), format!("{}/bin/{name}", w.path)))
            })
            .collect();
        for key in &wrappers {
            mapped.insert(key.clone(), usize::MAX);
        }
        for original in &manifest.entries {
            let (component, mut path) = place(original.path());
            if wrappers.contains(&(component.clone(), path.clone())) {
                if !matches!(original, Entry::File { .. }) {
                    return Err("toolchain wrapper must replace a regular file".into());
                }
                path = format!("partition-source/{path}");
            }
            if original.path().starts_with("partition-source/") {
                return Err("component source reserves partition provenance".into());
            }
            let mut entry = original.clone();
            set_path(&mut entry, path.clone());
            if let Entry::Symlink { target, .. } = &mut entry {
                let original_destination = link_destination(original.path(), target)?;
                let (target_component, target_path) = place(&original_destination);
                if target_component != component {
                    return Err(format!("cross-artifact symlink: {}", original.path()).into());
                }
                *target = relative_link(&path, &target_path);
            }
            if mapped
                .insert((component.clone(), path), entries.len())
                .is_some()
            {
                return Err("relocation path collision".into());
            }
            let stats = statistics.entry(component.clone()).or_default();
            stats.entries += 1;
            if let Entry::File { size, .. } = entry {
                stats.files += 1;
                stats.file_bytes += size;
            }
            entries.push(PlacedEntry {
                source_path: original.path().into(),
                component,
                entry,
            });
        }
        for placed in &entries {
            let mut destination = placed.entry.path().to_owned();
            let mut seen = BTreeSet::new();
            loop {
                let Some(index) = mapped.get(&(placed.component.clone(), destination.clone()))
                else {
                    return Err("dangling partition symlink".into());
                };
                if *index == usize::MAX {
                    break;
                }
                let Entry::Symlink { target, .. } = &entries[*index].entry else {
                    break;
                };
                if !seen.insert(destination.clone()) {
                    return Err("cyclic partition symlink".into());
                }
                destination = link_destination(&destination, target)?;
            }
        }
        for (component, path) in &wrappers {
            if !mapped.contains_key(&(component.clone(), format!("partition-source/{path}"))) {
                return Err("recipe names a missing compiler wrapper".into());
            }
        }
        Ok(Plan {
            source_runtime_id: manifest.id.clone(),
            recipe_sha256: digest(self)?,
            entries,
            statistics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::Fixture;
    use std::fs;

    #[test]
    fn partitions_exact_source_union_and_preserves_original_metadata_and_wrappers() {
        let fixture = Fixture::new();
        for name in ["gcc", "g++"] {
            fs::write(
                fixture.source.join("compiler/bin").join(name),
                "original compiler wrapper",
            )
            .unwrap();
        }
        std::os::unix::fs::symlink(
            "../lib/payload",
            fixture.source.join("compiler/bin/header-link"),
        )
        .unwrap();
        let mut recipe = fixture.recipe();
        recipe.toolchain_wrappers.push(ToolchainWrappers {
            component: "build".into(),
            path: "toolchain".into(),
            gcc_major: 13,
        });
        let manifest = fixture.manifest();
        let plan = recipe.plan(&manifest).unwrap();
        assert_eq!(plan.entries.len(), manifest.entries.len());
        assert_eq!(
            plan.entries
                .iter()
                .map(|e| &e.source_path)
                .collect::<BTreeSet<_>>()
                .len(),
            manifest.entries.len()
        );
        assert_eq!(
            plan.entries
                .iter()
                .find(|e| e.source_path == "artifact-metadata.json")
                .unwrap()
                .entry
                .path(),
            "partition-source/artifact-metadata.json"
        );
        assert_eq!(
            plan.entries
                .iter()
                .find(|e| e.source_path == "compiler/bin/gcc")
                .unwrap()
                .entry
                .path(),
            "partition-source/toolchain/bin/gcc"
        );
        assert_eq!(
            plan.entries
                .iter()
                .find(|e| e.source_path == "compiler/lib/payload")
                .unwrap()
                .component,
            "build"
        );
        for placed in &plan.entries {
            let original = manifest
                .entries
                .iter()
                .find(|e| e.path() == placed.source_path)
                .unwrap();
            assert_eq!(original.mode(), placed.entry.mode());
            if let (
                Entry::File {
                    sha256: a,
                    size: sa,
                    ..
                },
                Entry::File {
                    sha256: b,
                    size: sb,
                    ..
                },
            ) = (original, &placed.entry)
            {
                assert_eq!((a, sa), (b, sb));
            }
        }
        assert_eq!(digest(&recipe).unwrap(), plan.recipe_sha256);
    }
    #[test]
    fn refuses_cross_component_links_and_overlapping_or_missing_moves() {
        let fixture = Fixture::new();
        std::os::unix::fs::symlink(
            "../../runtime-env.sh",
            fixture.source.join("compiler/bin/boot-link"),
        )
        .unwrap();
        assert!(
            fixture
                .recipe()
                .plan(&fixture.manifest())
                .unwrap_err()
                .to_string()
                .contains("cross-artifact")
        );
        fs::remove_file(fixture.source.join("compiler/bin/boot-link")).unwrap();
        let manifest = fixture.manifest();
        let mut recipe = fixture.recipe();
        recipe.components[0].moves.push(Move {
            source: "compiler/lib".into(),
            destination: "other".into(),
        });
        assert!(recipe.plan(&manifest).is_err());
        recipe.components[0].moves[0].source = "missing".into();
        assert!(recipe.plan(&manifest).is_err());
    }
    #[test]
    fn refuses_incompatible_sources_nested_roots_and_reserved_names() {
        let fixture = Fixture::new();
        let manifest = fixture.manifest();
        let mut recipe = fixture.recipe();
        recipe
            .requires
            .insert("vllm".into(), "another version".into());
        assert!(recipe.plan(&manifest).is_err());
        recipe = fixture.recipe();
        recipe.components_root = "/opt/switchyard/runtime/components".into();
        assert!(recipe.plan(&manifest).is_err());
        recipe.components_root = "/opt/switchyard/components/../runtime".into();
        assert!(recipe.plan(&manifest).is_err());
        recipe = fixture.recipe();
        recipe.components[0].name = "boot".into();
        assert!(recipe.plan(&manifest).is_err());
        fs::create_dir(fixture.source.join("partition-source")).unwrap();
        assert!(fixture.recipe().plan(&fixture.manifest()).is_err());
    }
}
