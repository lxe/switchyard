//! ELF dependency closure relative to an explicit runtime inventory and loader paths.
use crate::{
    Result,
    partition::{PlacedEntry, Recipe, link_destination},
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
    process::Command,
};
use switchyard_artifacts::Entry;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Missing {
    pub elf: String,
    pub needed: String,
}
#[derive(Debug, Serialize)]
pub struct Dependency {
    pub elf: String,
    pub needed: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved: Option<String>,
    pub external: bool,
}
#[derive(Debug, Serialize)]
pub struct Closure {
    pub elf_files: u64,
    pub library_paths: Vec<String>,
    pub external_libraries_used: BTreeSet<String>,
    pub unresolved: Vec<Missing>,
    pub dependencies: Vec<Dependency>,
    pub source_unresolved: Vec<Missing>,
    pub newly_unresolved: Vec<Missing>,
}

pub fn library_paths(recipe: &Recipe, inventory: &[PlacedEntry]) -> Result<Vec<String>> {
    let directories: BTreeSet<_> = inventory
        .iter()
        .filter(|e| e.component == "boot" && matches!(e.entry, Entry::Directory { .. }))
        .map(|e| e.entry.path())
        .collect();
    let mut result = vec![];
    for pattern in &recipe.library_paths {
        let pattern: Vec<_> = pattern.split('/').collect();
        let matches: Vec<_> = directories
            .iter()
            .filter(|path| {
                let path: Vec<_> = path.split('/').collect();
                path.len() == pattern.len()
                    && path.iter().zip(&pattern).all(|(p, q)| *q == "*" || p == q)
            })
            .collect();
        if matches.is_empty() {
            return Err(format!(
                "no boot directory matches library path: {}",
                pattern.join("/")
            )
            .into());
        }
        for path in matches {
            if !result.iter().any(|p| p == *path) {
                result.push((*path).to_string());
            }
        }
    }
    Ok(result)
}
fn normalized(value: &str) -> Option<String> {
    if value.starts_with('/') || value.contains('$') {
        return None;
    }
    let mut parts = vec![];
    for part in value.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            _ => parts.push(part),
        }
    }
    Some(parts.join("/"))
}
fn resolve(mut path: String, entries: &BTreeMap<&str, &Entry>) -> Result<Option<String>> {
    for _ in 0..256 {
        let parts: Vec<_> = path.split('/').collect();
        let mut changed = false;
        for i in 1..=parts.len() {
            let prefix = parts[..i].join("/");
            if let Some(Entry::Symlink { target, .. }) = entries.get(prefix.as_str()) {
                let target = link_destination(&prefix, target)?;
                path = [
                    vec![target],
                    parts[i..].iter().map(|s| s.to_string()).collect(),
                ]
                .concat()
                .join("/");
                changed = true;
                break;
            }
        }
        if !changed {
            return Ok(
                matches!(entries.get(path.as_str()), Some(Entry::File { .. })).then_some(path),
            );
        }
    }
    Err("cyclic ELF search path".into())
}
pub fn inspect(source: &Path, inventory: &[PlacedEntry], recipe: &Recipe) -> Result<Closure> {
    let entries: BTreeMap<_, _> = inventory
        .iter()
        .filter(|e| e.component == "boot")
        .map(|e| (e.entry.path(), &e.entry))
        .collect();
    let search = library_paths(recipe, inventory)?;
    let mut out = Closure {
        elf_files: 0,
        library_paths: search.clone(),
        external_libraries_used: BTreeSet::new(),
        unresolved: vec![],
        dependencies: vec![],
        source_unresolved: vec![],
        newly_unresolved: vec![],
    };
    for entry in inventory
        .iter()
        .filter(|e| e.component == "boot" && matches!(e.entry, Entry::File { .. }))
    {
        let mut header = [0; 20];
        let mut file = File::open(source.join(&entry.source_path))?;
        if file.read(&mut header)? < 20
            || header[..4] != *b"\x7fELF"
            || u16::from_le_bytes([header[18], header[19]]) != 62
        {
            continue;
        }
        let output = Command::new("readelf")
            .args(["--wide", "--dynamic"])
            .arg(source.join(&entry.source_path))
            .output()?;
        if !output.status.success() {
            return Err(format!("readelf failed: {}", entry.source_path).into());
        }
        out.elf_files += 1;
        let text = String::from_utf8(output.stdout)?;
        let origin = entry.entry.path().rsplit_once('/').map_or("", |(p, _)| p);
        let mut needed = vec![];
        let mut local = vec![];
        for line in text.lines() {
            let Some(value) = line
                .split_once('[')
                .and_then(|(_, v)| v.split_once(']').map(|(v, _)| v))
            else {
                continue;
            };
            if line.contains("(NEEDED)") {
                needed.push(value);
            }
            if line.contains("(RUNPATH)") || line.contains("(RPATH)") {
                for path in value.split(':') {
                    let path = path.replace("${ORIGIN}", origin).replace("$ORIGIN", origin);
                    let prefix = recipe.runtime_root.clone() + "/";
                    if let Some(path) = normalized(path.strip_prefix(&prefix).unwrap_or(&path)) {
                        local.push(path);
                    }
                }
            }
        }
        for dependency in needed {
            let direct = dependency
                .replace("${ORIGIN}", origin)
                .replace("$ORIGIN", origin);
            let mut resolved = if direct.contains('/') {
                match normalized(&direct) {
                    Some(path) => resolve(path, &entries)?,
                    None => None,
                }
            } else {
                None
            };
            for directory in search.iter().chain(&local) {
                if resolved.is_some() {
                    break;
                }
                resolved = resolve(format!("{directory}/{dependency}"), &entries)?;
            }
            let external = resolved.is_none() && recipe.external_libraries.contains(dependency);
            if external {
                out.external_libraries_used.insert(dependency.into());
            }
            if resolved.is_none() && !external {
                out.unresolved.push(Missing {
                    elf: entry.entry.path().into(),
                    needed: dependency.into(),
                });
            } else {
                out.dependencies.push(Dependency {
                    elf: entry.entry.path().into(),
                    needed: dependency.into(),
                    resolved,
                    external,
                });
            }
        }
    }
    Ok(out)
}

/// Preserve baseline optional plugins, but refuse any dependency lost by relocation.
pub fn compare(
    source: &Path,
    inventory: &[PlacedEntry],
    original: &[Entry],
    recipe: &Recipe,
) -> Result<Closure> {
    let mut boot = inspect(source, inventory, recipe)?;
    let full: Vec<_> = original
        .iter()
        .map(|entry| PlacedEntry {
            source_path: entry.path().into(),
            component: "boot".into(),
            entry: entry.clone(),
        })
        .collect();
    let full = inspect(source, &full, recipe)?;
    let baseline: BTreeSet<_> = full
        .unresolved
        .iter()
        .map(|e| (&e.elf, &e.needed))
        .collect();
    let originals: BTreeMap<_, _> = inventory
        .iter()
        .filter(|e| e.component == "boot")
        .map(|e| (e.entry.path(), &e.source_path))
        .collect();
    boot.newly_unresolved = boot
        .unresolved
        .iter()
        .filter(|e| !baseline.contains(&(originals[e.elf.as_str()], &e.needed)))
        .cloned()
        .collect();
    boot.source_unresolved = full.unresolved;
    Ok(boot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::Fixture;
    #[test]
    fn detects_actual_elf_library_lost_from_boot_after_relocation() {
        let fixture = Fixture::new();
        std::fs::write(
            fixture.root.join("needed.c"),
            "int needed(void) { return 7; }\n",
        )
        .unwrap();
        std::fs::write(
            fixture.root.join("extension.c"),
            "extern int needed(void); int extension(void) { return needed(); }\n",
        )
        .unwrap();
        assert!(
            Command::new("cc")
                .args(["-shared", "-fPIC", "-Wl,-soname,libpartition_needed.so"])
                .arg(fixture.root.join("needed.c"))
                .arg("-o")
                .arg(fixture.source.join("compiler/lib/libpartition_needed.so"))
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("cc")
                .args(["-shared", "-fPIC", "-Wl,-rpath,$ORIGIN/../compiler/lib"])
                .arg(fixture.root.join("extension.c"))
                .arg("-L")
                .arg(fixture.source.join("compiler/lib"))
                .arg("-lpartition_needed")
                .arg("-o")
                .arg(fixture.source.join("lib/extension.so"))
                .status()
                .unwrap()
                .success()
        );
        let manifest = fixture.manifest();
        let recipe = fixture.recipe();
        let plan = recipe.plan(&manifest).unwrap();
        let closure = compare(&fixture.source, &plan.entries, &manifest.entries, &recipe).unwrap();
        assert!(closure.source_unresolved.is_empty());
        assert_eq!(
            closure.newly_unresolved,
            vec![Missing {
                elf: "lib/extension.so".into(),
                needed: "libpartition_needed.so".into()
            }]
        );
    }
}
