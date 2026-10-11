//! Runtime composition uses the ordinary archive packer and private local staging.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
};
use switchyard_artifacts::{Entry, Manifest, Metadata, PackOptions};
use switchyard_runtime_build::{
    native_closure,
    partition::{Plan, Recipe, digest},
    sha256,
};

fn preparation(error: Box<dyn std::error::Error + Send + Sync>) -> Error {
    Error::Config(error.to_string())
}
fn read<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}
fn stage(name: &str) {
    println!("{}", json!({"stage":name}));
}
fn execute(command: &mut Command, log: &Path) -> Result<()> {
    let file = fs::File::create(log)?;
    let status = command.stdout(file.try_clone()?).stderr(file).status()?;
    if !status.success() {
        return Err(Error::Config(format!(
            "preparation failed; see {}",
            log.display()
        )));
    }
    Ok(())
}
struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        if self.0.exists() {
            let names = [
                "source-copy.log",
                "source-bytecode.log",
                "materialization.log",
                "native-closure.json",
                "acceptance/container.log",
                "acceptance/control/boot-ready.json",
                "acceptance/control/complete.json",
            ];
            let logs: BTreeMap<_, _> = names
                .iter()
                .filter_map(|name| {
                    fs::read_to_string(self.0.join(name))
                        .ok()
                        .map(|text| (*name, text))
                })
                .collect();
            if !logs.is_empty() {
                let receipt = self.0.with_extension("failure.json");
                let _ = write(&receipt, &logs);
                println!("{}", json!({"failure_receipt":receipt}));
            }
        }
        fn writable(path: &Path) {
            if fs::symlink_metadata(path).is_ok_and(|m| m.is_dir()) {
                let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
                if let Ok(entries) = fs::read_dir(path) {
                    for entry in entries.flatten() {
                        writable(&entry.path());
                    }
                }
            }
        }
        writable(&self.0);
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fresh(out: &Path, source: &Path) -> Result<Scratch> {
    if fs::symlink_metadata(out).is_ok() {
        return Err(Error::Config("fresh output required".into()));
    }
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let parent = parent.canonicalize()?;
    if parent.starts_with(source) {
        return Err(Error::Config(
            "output must be outside the source runtime".into(),
        ));
    }
    let mut random = [0; 16];
    getrandom::fill(&mut random).map_err(|e| Error::Config(e.to_string()))?;
    let name: String = random.iter().map(|b| format!("{b:02x}")).collect();
    let path = parent.join(format!(".runtime-partition-{name}"));
    fs::create_dir(&path)?;
    Ok(Scratch(path))
}
fn publish(staging: &Path, out: &Path) -> Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        staging,
        rustix::fs::CWD,
        out,
        rustix::fs::RenameFlags::NOREPLACE,
    )
    .map_err(|e| Error::Io(e.into()))?;
    Ok(())
}
fn copy(source: &Path, out: &Path, log: &Path) -> Result<()> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    execute(
        Command::new("cp")
            .args(["--archive", "--reflink=auto"])
            .arg(source)
            .arg(out),
        log,
    )
}
fn move_preserving_parent(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(
        destination
            .parent()
            .ok_or_else(|| Error::Config("missing destination parent".into()))?,
    )?;
    let parent = source
        .parent()
        .ok_or_else(|| Error::Config("missing source parent".into()))?;
    let permissions = fs::metadata(parent)?.permissions();
    fs::set_permissions(
        parent,
        fs::Permissions::from_mode(permissions.mode() | 0o700),
    )?;
    let result = fs::rename(source, destination);
    fs::set_permissions(parent, permissions)?;
    Ok(result?)
}
fn pack(tree: &Path, out: &Path, metadata: Metadata, level: i32) -> Result<Manifest> {
    let manifest = switchyard_artifacts::pack(
        tree,
        out,
        metadata,
        PackOptions {
            zstd_level: level,
            ..PackOptions::default()
        },
    )?;
    write(&out.join("manifest.json"), &manifest)?;
    switchyard_artifacts::verify(tree, &manifest)?;
    Ok(manifest)
}
fn audit(trees: &BTreeMap<String, PathBuf>, plan: &Plan) -> Result<()> {
    for placed in &plan.entries {
        let path = trees[&placed.component].join(placed.entry.path());
        let metadata = fs::symlink_metadata(&path)?;
        let okay = match &placed.entry {
            Entry::File {
                size,
                sha256: expected,
                ..
            } => {
                metadata.is_file()
                    && metadata.len() == *size
                    && sha256(&path).map_err(preparation)? == *expected
            }
            Entry::Directory { .. } => metadata.is_dir(),
            Entry::Symlink { target, .. } => {
                metadata.is_symlink() && fs::read_link(&path)? == Path::new(target)
            }
        };
        if !okay || metadata.permissions().mode() & 0o777 != placed.entry.mode() {
            return Err(Error::Config(format!(
                "partition changed source entry: {}",
                placed.source_path
            )));
        }
    }
    Ok(())
}
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}
fn runtime_env(boot: &Path, recipe: &Recipe, environment: &BTreeMap<String, String>) -> Result<()> {
    let original = boot.join("partition-source/runtime-env.sh");
    let mut text = fs::read_to_string(&original)?;
    text.push_str("\n# Atomically published sibling runtime components.\n");
    text.push_str(&format!(
        "export SWITCHYARD_COMPONENTS_ROOT=\"${{SWITCHYARD_COMPONENTS_ROOT:-{}}}\"\n",
        recipe.components_root
    ));
    for (key, value) in environment {
        let quoted = value
            .split("{components_root}")
            .map(|v| shell_quote(&v.replace("{runtime_root}", &recipe.runtime_root)))
            .collect::<Vec<_>>()
            .join("\"${SWITCHYARD_COMPONENTS_ROOT}\"");
        text.push_str(&format!("export {key}={quoted}\n"));
    }
    fs::write(boot.join("runtime-env.sh"), text)?;
    fs::set_permissions(
        boot.join("runtime-env.sh"),
        fs::metadata(original)?.permissions(),
    )?;
    Ok(())
}
fn wrappers(trees: &BTreeMap<String, PathBuf>, recipe: &Recipe) -> Result<()> {
    for wrapper in &recipe.toolchain_wrappers {
        let tree = &trees[&wrapper.component];
        for name in ["gcc", "g++"] {
            let path = tree.join(&wrapper.path).join("bin").join(name);
            move_preserving_parent(
                &path,
                &tree
                    .join("partition-source")
                    .join(&wrapper.path)
                    .join("bin")
                    .join(name),
            )?;
            let major = wrapper.gcc_major;
            let text = format!(
                "#!/bin/sh\n# Relocated compiler; original bytes are retained in partition-source.\nset -eu\nroot=$(CDPATH= cd -- \"$(dirname -- \"$0\")/..\" && pwd)\nname=$(basename \"$0\")\nexport GCC_EXEC_PREFIX=\"$root/lib/gcc/x86_64-linux-gnu/{major}/\"\nexport CPLUS_INCLUDE_PATH=\"$root/sysroot/usr/include/c++/{major}:$root/sysroot/usr/include/x86_64-linux-gnu/c++/{major}${{CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}}\"\nexec \"$root/bin/$name.real\" --sysroot=\"$root/sysroot\" -B\"$root/lib/gcc/x86_64-linux-gnu/{major}/\" -B\"$root/bin/\" \"$@\"\n"
            );
            fs::write(&path, text)?;
            fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
        }
    }
    Ok(())
}

/// Source verification and patching happen only in a private copied full runtime.
fn derive(
    source: &Path,
    manifest: &Manifest,
    patch_lock: &Path,
    root: &Path,
    recipe: &Recipe,
    image: &str,
    level: i32,
) -> Result<(PathBuf, Manifest)> {
    stage("derive-patched-source");
    let tree = root.join("source/tree");
    copy(source, &tree, &root.join("source-copy.log"))?;
    let build = patch_lock
        .parent()
        .ok_or_else(|| Error::Config("patch lock has no parent".into()))?;
    let input: Value = read(patch_lock)?;
    if input["schema_version"] != 1
        || input["runtime_root"] != recipe.runtime_root
        || input.as_object().is_none_or(|o| {
            o.keys().any(|k| {
                !matches!(
                    k.as_str(),
                    "schema_version" | "runtime_root" | "source_patches"
                )
            })
        })
    {
        return Err(Error::Config("invalid source patch lock".into()));
    }
    let mut lock: Value = read(&tree.join("sources.lock.json"))?;
    lock["source_patches"] = input["source_patches"].clone();
    let mut info: Value = read(&tree.join("build-info.json"))?;
    let receipts =
        switchyard_runtime_build::apply_source_patches(build, &tree, &lock, &info["packages"])
            .map_err(preparation)?;
    if receipts.is_empty() {
        return Err(Error::Config(
            "derivation requested without pinned source patches".into(),
        ));
    }
    if tree.join("derivation-source").exists()
        || tree.join("source-patches.json").exists()
        || tree.join("source-patches-input.json").exists()
    {
        return Err(Error::Config(
            "source reserves derivation provenance paths".into(),
        ));
    }
    fs::create_dir(tree.join("derivation-source"))?;
    for name in [
        "artifact-metadata.json",
        "build-info.json",
        "sources.lock.json",
    ] {
        move_preserving_parent(&tree.join(name), &tree.join("derivation-source").join(name))?;
    }
    write(&tree.join("sources.lock.json"), &lock)?;
    write(&tree.join("source-patches-input.json"), &input)?;
    info["source_patches"] = serde_json::to_value(&receipts)?;
    info["derivation_parent_runtime_id"] = json!(manifest.id);
    info["sources"] = lock;
    info["sources_sha256"] = json!(sha256(&tree.join("sources.lock.json")).map_err(preparation)?);
    write(&tree.join("build-info.json"), &info)?;
    let mut metadata = manifest.metadata.clone();
    let hash = digest(&receipts).map_err(preparation)?;
    metadata
        .compatibility
        .insert("runtime_source_patches".into(), hash.clone());
    metadata
        .attributes
        .insert("derivation_parent_runtime_id".into(), manifest.id.clone());
    metadata
        .attributes
        .insert("source_patches_sha256".into(), hash);
    metadata.attributes.insert(
        "sources_sha256".into(),
        sha256(&tree.join("sources.lock.json")).map_err(preparation)?,
    );
    metadata.attributes.insert(
        "derivation_tool_version".into(),
        env!("CARGO_PKG_VERSION").into(),
    );
    write(&tree.join("artifact-metadata.json"), &metadata)?;
    write(&tree.join("source-patches.json"), &receipts)?;
    let mut command = Command::new("docker");
    command.args(["run","--rm","--network","none","--read-only","--cpus","2","--memory","4g","--tmpfs","/tmp:rw,size=256m","--env","CUDA_VISIBLE_DEVICES=","--env","NVIDIA_VISIBLE_DEVICES=void"])
        .arg("--mount").arg(format!("type=bind,source={},target={}",tree.display(),recipe.runtime_root))
        .arg(image).args(["/bin/bash","-c","source \"$1/runtime-env.sh\" && shift && exec python -m compileall -q -f --invalidation-mode unchecked-hash \"$@\"","bytecode"])
        .arg(&recipe.runtime_root);
    for receipt in &receipts {
        command.arg(format!("{}/{}", recipe.runtime_root, receipt.target));
    }
    execute(&mut command, &root.join("source-bytecode.log"))?;
    stage("pack-patched-full-source");
    let derived = pack(&tree, &root.join("source/artifact"), metadata, level)?;
    Ok((tree, derived))
}

pub(crate) struct Options {
    pub source: PathBuf,
    pub manifest: PathBuf,
    pub recipe: PathBuf,
    pub out: PathBuf,
    pub patch_lock: Option<PathBuf>,
    pub prefix: String,
    pub zstd_level: i32,
    pub acceptance_image: String,
    pub plan_only: bool,
}

pub(crate) fn build(options: Options) -> Result<Value> {
    let source = options.source.canonicalize()?;
    let original: Manifest = read(&options.manifest)?;
    original.validate()?;
    let recipe: Recipe = read(&options.recipe)?;
    let _ = recipe.plan(&original).map_err(preparation)?;
    let scratch = fresh(&options.out, &source)?;
    let root = &scratch.0;
    stage("verify-source");
    switchyard_artifacts::verify(&source, &original)?;
    let (source, manifest) = if let Some(lock) = &options.patch_lock {
        derive(
            &source,
            &original,
            &lock.canonicalize()?,
            root,
            &recipe,
            &options.acceptance_image,
            options.zstd_level,
        )?
    } else {
        (source, original.clone())
    };
    let plan = recipe.plan(&manifest).map_err(preparation)?;
    stage("native-dependency-closure");
    let closure = native_closure::compare(&source, &plan.entries, &manifest.entries, &recipe)
        .map_err(preparation)?;
    write(&root.join("native-closure.json"), &closure)?;
    if !closure.newly_unresolved.is_empty() {
        let report = options.out.with_extension("failure.json");
        write(&report, &closure)?;
        return Err(Error::Config(format!(
            "boot lost {} native dependencies; see {}",
            closure.newly_unresolved.len(),
            report.display()
        )));
    }
    write(&root.join("source-inventory.json"), &plan)?;
    write(&root.join("recipe.json"), &recipe)?;
    if options.plan_only {
        let result = json!({"source_runtime_id":manifest.id,"recipe_sha256":plan.recipe_sha256,"statistics":plan.statistics,"native_closure":closure});
        write(&root.join("plan.json"), &result)?;
        publish(root, &options.out)?;
        return Ok(result);
    }
    stage("materialize-components");
    let boot = root.join("boot/tree");
    copy(&source, &boot, &root.join("materialization.log"))?;
    let mut trees = BTreeMap::from([("boot".to_string(), boot.clone())]);
    for component in &recipe.components {
        let tree = root.join("components").join(&component.name).join("tree");
        fs::create_dir_all(&tree)?;
        for rule in &component.moves {
            move_preserving_parent(&boot.join(&rule.source), &tree.join(&rule.destination))?;
        }
        trees.insert(component.name.clone(), tree);
    }
    for name in ["artifact-metadata.json", "runtime-env.sh"] {
        move_preserving_parent(&boot.join(name), &boot.join("partition-source").join(name))?;
    }
    wrappers(&trees, &recipe)?;
    for placed in &plan.entries {
        if let Entry::Symlink { path, target, .. } = &placed.entry {
            let path = trees[&placed.component].join(path);
            if fs::read_link(&path)? != Path::new(target) {
                fs::remove_file(&path)?;
                std::os::unix::fs::symlink(target, path)?;
            }
        }
    }
    stage("audit-preserved-source-payload");
    audit(&trees, &plan)?;
    let mut compatibility = manifest.metadata.compatibility.clone();
    compatibility.insert("runtime_partition".into(), plan.recipe_sha256.clone());
    let common = BTreeMap::from([
        ("source_runtime_id".into(), manifest.id.clone()),
        ("parent_runtime_id".into(), original.id.clone()),
        ("partition_recipe_sha256".into(), plan.recipe_sha256.clone()),
        (
            "source_inventory_sha256".into(),
            digest(&manifest.entries).map_err(preparation)?,
        ),
        (
            "partition_tool_version".into(),
            env!("CARGO_PKG_VERSION").into(),
        ),
        (
            "partition_tool_sha256".into(),
            sha256(Path::new("/proc/self/exe")).map_err(preparation)?,
        ),
    ]);
    let mut components = vec![];
    for component in &recipe.components {
        stage(&format!("pack-{}", component.name));
        let mut attributes = common.clone();
        attributes.insert("component".into(), component.name.clone());
        let entries: Vec<_> = plan
            .entries
            .iter()
            .filter(|e| e.component == component.name)
            .collect();
        attributes.insert(
            "payload_inventory_sha256".into(),
            digest(&entries).map_err(preparation)?,
        );
        let metadata = Metadata {
            kind: "runtime_component".into(),
            compatibility: compatibility.clone(),
            attributes,
        };
        write(
            &trees[&component.name].join("artifact-metadata.json"),
            &metadata,
        )?;
        let packed = pack(
            &trees[&component.name],
            &root
                .join("components")
                .join(&component.name)
                .join("artifact"),
            metadata,
            options.zstd_level,
        )?;
        components.push(json!({"name":component.name,"artifact":{"id":packed.id,"prefix":format!("{}/{}",options.prefix.trim_end_matches('/'),packed.id)},"compatibility":compatibility,
                              "destination":format!("{}/{}",recipe.components_root,component.name),"manifest":format!("components/{}/artifact/manifest.json",component.name),"tree":format!("components/{}/tree",component.name)}));
    }
    let mut environment = recipe.environment.clone();
    environment.insert(
        "LD_LIBRARY_PATH".into(),
        closure
            .library_paths
            .iter()
            .map(|p| format!("{{runtime_root}}/{p}"))
            .collect::<Vec<_>>()
            .join(":"),
    );
    runtime_env(&boot, &recipe, &environment)?;
    let mut metadata = manifest.metadata.clone();
    metadata.compatibility = compatibility.clone();
    metadata.attributes.extend(common.clone());
    metadata.attributes.insert(
        "runtime_components_sha256".into(),
        digest(&components).map_err(preparation)?,
    );
    write(&boot.join("artifact-metadata.json"), &metadata)?;
    write(
        &boot.join("runtime-components.json"),
        &json!({"schema_version":1,"provenance":common,"components":components,"environment":environment,"source_payload_preserved":true,"toolchain_wrappers":recipe.toolchain_wrappers}),
    )?;
    let accepted = crate::runtime_smoke::check(
        &boot,
        &trees,
        &recipe,
        &plan,
        &options.acceptance_image,
        &root.join("acceptance"),
    )?;
    let loaded: std::collections::BTreeSet<_> = accepted["boot"]["loaded_runtime_libraries"]
        .as_array()
        .ok_or_else(|| Error::Config("missing native import receipt".into()))?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if closure
        .unresolved
        .iter()
        .any(|e| loaded.contains(e.elf.as_str()))
    {
        return Err(Error::Config(
            "loaded boot ELF has unresolved static dependencies".into(),
        ));
    }
    stage("pack-boot");
    let packed = pack(
        &boot,
        &root.join("boot/artifact"),
        metadata,
        options.zstd_level,
    )?;
    let set_id = digest(&json!({"boot":packed.id,"components":components,"environment":environment,"recipe":plan.recipe_sha256,"runtime_root":recipe.runtime_root,"components_root":recipe.components_root})).map_err(preparation)?;
    let descriptor = json!({"format":"switchyard-runtime-components-v1","schema_version":1,"runtime_set_id":set_id,"runtime_root":recipe.runtime_root,"components_root":recipe.components_root,
        "provenance":common,"boot":{"artifact":{"id":packed.id,"prefix":format!("{}/{}",options.prefix.trim_end_matches('/'),packed.id)},"compatibility":compatibility,"manifest":"boot/artifact/manifest.json","tree":"boot/tree"},
        "components":components,"environment":environment,"statistics":plan.statistics,"fresh_cache_bake_required":true});
    write(&root.join("runtime-set.json"), &descriptor)?;
    stage("publish-local-composition");
    publish(root, &options.out)?;
    Ok(descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "switchyard-composition-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn local_publication_never_replaces_an_existing_directory() {
        let fixture = Fixture::new();
        let staging = fixture.0.join("staging");
        let out = fixture.0.join("out");
        fs::create_dir(&staging).unwrap();
        fs::write(staging.join("complete"), "all artifacts accepted").unwrap();
        fs::create_dir(&out).unwrap();
        assert!(publish(&staging, &out).is_err());
        assert!(staging.join("complete").is_file());
        assert!(fs::read_dir(&out).unwrap().next().is_none());
        fs::remove_dir(&out).unwrap();
        publish(&staging, &out).unwrap();
        assert!(!staging.exists());
        assert_eq!(
            fs::read_to_string(out.join("complete")).unwrap(),
            "all artifacts accepted"
        );
    }
    #[test]
    fn private_copy_preserves_source_under_changes_and_wrapper_relocation() {
        let fixture = Fixture::new();
        let source = fixture.0.join("source");
        let tree = fixture.0.join("tree");
        fs::create_dir_all(source.join("toolchain/bin")).unwrap();
        for name in ["gcc", "g++"] {
            fs::write(
                source.join("toolchain/bin").join(name),
                "original pinned wrapper",
            )
            .unwrap();
        }
        copy(&source, &tree, &fixture.0.join("copy.log")).unwrap();
        let recipe: Recipe = serde_json::from_value(json!({"format":"switchyard-runtime-partition-v1","schema_version":1,"runtime_root":"/opt/switchyard/runtime","components_root":"/opt/switchyard/components","requires":{},"components":[],"environment":{},"library_paths":[],"external_libraries":[],"boot_imports":[],"resource_checks":[],"toolchain_wrappers":[{"component":"build","path":"toolchain","gcc_major":13}]})).unwrap();
        wrappers(&BTreeMap::from([("build".into(), tree.clone())]), &recipe).unwrap();
        assert_eq!(
            fs::read(source.join("toolchain/bin/gcc")).unwrap(),
            fs::read(tree.join("partition-source/toolchain/bin/gcc")).unwrap()
        );
        assert_eq!(
            fs::read_to_string(source.join("toolchain/bin/gcc")).unwrap(),
            "original pinned wrapper"
        );
        assert!(
            fs::read_to_string(tree.join("toolchain/bin/gcc"))
                .unwrap()
                .contains("dirname")
        );
        assert_ne!(
            fs::metadata(source.join("toolchain/bin/gcc"))
                .unwrap()
                .ino(),
            fs::metadata(tree.join("partition-source/toolchain/bin/gcc"))
                .unwrap()
                .ino()
        );
    }
    #[test]
    fn shell_environment_keeps_metacharacters_literal() {
        assert_eq!(shell_quote("a'$(secret)"), "'a'\"'\"'$(secret)'");
    }
    use std::os::unix::fs::MetadataExt;
}
