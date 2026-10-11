//! CPU-only Python imports before publication, then resource and compiler checks.
use crate::{Error, Result};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};
use switchyard_artifacts::Entry;
use switchyard_runtime_build::partition::{Plan, Recipe};

struct Container {
    name: String,
    child: Child,
    moved: Vec<(PathBuf, PathBuf)>,
}
impl Drop for Container {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = Command::new("docker")
                .args(["rm", "--force", &self.name])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let _ = self.child.wait();
        }
        for (source, destination) in self.moved.iter().rev() {
            let _ = fs::rename(destination, source);
        }
    }
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
pub(crate) fn check(
    boot: &Path,
    trees: &BTreeMap<String, PathBuf>,
    recipe: &Recipe,
    plan: &Plan,
    image: &str,
    out: &Path,
) -> Result<Value> {
    let control = out.join("control");
    let visible = out.join("published-components");
    fs::create_dir_all(&control)?;
    fs::create_dir(&visible)?;
    let mut resources = vec![];
    for rule in &recipe.resource_checks {
        let entry = plan
            .entries
            .iter()
            .filter(|e| {
                e.component == rule.component
                    && e.entry.path().starts_with(&(rule.path.clone() + "/"))
                    && e.entry.path().ends_with(".cubin")
            })
            .filter_map(|e| match &e.entry {
                Entry::File {
                    path, size, sha256, ..
                } if *size > 0 => Some((size, path, sha256)),
                _ => None,
            })
            .min()
            .ok_or_else(|| Error::Config("resource acceptance has no cubin payload".into()))?;
        let (size, path, hash) = entry;
        resources.push(json!({"module":rule.module,"attribute":rule.attribute,"loader":rule.loader,
            "directory":format!("{}/{}/{}",recipe.components_root,rule.component,rule.path),
            "file":path.strip_prefix(&(rule.path.clone() + "/")).unwrap(),"size":size,"sha256":hash}));
    }
    let mut compilers = recipe.compiler_checks.clone();
    for action in &mut compilers {
        for argument in &mut action.command {
            *argument = recipe.expand(argument);
        }
    }
    let config = json!({"runtime_root":recipe.runtime_root,"imports":recipe.boot_imports,"resources":resources,"compilers":compilers});
    fs::write(
        control.join("probe.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    fs::write(
        control.join("check.py"),
        include_str!("../../../scripts/runtime/component-smoke.py"),
    )?;
    let output = Command::new("docker")
        .args(["image", "inspect", "--format", "{{.Id}}", image])
        .output()?;
    if !output.status.success() {
        return Err(Error::Config(
            "acceptance image must already exist locally".into(),
        ));
    }
    let image_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let name = format!(
        "switchyard-partition-{}",
        out.parent()
            .and_then(Path::file_name)
            .unwrap_or_default()
            .to_string_lossy()
            .trim_start_matches('.')
    );
    let log = fs::File::create(out.join("container.log"))?;
    let child = Command::new("docker")
        .args([
            "run",
            "--rm",
            "--name",
            &name,
            "--network",
            "none",
            "--read-only",
            "--cpus",
            "2",
            "--memory",
            "8g",
            "--tmpfs",
            "/tmp:rw,exec,size=1g",
        ])
        .arg("--mount")
        .arg(format!(
            "type=bind,source={},target={},readonly",
            boot.display(),
            recipe.runtime_root
        ))
        .arg("--mount")
        .arg(format!(
            "type=bind,source={},target={},readonly",
            visible.display(),
            recipe.components_root
        ))
        .arg("--mount")
        .arg(format!(
            "type=bind,source={},target=/control",
            control.display()
        ))
        .args([
            "--env",
            "SWITCHYARD_CACHE_ROOT=/tmp/cache",
            "--env",
            "CUDA_VISIBLE_DEVICES=",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=void",
            image,
            "/bin/bash",
            "-c",
            "source \"$1/runtime-env.sh\" && exec python /control/check.py",
            "acceptance",
        ])
        .arg(&recipe.runtime_root)
        .stdout(log.try_clone()?)
        .stderr(log)
        .spawn()?;
    let mut container = Container {
        name,
        child,
        moved: vec![],
    };
    println!(
        "{}",
        json!({"stage":"boot-import-acceptance","image_id":image_id})
    );
    let deadline = Instant::now() + Duration::from_secs(120);
    while !control.join("boot-ready.json").exists() {
        if container.child.try_wait()?.is_some() || Instant::now() >= deadline {
            return Err(Error::Config(format!(
                "boot import failed or timed out; see {}",
                out.join("container.log").display()
            )));
        }
        thread::sleep(Duration::from_millis(100));
    }
    for component in &recipe.components {
        let source = &trees[&component.name];
        let destination = visible.join(&component.name);
        fs::rename(source, &destination)?;
        container.moved.push((source.clone(), destination));
    }
    fs::write(control.join("components.ready"), b"ready\n")?;
    println!("{}", json!({"stage":"component-resource-acceptance"}));
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if let Some(status) = container.child.try_wait()? {
            if !status.success() {
                return Err(Error::Config(format!(
                    "component acceptance failed; see {}",
                    out.join("container.log").display()
                )));
            }
            break;
        }
        if Instant::now() >= deadline {
            return Err(Error::Config("component acceptance timed out".into()));
        }
        thread::sleep(Duration::from_millis(100));
    }
    let receipt = json!({"image":image,"image_id":image_id,"network":"none","gpu_devices":false,"boot":read(&control.join("boot-ready.json"))?,"complete":read(&control.join("complete.json"))?});
    if receipt["complete"]["cuda_initialized"] != false {
        return Err(Error::Config("CPU acceptance initialized CUDA".into()));
    }
    fs::write(
        out.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    // Drop stops only our owned process and restores complete component trees.
    drop(container);
    if trees.values().any(|p| !p.is_dir()) {
        return Err(Error::Config(
            "acceptance did not restore component trees".into(),
        ));
    }
    Ok(receipt)
}
