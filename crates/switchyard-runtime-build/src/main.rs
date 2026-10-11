//! Build-time assembly of fixed-prefix serving hydration artifacts.
//!
//! This program runs only inside the pinned Docker builder. It is independent of
//! cloud providers, worker hydration, the simulator, and the scheduler.

mod bytecode;
mod command;
mod cuda;
mod download;
mod filesystem;
mod native;
mod patches;

use command::{capture, run};
use download::{download, sha256};
use filesystem::{copy_file, internalize_symlinks};
use native::{assemble_native_libraries, assemble_toolchain};
use serde_json::{Value, json};
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const BUILD: &str = "/build";
const ROOT: &str = "/opt/switchyard/runtime";

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    fs::write(path, text)?;
    Ok(())
}

fn package_version<'a>(packages: &'a Value, name: &str) -> Result<&'a str> {
    packages
        .as_array()
        .and_then(|packages| packages.iter().find(|package| package["name"] == name))
        .and_then(|package| package["version"].as_str())
        .ok_or_else(|| format!("installed package has no recorded version: {name}").into())
}

#[tokio::main]
async fn main() -> Result<()> {
    let build = Path::new(BUILD);
    let root = Path::new(ROOT);
    let lock: Value = serde_json::from_slice(&fs::read(build.join("sources.lock.json"))?)?;
    if lock["runtime_root"] != ROOT {
        return Err("source lock runtime prefix mismatch".into());
    }
    fs::create_dir_all(build.join("downloads"))?;
    fs::create_dir_all(root.join("python"))?;
    fs::create_dir_all(build.join("uv"))?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    let python_archive = download(&client, &lock["python"], "python.tar.gz").await?;
    let uv_archive = download(&client, &lock["uv"], "uv.tar.gz").await?;
    run(
        "tar",
        &[
            "-xzf",
            python_archive
                .to_str()
                .ok_or("invalid Python archive path")?,
            "--strip-components=1",
            "-C",
            "/opt/switchyard/runtime/python",
        ],
    )?;
    run(
        "tar",
        &[
            "-xzf",
            uv_archive.to_str().ok_or("invalid uv archive path")?,
            "--strip-components=1",
            "-C",
            "/build/uv",
        ],
    )?;
    let uv = build.join("uv/uv");
    run(
        &uv,
        &[
            "venv",
            "--python",
            "/opt/switchyard/runtime/python/bin/python3.12",
            "--no-python-downloads",
            "/opt/switchyard/runtime/venv",
        ],
    )?;
    run(
        &uv,
        &[
            "pip",
            "sync",
            "--python",
            "/opt/switchyard/runtime/venv/bin/python",
            "--require-hashes",
            "--no-build",
            "--link-mode",
            "copy",
            "--index",
            "https://flashinfer.ai/whl/",
            "--cache-dir",
            "/build/uv-cache",
            "/build/requirements.lock",
        ],
    )?;
    run(
        &uv,
        &[
            "pip",
            "check",
            "--python",
            "/opt/switchyard/runtime/venv/bin/python",
        ],
    )?;
    assemble_toolchain(root)?;
    cuda::assemble(root, &lock, &client).await?;
    let native = assemble_native_libraries(root)?;
    copy_file(
        Path::new("/etc/ssl/certs/ca-certificates.crt"),
        &root.join("certs/ca-certificates.crt"),
    )?;
    for filename in [
        "requirements.in",
        "requirements.lock",
        "sources.lock.json",
        "runtime-env.sh",
        "smoke.py",
    ] {
        copy_file(&build.join(filename), &root.join(filename))?;
    }
    let output = Command::new(&uv)
        .args([
            "pip",
            "list",
            "--python",
            "/opt/switchyard/runtime/venv/bin/python",
            "--format",
            "json",
        ])
        .output()?;
    if !output.status.success() {
        return Err("cannot record installed Python packages".into());
    }
    let packages: Value = serde_json::from_slice(&output.stdout)?;
    let source_patches = patches::apply(build, root, &lock, &packages)?;
    bytecode::compile(root)?;
    internalize_symlinks(root)?;
    let python_version = lock["python"]["version"]
        .as_str()
        .ok_or("missing Python version")?;
    let engine = lock["serving_engine"].as_str().unwrap_or("vllm");
    if !matches!(engine, "vllm" | "sglang") {
        return Err("unsupported serving engine in source lock".into());
    }
    let engine_version = package_version(&packages, engine)?;
    if lock[engine]["version"].as_str() != Some(engine_version) {
        return Err("installed serving engine differs from source lock".into());
    }
    let torch_version = package_version(&packages, "torch")?;
    let apt_output = capture("dpkg-query", &["-W", "-f=${Package}=${Version}\n"])?;
    let mut apt: Vec<_> = String::from_utf8(apt_output.stdout)?
        .lines()
        .map(str::to_owned)
        .collect();
    apt.sort();
    let requirements_hash = sha256(&build.join("requirements.lock"))?;
    let sources_hash = sha256(&build.join("sources.lock.json"))?;
    let builder_hash = sha256(&build.join("switchyard-runtime-build"))?;
    write_json(
        &root.join("build-info.json"),
        &json!({"schema_version": 1, "sources": lock,
        "sources_sha256": sources_hash, "builder_sha256": builder_hash,
        "requirements_sha256": requirements_hash, "packages": packages, "apt_packages": apt,
        "native_libraries": native, "source_patches": source_patches, "gpu_bake_performed": false,
        "python_bytecode_invalidation": bytecode::INVALIDATION_MODE}),
    )?;
    write_json(
        &root.join("artifact-metadata.json"),
        &json!({"kind": "runtime", "compatibility": {
        "os": "ubuntu24.04", "architecture": "x86_64", "python": python_version, (engine): engine_version,
        "cuda": lock["cuda"].as_str().unwrap_or("12.9"), "torch": torch_version}, "attributes": {"runtime_root": ROOT,
        "requirements_sha256": requirements_hash, "sources_sha256": sources_hash,
        "builder_sha256": builder_hash, "gpu_bake_performed": "false",
        "python_bytecode_invalidation": bytecode::INVALIDATION_MODE}}),
    )?;
    println!(
        "{}",
        json!({"runtime_root": ROOT, "packages": packages.as_array().map(Vec::len), "native_libraries": native.len()})
    );
    Ok(())
}
