//! GPU preparation is an owned subprocess; environment changes never mutate
//! a multithreaded serving process. Driver caches remain separate from weights.
use clap::Parser;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
use switchyard_engine::config::{Config, DeviceConfig};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    engine: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    requests: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn cache_size(root: &Path) -> Result<(usize, u64)> {
    let mut files = 0;
    let mut bytes = 0;
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            let (n, b) = cache_size(&entry.path())?;
            files += n;
            bytes += b;
        } else if metadata.is_file() {
            files += 1;
            bytes += metadata.len();
        }
    }
    Ok((files, bytes))
}
fn main() -> Result<()> {
    let args = Args::parse();
    let config: Config = serde_json::from_slice(&std::fs::read(&args.config)?)?;
    config.validate()?;
    let ordinal = match config.device {
        DeviceConfig::Cuda { ordinal } => ordinal,
        DeviceConfig::Cpu => return Err("GPU bake requires a CUDA configuration".into()),
    };
    std::fs::create_dir(&args.output)?;
    let output = args.output.canonicalize()?;
    let cache = output.join("cuda");
    std::fs::create_dir(&cache)?;
    let hardware = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
            "-i",
            &ordinal.to_string(),
        ])
        .output()?;
    if !hardware.status.success() {
        return Err("Cannot identify allocated GPU".into());
    }
    let began = Instant::now();
    let status = Command::new(&args.engine)
        .arg("--config")
        .arg(&args.config)
        .arg("generate")
        .arg("--input")
        .arg(&args.requests)
        .arg("--output")
        .arg(output.join("generated.json"))
        .env("CUDA_CACHE_PATH", &cache)
        .env("CUDA_CACHE_DISABLE", "0")
        .env("CUDA_CACHE_MAXSIZE", "536870912")
        .status()?;
    if !status.success() {
        return Err("GPU preparation failed; preserve its telemetry for diagnosis".into());
    }
    let (files, bytes) = cache_size(&cache)?;
    if bytes == 0 {
        return Err("Driver cache is empty; bake is not reusable".into());
    }
    let receipt = serde_json::json!({"format":"switchyard-cuda-bake-v1",
        "engine_sha256":format!("{:x}",Sha256::digest(std::fs::read(&args.engine)?)),
        "hardware":String::from_utf8(hardware.stdout)?.trim(),"config":config,
        "requests":serde_json::from_slice::<serde_json::Value>(&std::fs::read(&args.requests)?)?,
        "elapsed_s":began.elapsed().as_secs_f64(),"cache_files":files,"cache_bytes":bytes,
        "cache_path":"cuda","cuda_compute_cap_build":option_env!("CUDA_COMPUTE_CAP"),
        "candle_revision":"c68b24997319b8d76e5a4e775dab197c7f26322a",
        "gpu_process_snapshot":false,"purpose":"offline kernel compilation; no serving readiness inference"});
    std::fs::write(
        output.join("bake.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
