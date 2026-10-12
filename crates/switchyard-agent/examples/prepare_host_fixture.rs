//! Prepare a loopback-only multi-model hydration fixture using the production packers.
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};
use switchyard_agent::{
    Ticket,
    host::{HostTicket, HostedEndpoint},
    model::{ExtentModel, ExtentSource, ModelSource},
    runtime::RuntimeTicket,
    startup::StartupPolicy,
};
use switchyard_artifacts::{
    ArtifactTicket, Metadata, PackOptions,
    extents::{ExtentTicket, pack_file_ranges},
    pack,
};
use switchyard_core::topology::{EndpointPlan, HostPlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 && args.len() != 7 {
        return Err(
            "usage: prepare_host_fixture OUTPUT ENGINE LOOPBACK_BASE CONFIG_A CONFIG_B [CUDA_SEED_A CUDA_SEED_B]".into(),
        );
    }
    let root = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&root)?;
    let root = std::fs::canonicalize(root)?;
    let base = &args[2];
    if !base.starts_with("http://127.0.0.1:") {
        return Err("fixture artifact server must bind loopback".into());
    }
    let binary = std::fs::canonicalize(&args[1])?;
    let engine_sha = format!("{:x}", Sha256::digest(std::fs::read(&binary)?));
    let source = root.join("runtime-source");
    std::fs::create_dir_all(&source)?;
    std::fs::copy(&binary, source.join("engine"))?;
    let artifacts = root.join("http");
    std::fs::create_dir_all(&artifacts)?;
    let metadata = Metadata {
        kind: "runtime".into(),
        compatibility: BTreeMap::from([
            ("architecture".into(), std::env::consts::ARCH.into()),
            ("engine_sha256".into(), engine_sha),
        ]),
        attributes: BTreeMap::new(),
    };
    let runtime = pack(
        &source,
        &artifacts.join("runtime"),
        metadata,
        PackOptions::default(),
    )?;
    let runtime = RuntimeTicket {
        hydration: None,
        artifact: ArtifactTicket {
            urls: runtime
                .chunks
                .iter()
                .map(|c| (c.name.clone(), format!("{base}/runtime/{}", c.name)))
                .collect(),
            manifest: runtime,
        },
        compatibility: BTreeMap::new(),
        components: vec![],
        program: "engine".into(),
        args: vec![],
        environment: BTreeMap::from([("CUDA_CACHE_PATH".into(), "{cache_root}/cuda".into())]),
        caches: vec![],
        cache_export: None,
    };
    let mut endpoints = vec![];
    let mut plans = vec![];
    for (i, path) in args[3..5].iter().enumerate() {
        let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let id = config["model"].as_str().ok_or("missing model")?.to_owned();
        let source = std::fs::canonicalize(config["model_root"].as_str().ok_or("missing source")?)?;
        let metadata = Metadata {
            kind: "model".into(),
            compatibility: BTreeMap::new(),
            attributes: BTreeMap::new(),
        };
        let manifest = pack_file_ranges(&source, &artifacts.join(&id), metadata, 64 << 20)?;
        let startup = StartupPolicy {
            mode: switchyard_artifacts::HydrationMode::TrustedStreaming,
            model_root: Some(
                root.join("models")
                    .join(&manifest.content_id)
                    .to_string_lossy()
                    .into(),
            ),
            ..Default::default()
        };
        config["model_root"] = serde_json::json!(startup.model_directory());
        let port = 8100 + i as u16;
        plans.push(EndpointPlan {
            id: id.clone(),
            profile: format!("{id}-local"),
            gpu_memory_bytes: if i == 0 { 6 << 30 } else { 12 << 30 },
            compute_fraction: 0.5,
        });
        let mut endpoint_runtime = runtime.clone();
        if args.len() == 7 {
            let source = std::fs::canonicalize(&args[5 + i])?;
            let mut compatibility = endpoint_runtime
                .artifact
                .manifest
                .metadata
                .compatibility
                .clone();
            compatibility.insert(
                "runtime_id".into(),
                endpoint_runtime.artifact.manifest.id.clone(),
            );
            compatibility.insert("gpu_compute_capability".into(), "12.0".into());
            compatibility.insert("model".into(), id.clone());
            let cache_path = format!("cache-{i}");
            let cache = pack(
                &source,
                &artifacts.join(&cache_path),
                Metadata {
                    kind: "cache".into(),
                    compatibility: compatibility.clone(),
                    attributes: BTreeMap::new(),
                },
                PackOptions::default(),
            )?;
            endpoint_runtime
                .caches
                .push(switchyard_agent::runtime::CacheTicket {
                    name: "cuda".into(),
                    compatibility,
                    artifact: ArtifactTicket {
                        urls: cache
                            .chunks
                            .iter()
                            .map(|c| (c.name.clone(), format!("{base}/{cache_path}/{}", c.name)))
                            .collect(),
                        manifest: cache,
                    },
                });
        }
        endpoints.push(HostedEndpoint {
            id: id.clone(),
            port,
            ticket: Ticket {
                model: id.clone(),
                revision: "local-fixture".into(),
                engine_args: vec![
                    "--config-json".into(),
                    config.to_string(),
                    "serve".into(),
                    "--bind".into(),
                    format!("127.0.0.1:{port}"),
                ],
                concurrency: 20,
                stall_s: 60,
                lifetime_s: 180,
                runtime: Some(endpoint_runtime),
                startup,
                source: ModelSource::Extents(ExtentSource {
                    model_source: ExtentModel::Files {
                        artifact: ExtentTicket {
                            urls: manifest
                                .extents
                                .iter()
                                .map(|e| (e.object.clone(), format!("{base}/{id}/{}", e.object)))
                                .collect(),
                            manifest,
                        },
                    },
                }),
            },
        });
    }
    let ticket = HostTicket {
        schema_version: 1,
        storage_root: Some(root.join("installation").to_string_lossy().into()),
        plan: HostPlan {
            id: "local-warm-pair".into(),
            gpu: "NVIDIA GeForce RTX 5090".into(),
            gpu_memory_bytes: 24 << 30,
            reserved_gpu_memory_bytes: 4 << 30,
            contention_factor: 0.8,
            endpoints: plans,
        },
        endpoints,
    };
    ticket.validate()?;
    std::fs::write(
        artifacts.join("ticket.json"),
        serde_json::to_vec_pretty(&ticket)?,
    )?;
    println!("host fixture prepared at {}", artifacts.display());
    Ok(())
}
