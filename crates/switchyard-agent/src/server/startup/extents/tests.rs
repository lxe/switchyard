use super::*;
use axum::{
    Router,
    body::Body,
    extract::{Path as HttpPath, State as HttpState},
    routing::get,
};
use std::{collections::BTreeMap, os::unix::fs::PermissionsExt};
use switchyard_artifacts::{ArtifactTicket, Metadata, PackOptions, pack};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
#[derive(Clone)]
struct Objects {
    bytes: Arc<BTreeMap<String, Vec<u8>>>,
    held: String,
    gate: watch::Receiver<bool>,
}
async fn object(
    HttpState(mut source): HttpState<Objects>,
    HttpPath(name): HttpPath<String>,
) -> Body {
    if name == source.held {
        while !*source.gate.borrow() {
            source.gate.changed().await.unwrap();
        }
    }
    Body::from(source.bytes[&name].clone())
}
struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
async fn until(mut f: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !f() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn metadata_releases_engine_before_payload_and_bridge_owns_completed_ranges() {
    let mut nonce = [0; 8];
    getrandom::fill(&mut nonce).unwrap();
    let scratch =
        Scratch(std::env::temp_dir().join(format!("extent-agent-{:x}", u64::from_le_bytes(nonce))));
    let runtime = scratch.0.join("runtime-source");
    let source = scratch.0.join("model-source");
    fs::create_dir_all(&runtime).unwrap();
    fs::create_dir(&source).unwrap();
    fs::write(runtime.join("engine"),"#!/bin/sh\ntouch \"$IMPORTED\"\nread gate\ntest \"$gate\" = switchyard-metadata-ready-v1 || exit 1\ntest -f \"$3/config.json\" || exit 2\ntouch \"$METADATA\"\nsleep 60\n").unwrap();
    fs::set_permissions(runtime.join("engine"), fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(source.join("config.json"), b"{}").unwrap();
    let header = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut tensor = (header.len() as u64).to_le_bytes().to_vec();
    tensor.extend(header);
    tensor.extend(42.0_f32.to_le_bytes());
    fs::write(source.join("model.safetensors"), tensor).unwrap();
    let metadata = Metadata {
        kind: "test".into(),
        compatibility: BTreeMap::new(),
        attributes: BTreeMap::new(),
    };
    let runtime_manifest = pack(
        &runtime,
        &scratch.0.join("runtime-objects"),
        metadata.clone(),
        PackOptions::default(),
    )
    .unwrap();
    let manifest =
        switchyard_weights::pack_weights(&source, &scratch.0.join("model-objects"), metadata, 4096)
            .unwrap();
    let held = manifest
        .extents
        .iter()
        .find(|e| e.offset != 0)
        .unwrap()
        .object
        .clone();
    let mut bytes = BTreeMap::new();
    for c in &runtime_manifest.chunks {
        bytes.insert(
            c.name.clone(),
            fs::read(scratch.0.join("runtime-objects").join(&c.name)).unwrap(),
        );
    }
    for e in &manifest.extents {
        bytes.insert(
            e.object.clone(),
            fs::read(scratch.0.join("model-objects").join(&e.object)).unwrap(),
        );
    }
    let (gate, waiting) = watch::channel(false);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/{name}", get(object))
        .with_state(Objects {
            bytes: Arc::new(bytes),
            held,
            gate: waiting,
        });
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let roots = RootPaths::under(&scratch.0.join("hydrated"));
    fs::create_dir_all(&roots.models).unwrap();
    let imported = scratch.0.join("imported");
    let metadata_ready = scratch.0.join("metadata-ready");
    let ticket = Ticket {
        model: "tiny".into(),
        revision: "a".repeat(40),
        source: crate::model::ModelSource::Extents(crate::model::ExtentSource {
            model_source: crate::model::ExtentModel::Extents {
                artifact: switchyard_artifacts::extents::ExtentTicket {
                    urls: manifest
                        .extents
                        .iter()
                        .map(|e| (e.object.clone(), format!("{base}/{}", e.object)))
                        .collect(),
                    manifest,
                },
                loader: switchyard_weights::VllmLoaderConfig {
                    python_program: "engine".into(),
                    cuda_stage: false,
                    pinned_bytes: 4096,
                },
            },
        }),
        engine_args: vec!["serve".into(), "model".into()],
        concurrency: 20,
        stall_s: 10,
        lifetime_s: 30,
        startup: crate::startup::StartupPolicy {
            mode: crate::startup::HydrationMode::TrustedStreaming,
            ..Default::default()
        },
        runtime: Some(crate::runtime::RuntimeTicket {
            artifact: ArtifactTicket {
                urls: runtime_manifest
                    .chunks
                    .iter()
                    .map(|c| (c.name.clone(), format!("{base}/{}", c.name)))
                    .collect(),
                manifest: runtime_manifest,
            },
            compatibility: BTreeMap::new(),
            program: "engine".into(),
            args: vec![],
            environment: BTreeMap::from([
                ("IMPORTED".into(), imported.to_string_lossy().into_owned()),
                (
                    "METADATA".into(),
                    metadata_ready.to_string_lossy().into_owned(),
                ),
            ]),
            caches: vec![],
            cache_export: None,
            hydration: Some(crate::runtime::RuntimeHydrationOptions { retries: 0 }),
        }),
    };
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let observed = state.clone();
    let model_root = roots.models.clone();
    let pending =
        tokio::spawn(async move { prepare(&ticket, &roots, state, Instant::now()).await });
    until(|| metadata_ready.exists()).await;
    assert!(imported.exists());
    assert!(!model_root.join("model").exists());
    assert!(!pending.is_finished());
    let owner = fs::read_dir(&model_root)
        .unwrap()
        .find_map(|e| {
            let e = e.unwrap();
            e.file_name()
                .to_string_lossy()
                .starts_with(".model-loader-")
                .then(|| e.path())
        })
        .unwrap();
    let socket = tokio::net::UnixStream::connect(owner.join("weights.sock"))
        .await
        .unwrap();
    let (read, mut write) = socket.into_split();
    let mut read = BufReader::new(read);
    write.write_all(b"next\n").await.unwrap();
    gate.send(true).unwrap();
    let mut descriptor = String::new();
    read.read_line(&mut descriptor).await.unwrap();
    let value: serde_json::Value = serde_json::from_str(&descriptor).unwrap();
    assert_eq!(value["tensor"]["length"], 4);
    write.write_all(b"next\n").await.unwrap();
    descriptor.clear();
    read.read_line(&mut descriptor).await.unwrap();
    assert!(descriptor.contains("done"));
    write.write_all(b"finish\n").await.unwrap();
    let mut process = pending.await.unwrap().unwrap();
    assert!(owner.exists());
    let s = observed.read().await;
    assert_eq!(s.weight_loading.as_ref().unwrap().tensors_delivered, 1);
    assert!(s.model_transport.is_some());
    assert!(s.dependency_budget.is_some());
    drop(s);
    process.stop(Duration::from_secs(2)).await.unwrap();
    drop(process);
    until(|| !owner.exists()).await;
    assert!(model_root.join("model/config.json").exists());
    server.abort();
}
