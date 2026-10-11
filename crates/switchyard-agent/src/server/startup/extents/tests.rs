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
    held_runtime: String,
    runtime_gate: watch::Receiver<bool>,
    runtime_requested: Arc<tokio::sync::Notify>,
    held_dependency: String,
    dependency_gate: watch::Receiver<bool>,
}
async fn object(
    HttpState(mut source): HttpState<Objects>,
    HttpPath(name): HttpPath<String>,
) -> Body {
    if name == source.held_runtime {
        source.runtime_requested.notify_one();
        while !*source.runtime_gate.borrow() {
            source.runtime_gate.changed().await.unwrap();
        }
    }
    if name == source.held_dependency {
        while !*source.dependency_gate.borrow() {
            source.dependency_gate.changed().await.unwrap();
        }
    }
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
#[test]
fn loader_paths_are_stable_and_refuse_an_existing_owner() {
    let mut nonce = [0; 8];
    getrandom::fill(&mut nonce).unwrap();
    let scratch =
        Scratch(std::env::temp_dir().join(format!("extent-path-{:x}", u64::from_le_bytes(nonce))));
    fs::create_dir(&scratch.0).unwrap();
    let content = "a".repeat(64);
    let loader = switchyard_weights::VllmLoaderConfig::default();
    let first = sidecar(&scratch.0, &content, &loader).unwrap();
    let path = first.root.clone();
    fs::write(path.join("owner-data"), b"preserve active owner").unwrap();
    assert!(sidecar(&scratch.0, &content, &loader).is_err());
    assert!(path.join("owner-data").is_file());
    drop(first);
    let second = sidecar(&scratch.0, &content, &loader).unwrap();
    assert_eq!(second.root, path);
    let different = sidecar(&scratch.0, &"b".repeat(64), &loader).unwrap();
    assert_ne!(different.root, path);
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
async fn boot_imports_before_dependencies_and_metadata_gate_precedes_tensor_payload() {
    gated_fixture(false, true).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancellation_while_dependency_is_pending_stops_early_child_and_bridge() {
    gated_fixture(true, true).await;
}
async fn gated_fixture(cancel_before_dependencies: bool, early_launch: bool) {
    let mut nonce = [0; 8];
    getrandom::fill(&mut nonce).unwrap();
    let scratch =
        Scratch(std::env::temp_dir().join(format!("extent-agent-{:x}", u64::from_le_bytes(nonce))));
    let runtime = scratch.0.join("runtime-source");
    let source = scratch.0.join("model-source");
    fs::create_dir_all(&runtime).unwrap();
    fs::create_dir(&source).unwrap();
    fs::write(runtime.join("engine"),"#!/bin/sh\nprintf '%s' \"$PYTHONPATH\" > \"$METADATA.environment\"\nprintf '%s' \"$$\" > \"$IMPORTED\"\nread gate\ntest \"$gate\" = switchyard-metadata-ready-v1 || exit 1\ntest -f \"$3/config.json\" || exit 2\ntouch \"$METADATA\"\nsleep 60\n").unwrap();
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
    let dependency_source = scratch.0.join("dependency-source");
    fs::create_dir(&dependency_source).unwrap();
    fs::write(
        dependency_source.join("compiler-ready"),
        b"complete dependency",
    )
    .unwrap();
    let dependency_manifest = pack(
        &dependency_source,
        &scratch.0.join("dependency-objects"),
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
    for c in &dependency_manifest.chunks {
        bytes.insert(
            c.name.clone(),
            fs::read(scratch.0.join("dependency-objects").join(&c.name)).unwrap(),
        );
    }
    for e in &manifest.extents {
        bytes.insert(
            e.object.clone(),
            fs::read(scratch.0.join("model-objects").join(&e.object)).unwrap(),
        );
    }
    let (gate, waiting) = watch::channel(false);
    let (runtime_gate, runtime_waiting) = watch::channel(false);
    let (dependency_gate, dependency_waiting) = watch::channel(false);
    let runtime_requested = Arc::new(tokio::sync::Notify::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/{name}", get(object))
        .with_state(Objects {
            bytes: Arc::new(bytes),
            held,
            gate: waiting,
            held_runtime: runtime_manifest.chunks[0].name.clone(),
            runtime_gate: runtime_waiting,
            runtime_requested: runtime_requested.clone(),
            held_dependency: dependency_manifest.chunks[0].name.clone(),
            dependency_gate: dependency_waiting,
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
                    buffers: Default::default(),
                },
            },
        }),
        engine_args: vec!["serve".into(), "model".into()],
        concurrency: 20,
        stall_s: 10,
        lifetime_s: 30,
        startup: crate::startup::StartupPolicy {
            mode: crate::startup::HydrationMode::TrustedStreaming,
            early_launch,
            ..Default::default()
        },
        runtime: Some(crate::runtime::RuntimeTicket {
            components: vec![crate::runtime::RuntimeComponent {
                name: "build".into(),
                artifact: ArtifactTicket {
                    urls: dependency_manifest
                        .chunks
                        .iter()
                        .map(|c| (c.name.clone(), format!("{base}/{}", c.name)))
                        .collect(),
                    manifest: dependency_manifest,
                },
                compatibility: BTreeMap::new(),
            }],
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
                (
                    "PYTHONPATH".into(),
                    "{runtime_root}/providers:{cache_root}/compiled/sampling-provider".into(),
                ),
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
    let boot_root = roots.runtime.clone();
    let expected_python_suffix = format!(
        "{}/providers:{}/{}/compiled/sampling-provider",
        roots.runtime.display(),
        roots.caches.display(),
        ticket.runtime.as_ref().unwrap().artifact.manifest.id
    );

    let (selected, loader) = ticket.source.extent().unwrap();
    let content_id = selected.manifest.content_id.clone();
    let loader = loader.clone();
    let mut pending =
        tokio::spawn(async move { prepare(&ticket, &roots, state, Instant::now()).await });
    tokio::time::timeout(Duration::from_secs(5), runtime_requested.notified())
        .await
        .unwrap();
    // A control-plane reader holds the state while runtime reception queues an
    // update. Leave it held across multiple observer ticks. Both futures must
    // keep being polled when the reader releases the queued state writers.
    let reader = observed.read().await;
    runtime_gate.send(true).unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        observed.try_read().is_err(),
        "runtime state writer was not queued"
    );
    drop(reader);
    until(|| boot_root.exists()).await;
    if early_launch {
        until(|| imported.exists()).await;
    } else {
        assert!(
            !imported.exists(),
            "disabled early launch started before dependencies"
        );
    }
    assert!(
        !metadata_ready.exists(),
        "engine gate released before its dependency was published"
    );
    assert!(!pending.is_finished());
    if cancel_before_dependencies {
        let pid = fs::read_to_string(&imported)
            .unwrap()
            .parse::<i32>()
            .unwrap();
        pending.abort();
        assert!(matches!(pending.await, Err(ref error) if error.is_cancelled()));
        until(|| !engine::live_group(nix::unistd::Pid::from_raw(pid)).unwrap()).await;
        // Drop may retain a directory until its child has actually stopped. Once the
        // group is proven gone, a fresh lease must safely reclaim any abandoned tree.
        let replacement = sidecar(&model_root, &content_id, &loader).unwrap();
        let reclaimed = replacement.root.clone();
        assert!(!reclaimed.join("weights.sock").exists());
        drop(replacement);
        assert!(!reclaimed.exists());
        assert!(!metadata_ready.exists());
        server.abort();
        return;
    }
    dependency_gate.send(true).unwrap();
    tokio::select! {
        _ = until(|| metadata_ready.exists()) => {},
        result = &mut pending => match result {
            Ok(Err(error)) => panic!("startup failed before metadata gate: {error}"),
            Ok(Ok(_)) => panic!("startup completed while tensor payload is held"),
            Err(error) => panic!("startup task failed before metadata gate: {error}"),
        },
    }
    assert!(imported.exists());
    let environment =
        fs::read_to_string(format!("{}.environment", metadata_ready.display())).unwrap();
    assert!(environment.ends_with(&expected_python_suffix));
    assert!(!environment.contains("{cache_root}"));

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
                .filter(|p| p.is_dir())
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
    let milestones = &s.startup_milestones;
    let position = |event| {
        milestones
            .iter()
            .position(|m| std::mem::discriminant(&m.event) == std::mem::discriminant(&event))
            .unwrap()
    };
    if early_launch {
        assert!(
            position(StartupEvent::RuntimeBootReady)
                < position(StartupEvent::RuntimeHydrationFinished)
        );
        assert!(
            position(StartupEvent::EngineSpawned)
                < position(StartupEvent::RuntimeHydrationFinished)
        );
    } else {
        assert!(
            position(StartupEvent::RuntimeHydrationFinished)
                < position(StartupEvent::RuntimeBootReady)
        );
        assert!(position(StartupEvent::RuntimeBootReady) < position(StartupEvent::EngineSpawned));
    }
    assert!(
        position(StartupEvent::RuntimeHydrationFinished)
            < position(StartupEvent::EngineGateReleased)
    );
    drop(s);
    process.stop(Duration::from_secs(2)).await.unwrap();
    drop(process);
    until(|| !owner.exists()).await;
    assert!(model_root.join("model/config.json").exists());
    server.abort();
}

#[tokio::test]
async fn disabled_early_launch_waits_for_supplemental_runtime_publication() {
    gated_fixture(false, false).await;
}
