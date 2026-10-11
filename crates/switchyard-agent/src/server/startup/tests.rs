use super::*;
#[tokio::test]
async fn cooperative_engine_waits_for_current_model_gate_and_observes_eof() {
    let mut command = Command::new("sh");
    command
        .args([
            "-c",
            "read line; test \"$line\" = switchyard-model-ready-v1 && ! read extra",
        ])
        .stdin(Stdio::piped());
    let mut process = engine::Process::spawn(&mut command).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    assert!(process.child.try_wait().unwrap().is_none());
    release(&mut process).await.unwrap();
    assert!(process.child.wait().await.unwrap().success());
}
#[tokio::test]
async fn cancellation_before_gate_kills_waiting_engine_group() {
    let mut command = Command::new("sh");
    command
        .args(["-c", "read line; sleep 60"])
        .stdin(Stdio::piped());
    let process = engine::Process::spawn(&mut command).unwrap();
    let pid = process.child.id().unwrap();
    drop(process);
    for _ in 0..100 {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"));
        if stat.is_err()
            || stat.as_ref().is_ok_and(|stat| {
                stat.rsplit_once(')')
                    .unwrap()
                    .1
                    .trim_start()
                    .starts_with('Z')
            })
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("cancelled cooperative engine remains live");
}

#[tokio::test]
async fn runtime_imports_while_current_model_stream_is_pending_and_gate_follows_publication() {
    use axum::{
        Router,
        body::Body,
        extract::{Path as RoutePath, State as AxumState},
        routing::get,
    };
    use std::{collections::BTreeMap, fs, path::PathBuf};
    use switchyard_artifacts::{ArtifactTicket, Metadata, PackOptions, pack};
    #[derive(Clone)]
    struct Source {
        bodies: Arc<BTreeMap<String, Vec<u8>>>,
        barrier: Arc<tokio::sync::Barrier>,
        release: Arc<tokio::sync::Notify>,
    }
    async fn piece(
        AxumState(source): AxumState<Source>,
        RoutePath(name): RoutePath<String>,
    ) -> Body {
        source.barrier.wait().await;
        if name == "model-piece" {
            source.release.notified().await;
        }
        Body::from(source.bodies[&name].clone())
    }
    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let mut random = [0_u8; 8];
    getrandom::fill(&mut random).unwrap();
    let scratch = Scratch(std::env::temp_dir().join(format!(
        "switchyard-overlap-{:x}",
        u64::from_le_bytes(random)
    )));
    fs::create_dir(&scratch.0).unwrap();
    let source_dir = scratch.0.join("source");
    fs::create_dir(&source_dir).unwrap();
    let executable = source_dir.join("engine");
    fs::write(&executable, "#!/bin/sh\ntouch \"$IMPORTED\"\nread line\ntest \"$line\" = switchyard-model-ready-v1 || exit 1\ntest -f \"$MODEL/config.json\" || exit 2\ntouch \"$RELEASED\"\nsleep 60\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
    let packed = scratch.0.join("packed");
    fs::create_dir(&packed).unwrap();
    let runtime_manifest = pack(
        &source_dir,
        &packed,
        Metadata {
            kind: "runtime".into(),
            compatibility: BTreeMap::from([("architecture".into(), std::env::consts::ARCH.into())]),
            attributes: BTreeMap::new(),
        },
        PackOptions::default(),
    )
    .unwrap();
    assert_eq!(runtime_manifest.chunks.len(), 1);
    let mut bodies = BTreeMap::new();
    for chunk in &runtime_manifest.chunks {
        bodies.insert(
            chunk.name.clone(),
            fs::read(packed.join(&chunk.name)).unwrap(),
        );
    }
    let mut tar = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(2);
    header.set_mode(0o644);
    header.set_cksum();
    tar.append_data(&mut header, "config.json", b"{}".as_slice())
        .unwrap();
    let model_bytes = tar.into_inner().unwrap();
    bodies.insert("model-piece".into(), model_bytes.clone());
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let release = Arc::new(tokio::sync::Notify::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/{name}", get(piece))
        .with_state(Source {
            bodies: Arc::new(bodies),
            barrier,
            release: release.clone(),
        });
    let source_server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let roots = RootPaths::under(&scratch.0);
    let imported = scratch.0.join("imported");
    let released = scratch.0.join("released");
    let ticket = Ticket {
        source: crate::model::ModelSource::Legacy(crate::model::LegacySource {
            manifest: crate::Manifest {
                format: "shards-v1".into(),
                compress: "none".into(),
                sha256: "a".repeat(64),
                total_size: model_bytes.len() as u64,
                raw_size: model_bytes.len() as u64,
                chunks: vec![crate::Chunk {
                    name: "model-piece".into(),
                    size: model_bytes.len() as u64,
                    raw: model_bytes.len() as u64,
                    shard: 0,
                    seq: 0,
                    sha256: "b".repeat(64),
                }],
                files: BTreeMap::from([(
                    "config.json".into(),
                    crate::FileInfo {
                        size: 2,
                        sha256: None,
                    },
                )]),
            },
            urls: BTreeMap::from([("model-piece".into(), format!("{base}/model-piece"))]),
        }),
        model: "test".into(),
        revision: "revision".into(),
        concurrency: 2,
        stall_s: 10,
        lifetime_s: 60,
        engine_args: vec![],
        startup: crate::startup::StartupPolicy {
            mode: switchyard_artifacts::HydrationMode::TrustedStreaming,
            early_launch: true,
            ..Default::default()
        },
        runtime: Some(crate::runtime::RuntimeTicket {
            components: vec![],
            hydration: None,
            artifact: ArtifactTicket {
                urls: runtime_manifest
                    .chunks
                    .iter()
                    .map(|chunk| (chunk.name.clone(), format!("{base}/{}", chunk.name)))
                    .collect(),
                manifest: runtime_manifest,
            },
            compatibility: BTreeMap::new(),
            program: "engine".into(),
            args: vec![],
            caches: vec![],
            cache_export: None,
            environment: BTreeMap::from([
                ("IMPORTED".into(), imported.to_string_lossy().into_owned()),
                ("RELEASED".into(), released.to_string_lossy().into_owned()),
                (
                    "MODEL".into(),
                    roots.models.join("model").to_string_lossy().into_owned(),
                ),
            ]),
        }),
    };
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let observed = state.clone();
    let preparing =
        tokio::spawn(async move { prepare(&ticket, &roots, state, Instant::now()).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !imported.exists() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(!released.exists());
    assert!(
        observed
            .read()
            .await
            .startup_milestones
            .iter()
            .any(|milestone| matches!(milestone.event, StartupEvent::EngineSpawned))
    );
    release.notify_one();
    let process = tokio::time::timeout(std::time::Duration::from_secs(5), preparing)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !released.exists() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let state = observed.read().await;
    assert!(matches!(state.phase, Phase::Initializing));
    assert!(
        state
            .startup_milestones
            .windows(2)
            .all(|pair| pair[0].elapsed_s <= pair[1].elapsed_s)
    );
    let events = state
        .startup_milestones
        .iter()
        .map(|milestone| &milestone.event)
        .collect::<Vec<_>>();
    let spawned = events
        .iter()
        .position(|event| matches!(event, StartupEvent::EngineSpawned))
        .unwrap();
    let hydrated = events
        .iter()
        .position(|event| matches!(event, StartupEvent::ModelHydrationFinished))
        .unwrap();
    let gate = events
        .iter()
        .position(|event| matches!(event, StartupEvent::ModelGateReleased))
        .unwrap();
    assert!(spawned < hydrated && hydrated < gate);
    assert!(
        state
            .artifacts
            .iter()
            .all(|artifact| artifact.verified_chunks == 0)
    );
    drop(state);
    drop(process);
    source_server.abort();
}
