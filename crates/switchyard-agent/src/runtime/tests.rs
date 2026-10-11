use super::*;

use crate::Manifest;
use axum::{
    Router,
    body::Body,
    extract::{Path as RoutePath, State as AxumState},
    response::Response,
    routing::get,
};
use std::{
    collections::BTreeMap,
    fs,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use switchyard_artifacts::{Metadata, PackOptions, pack};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);
struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "switchyard-runtime-{}-{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct Source {
    bodies: Arc<BTreeMap<String, Vec<u8>>>,
    calls: Arc<AtomicUsize>,
    corrupt_first: Arc<AtomicBool>,
}
struct Server {
    task: tokio::task::JoinHandle<()>,
    calls: Arc<AtomicUsize>,
    corrupt_first: Arc<AtomicBool>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn chunk(
    AxumState(source): AxumState<Source>,
    RoutePath(name): RoutePath<String>,
) -> Response {
    source.calls.fetch_add(1, Ordering::Relaxed);
    let mut body = source.bodies[&name].clone();
    if source.corrupt_first.swap(false, Ordering::Relaxed) {
        body[0] ^= 1;
    }
    Response::builder()
        .status(200)
        .body(Body::from(body))
        .unwrap()
}
async fn serve(tickets: &mut [&mut ArtifactTicket], bodies: BTreeMap<String, Vec<u8>>) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    for ticket in tickets {
        ticket.urls = ticket
            .manifest
            .chunks
            .iter()
            .map(|c| {
                (
                    c.name.clone(),
                    format!("http://{address}/{}?signature=must-not-leak", c.name),
                )
            })
            .collect();
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let corrupt_first = Arc::new(AtomicBool::new(false));
    let app = Router::new()
        .route("/{name}", get(chunk))
        .with_state(Source {
            bodies: Arc::new(bodies),
            calls: calls.clone(),
            corrupt_first: corrupt_first.clone(),
        });
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Server {
        task,
        calls,
        corrupt_first,
    }
}
fn packed(
    source: &Path,
    compatibility: BTreeMap<String, String>,
    kind: &str,
    bodies: &mut BTreeMap<String, Vec<u8>>,
) -> ArtifactTicket {
    let output = TempDir::new();
    let manifest = pack(
        source,
        &output.0,
        Metadata {
            kind: kind.into(),
            compatibility,
            attributes: BTreeMap::new(),
        },
        PackOptions {
            target_shard_bytes: 1024,
            zstd_level: 1,
        },
    )
    .unwrap();
    for chunk in &manifest.chunks {
        bodies.insert(
            chunk.name.clone(),
            fs::read(output.0.join(&chunk.name)).unwrap(),
        );
    }
    ArtifactTicket {
        manifest,
        urls: BTreeMap::new(),
    }
}
fn ticket(runtime: RuntimeTicket) -> Ticket {
    Ticket {
        source: crate::model::ModelSource::Legacy(crate::model::LegacySource {
            manifest: Manifest {
                format: "shards-v1".into(),
                compress: "none".into(),
                chunks: Vec::new(),
                total_size: 123,
                raw_size: 0,
                sha256: "0".repeat(64),
                files: BTreeMap::new(),
            },
            urls: BTreeMap::new(),
        }),
        model: "test-model".into(),
        revision: "model-commit".into(),
        engine_args: vec![
            "serve".into(),
            "--model".into(),
            "/work/switchyard/model".into(),
        ],
        concurrency: 2,
        stall_s: 20,
        lifetime_s: 300,
        startup: Default::default(),
        runtime: Some(runtime),
    }
}
async fn fixture() -> (Ticket, Server) {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let source = TempDir::new();
    fs::create_dir(source.0.join("bin")).unwrap();
    let executable = source.0.join("bin/python-real");
    fs::write(&executable, b"#!/bin/sh\nprintf '%s\\n' \"$0\"\n").unwrap();
    fs::set_permissions(executable, fs::Permissions::from_mode(0o755)).unwrap();
    symlink("python-real", source.0.join("bin/python3")).unwrap();
    let mut bodies = BTreeMap::new();
    let mut runtime = packed(
        &source.0,
        BTreeMap::from([("architecture".into(), std::env::consts::ARCH.into())]),
        "runtime",
        &mut bodies,
    );
    let cache_source = TempDir::new();
    fs::create_dir(cache_source.0.join("vllm")).unwrap();
    fs::write(
        cache_source.0.join("vllm/kernels.bin"),
        b"verified compilation seed",
    )
    .unwrap();
    let mut cache = packed(
        &cache_source.0,
        BTreeMap::from([
            ("runtime_id".into(), runtime.manifest.id.clone()),
            ("model_revision".into(), "model-commit".into()),
            ("gpu".into(), "test-gpu".into()),
            ("engine_configuration".into(), "test-config".into()),
        ]),
        "cache",
        &mut bodies,
    );
    let server = serve(&mut [&mut runtime, &mut cache], bodies).await;
    (
        ticket(RuntimeTicket {
            hydration: None,
            artifact: runtime,
            compatibility: BTreeMap::new(),
            program: "bin/python3".into(),
            args: vec!["-m".into(), "vllm.entrypoints.cli.main".into()],
            environment: BTreeMap::from([
                ("VLLM_CACHE_ROOT".into(), "{cache_root}/engine/vllm".into()),
                ("PYTHONPATH".into(), "{runtime_root}/lib".into()),
            ]),
            caches: vec![CacheTicket {
                name: "engine".into(),
                artifact: cache,
                compatibility: BTreeMap::from([("gpu".into(), "test-gpu".into())]),
            }],
            cache_export: None,
        }),
        server,
    )
}

#[tokio::test]
async fn trusted_model_reuse_passes_capacity_without_another_model_copy_or_transfer() {
    use crate::{
        Chunk, FileInfo,
        model::{ExtentModel, ExtentSource, LegacySource, ModelSource},
    };
    use std::os::unix::fs::PermissionsExt;
    use switchyard_artifacts::{
        Entry,
        extents::{Extent, ExtentManifest, ExtentTicket},
    };
    for extents in [false, true] {
        let (mut ticket, server) = fixture().await;
        ticket.runtime.as_mut().unwrap().caches.clear();
        ticket.startup.mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
        ticket.startup.require_model_ram = false;
        let scratch = TempDir::new();
        let roots = RootPaths::under(&scratch.0);
        fs::create_dir_all(&roots.models).unwrap();
        // Sparse output exceeds free space without consuming it. The trusted
        // receipt represents an existing publication; no content verification
        // or native model loading is claimed by this capacity regression.
        let huge = crate::startup::observe(&roots.models, 0)
            .unwrap()
            .available_bytes
            + 128 * 1024 * 1024;
        let origin = reqwest::Url::parse(
            ticket
                .runtime
                .as_ref()
                .unwrap()
                .artifact
                .urls
                .values()
                .next()
                .unwrap(),
        )
        .unwrap()
        .origin()
        .ascii_serialization();
        ticket.source = if extents {
            ModelSource::Extents(ExtentSource {
                model_source: ExtentModel::Extents {
                    artifact: ExtentTicket {
                        manifest: ExtentManifest {
                            format: "switchyard-extents-v1".into(),
                            id: "1".repeat(64),
                            content_id: "2".repeat(64),
                            metadata: Metadata {
                                kind: "weights".into(),
                                compatibility: BTreeMap::new(),
                                attributes: BTreeMap::new(),
                            },
                            entries: vec![
                                Entry::File {
                                    path: "config.json".into(),
                                    mode: 0o644,
                                    size: 2,
                                    sha256: "0".repeat(64),
                                },
                                Entry::File {
                                    path: "model.safetensors".into(),
                                    mode: 0o644,
                                    size: huge,
                                    sha256: "0".repeat(64),
                                },
                            ],
                            extents: vec![
                                Extent {
                                    entry: 0,
                                    offset: 0,
                                    length: 2,
                                    object: "config".into(),
                                    object_offset: 0,
                                    object_size: 2,
                                    sha256: "0".repeat(64),
                                },
                                Extent {
                                    entry: 1,
                                    offset: 0,
                                    length: huge,
                                    object: "weights".into(),
                                    object_offset: 0,
                                    object_size: huge,
                                    sha256: "0".repeat(64),
                                },
                            ],
                        },
                        urls: BTreeMap::from([
                            ("config".into(), format!("{origin}/config")),
                            ("weights".into(), format!("{origin}/weights")),
                        ]),
                    },
                    loader: Default::default(),
                },
            })
        } else {
            ModelSource::Legacy(LegacySource {
                manifest: Manifest {
                    format: "shards-v1".into(),
                    compress: "none".into(),
                    total_size: huge + 1024,
                    raw_size: huge + 1024,
                    sha256: "0".repeat(64),
                    chunks: vec![Chunk {
                        name: "weights".into(),
                        size: huge + 1024,
                        raw: huge + 1024,
                        shard: 0,
                        seq: 0,
                        sha256: "0".repeat(64),
                    }],
                    files: BTreeMap::from([(
                        "model.safetensors".into(),
                        FileInfo {
                            size: huge,
                            sha256: None,
                        },
                    )]),
                },
                urls: BTreeMap::from([("weights".into(), format!("{origin}/weights"))]),
            })
        };
        let state = Arc::new(RwLock::new(State::new(&ticket)));
        assert!(
            crate::hydrate::preflight(&ticket, &roots.models, &state)
                .await
                .is_err()
        );
        let published = roots.models.join("model");
        fs::create_dir(&published).unwrap();
        fs::File::create(published.join("model.safetensors"))
            .unwrap()
            .set_len(huge)
            .unwrap();
        let marker = if extents {
            fs::write(published.join("config.json"), b"{}").unwrap();
            let marker = published.join(".switchyard-extents-id");
            fs::write(&marker, format!("{}\n{}\n", "1".repeat(64), "2".repeat(64))).unwrap();
            marker
        } else {
            let marker = published.join(".switchyard-archive");
            fs::write(&marker, "0".repeat(64)).unwrap();
            marker
        };
        fs::set_permissions(marker, fs::Permissions::from_mode(0o600)).unwrap();
        crate::hydrate::preflight(&ticket, &roots.models, &state)
            .await
            .unwrap();
        preflight(&ticket, &roots, &state).await.unwrap();
        crate::hydrate::hydrate_with_limit(
            &ticket,
            &roots.models,
            state.clone(),
            Arc::new(tokio::sync::Semaphore::new(20)),
        )
        .await
        .unwrap();
        let state = state.read().await;
        assert!(state.model_cached);
        assert_eq!(
            state
                .dependency_budget
                .as_ref()
                .unwrap()
                .model_output_required_bytes,
            if extents { 2 } else { 0 }
        );
        assert!(state.dependency_budget.as_ref().unwrap().model_bytes >= huge);
        assert_eq!(state.bytes_downloaded, 0);
        assert_eq!(server.calls.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn decoder_budget_includes_independent_model_and_runtime_workers() {
    let (mut ticket, _server) = fixture().await;
    ticket.concurrency = 20;
    let crate::model::ModelSource::Legacy(source) = &mut ticket.source else {
        unreachable!();
    };
    source.manifest.compress = "zstd".into();
    source.manifest.chunks = [0, 0, 1, 2]
        .into_iter()
        .enumerate()
        .map(|(seq, shard)| crate::Chunk {
            name: format!("chunk-{seq}"),
            size: 1,
            raw: 1,
            shard,
            seq,
            sha256: "0".repeat(64),
        })
        .collect();
    let history = 128 * 1024 * 1024;
    assert_eq!(budget::decoder_history_bound(&ticket), (8 + 3) * history);
    ticket.startup.mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
    assert_eq!(budget::decoder_history_bound(&ticket), (4 + 3) * history);
    ticket.concurrency = 2;
    assert_eq!(budget::decoder_history_bound(&ticket), (2 + 2) * history);
    ticket.runtime = None;
    assert_eq!(budget::decoder_history_bound(&ticket), 2 * history);
}

#[tokio::test]
async fn runtime_and_cache_hydrate_before_command_with_safe_environment() {
    let (ticket, server) = fixture().await;
    let destination = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let initial = state.read().await.clone();
    assert_eq!(initial.artifacts.len(), 2);
    assert!(initial.artifact_bytes_total > 0);
    assert_eq!(initial.bytes_total, 123);
    let command = command(&ticket, &roots, state.clone())
        .await
        .unwrap()
        .unwrap();
    let args = command
        .as_std()
        .get_args()
        .map(|a| a.to_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        args,
        [
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "--model",
            "/work/switchyard/model"
        ]
    );
    assert_eq!(
        command.as_std().get_program(),
        roots.runtime.join("bin/python3")
    );
    let env = command
        .as_std()
        .get_envs()
        .map(|(k, v)| {
            (
                k.to_str().unwrap().to_owned(),
                v.unwrap().to_string_lossy().into_owned(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let cache_root = roots
        .caches
        .join(&ticket.runtime.as_ref().unwrap().artifact.manifest.id);
    assert_eq!(
        env["VLLM_CACHE_ROOT"],
        cache_root.join("engine/vllm").to_string_lossy()
    );
    assert_eq!(
        env["PYTHONPATH"],
        roots.runtime.join("lib").to_string_lossy()
    );
    assert!(!env.contains_key("SWITCHYARD_AGENT_TOKEN"));
    assert!(!env.contains_key("SWITCHYARD_TICKET_URL"));
    assert!(!env.contains_key("RUNPOD_API_KEY"));
    let st = state.read().await;
    assert_eq!(st.artifact_bytes_downloaded, st.artifact_bytes_total);
    assert_eq!(
        st.artifact_network_bytes_downloaded,
        st.artifact_bytes_total
    );
    assert!(
        st.artifact_volume_bytes_per_s
            .is_some_and(|rate| rate > 0.0)
    );
    assert_eq!(st.bytes_downloaded, 0);
    assert!(st.artifacts.iter().all(|a| a.phase == ArtifactPhase::Ready));
    assert!(
        !serde_json::to_string(&*st)
            .unwrap()
            .contains("must-not-leak")
    );
    assert!(server.calls.load(Ordering::Relaxed) > 0);
}

#[tokio::test]
async fn executable_symlink_preserves_invoked_directory_for_virtual_environments() {
    let (ticket, _server) = fixture().await;
    let destination = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let output = command(&ticket, &roots, state)
        .await
        .unwrap()
        .unwrap()
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        roots.runtime.join("bin/python3").to_string_lossy()
    );
}

#[tokio::test]
async fn retried_runtime_chunks_separate_verified_progress_from_wire_bytes() {
    let (ticket, server) = fixture().await;
    server.corrupt_first.store(true, Ordering::Relaxed);
    let destination = TempDir::new();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    command(&ticket, &RootPaths::under(&destination.0), state.clone())
        .await
        .unwrap();
    let state = state.read().await;
    assert_eq!(state.artifact_bytes_downloaded, state.artifact_bytes_total);
    assert!(state.artifact_network_bytes_downloaded > state.artifact_bytes_total);
    assert_eq!(
        state
            .artifacts
            .iter()
            .map(|artifact| artifact.network_verified_bytes)
            .sum::<u64>(),
        state.artifact_bytes_total
    );
    assert!(
        state
            .artifact_volume_bytes_per_s
            .is_some_and(|rate| rate.is_finite() && rate > 0.0)
    );
}

#[tokio::test]
async fn runtime_transfer_uses_the_ticket_stall_timeout() {
    async fn stalled() -> &'static str {
        tokio::time::sleep(std::time::Duration::from_secs(20)).await;
        "never reached"
    }
    let (mut ticket, _source) = fixture().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = Server {
        task: tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/stall", get(stalled)))
                .await
                .unwrap();
        }),
        calls: Arc::new(AtomicUsize::new(0)),
        corrupt_first: Arc::new(AtomicBool::new(false)),
    };
    let runtime = ticket.runtime.as_mut().unwrap();
    for url in runtime.artifact.urls.values_mut() {
        *url = format!("http://{address}/stall?signature=must-not-leak");
    }
    ticket.stall_s = 1;
    let destination = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(6),
        command(&ticket, &roots, state.clone()),
    )
    .await
    .expect("a stalled runtime transfer should stop within its configured retry window");
    assert!(result.is_err());
    assert!(!roots.runtime.exists());
    let state = state.read().await;
    assert_eq!(state.artifacts[0].phase, ArtifactPhase::Failed);
    assert!(
        !serde_json::to_string(&*state)
            .unwrap()
            .contains("must-not-leak")
    );
    drop(server);
}

#[tokio::test]
async fn reused_seed_accepts_mutable_compiler_outputs_without_redownload() {
    let (ticket, server) = fixture().await;
    let destination = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    command(&ticket, &roots, state).await.unwrap();
    let calls = server.calls.load(Ordering::Relaxed);
    let runtime = ticket.runtime.as_ref().unwrap();
    let cache_root = roots.caches.join(&runtime.artifact.manifest.id);
    fs::write(
        cache_root.join("engine/vllm/new-kernel.bin"),
        b"engine-generated",
    )
    .unwrap();
    fs::write(
        cache_root.join("engine/vllm/kernels.bin"),
        b"updated compilation",
    )
    .unwrap();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    command(&ticket, &roots, state.clone()).await.unwrap();
    assert_eq!(server.calls.load(Ordering::Relaxed), calls);
    assert_eq!(
        fs::read(cache_root.join("engine/vllm/kernels.bin")).unwrap(),
        b"updated compilation"
    );
    switchyard_artifacts::verify(
        &cache_root.join(".seeds/engine"),
        &runtime.caches[0].artifact.manifest,
    )
    .unwrap();
    let state = state.read().await;
    assert!(state.artifacts.iter().all(|a| a.cached));
    assert_eq!(state.artifact_bytes_downloaded, state.artifact_bytes_total);
    assert_eq!(state.artifact_network_bytes_downloaded, 0);
    assert_eq!(state.artifact_volume_bytes_per_s, None);
}

#[tokio::test]
async fn cancelled_cache_copy_cannot_publish_after_its_final_worker_check() {
    let (ticket, _server) = fixture().await;
    let destination = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    command(&ticket, &roots, state).await.unwrap();
    let runtime = ticket.runtime.as_ref().unwrap();
    let cache_root = roots.caches.join(&runtime.artifact.manifest.id);
    let cache = cache_root.join("engine");
    fs::remove_dir_all(&cache).unwrap();
    let seed = cache_root.join(".seeds/engine");
    let manifest = runtime.caches[0].artifact.manifest.clone();
    let (prepared, ready) = tokio::sync::oneshot::channel();
    let (release, resume) = std::sync::mpsc::channel();
    let caller = tokio::spawn(cache::writable_copy_paused(
        seed.clone(),
        cache.clone(),
        manifest.clone(),
        prepared,
        resume,
    ));
    // The worker has copied every file, written its receipt, and passed its final cancellation
    // check. Cancellation now exercises the publication race rather than an early exit.
    ready.await.unwrap();
    let staging = fs::read_dir(&cache_root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".cache-staging-")
        })
        .unwrap();
    assert!(!cache.exists());
    assert_eq!(
        fs::read(staging.join("vllm/kernels.bin")).unwrap(),
        b"verified compilation seed"
    );
    assert_eq!(
        fs::read_to_string(staging.join(".switchyard-cache-seed")).unwrap(),
        manifest.id
    );
    caller.abort();
    assert!(caller.await.unwrap_err().is_cancelled());
    release.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while staging.exists() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("cancelled worker output must clean up its completed staging tree");
    assert!(!cache.exists());
    switchyard_artifacts::verify(&seed, &manifest).unwrap();
}

#[tokio::test]
async fn cache_compatibility_rejection_precedes_all_transfers() {
    let (mut ticket, server) = fixture().await;
    ticket.runtime.as_mut().unwrap().caches[0]
        .compatibility
        .insert("gpu".into(), "different-gpu".into());
    let destination = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let error = command(&ticket, &roots, state.clone())
        .await
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("compatibility"));
    assert!(!error.contains("different-gpu"));
    assert!(!error.contains("must-not-leak"));
    assert_eq!(server.calls.load(Ordering::Relaxed), 0);
    assert!(!roots.runtime.exists());
    assert_eq!(state.read().await.artifacts[1].phase, ArtifactPhase::Failed);
}

#[tokio::test]
async fn launch_rejects_unsafe_programs_cache_names_and_secret_environment() {
    let (ticket, server) = fixture().await;
    for program in ["/bin/python3", "../python3", "bin/missing"] {
        let mut invalid = ticket.clone();
        invalid.runtime.as_mut().unwrap().program = program.into();
        assert!(
            invalid
                .runtime
                .as_ref()
                .unwrap()
                .validate_launch(switchyard_artifacts::HydrationMode::Verified)
                .is_err()
        );
    }
    for name in ["../engine", "", "runtime", ".seeds"] {
        let mut invalid = ticket.clone();
        invalid.runtime.as_mut().unwrap().caches[0].name = name.into();
        assert!(
            invalid
                .runtime
                .as_ref()
                .unwrap()
                .validate_launch(switchyard_artifacts::HydrationMode::Verified)
                .is_err()
        );
    }
    for key in [
        "RUNPOD_API_KEY",
        "SWITCHYARD_AGENT_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "INVALID=NAME",
    ] {
        let mut invalid = ticket.clone();
        invalid
            .runtime
            .as_mut()
            .unwrap()
            .environment
            .insert(key.into(), "sensitive-value".into());
        let destination = TempDir::new();
        let roots = RootPaths::under(&destination.0);
        let state = Arc::new(RwLock::new(State::new(&invalid)));
        let error = command(&invalid, &roots, state)
            .await
            .err()
            .unwrap()
            .to_string();
        assert!(!error.contains("sensitive-value"));
    }
    assert_eq!(server.calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn writable_cache_rejects_unowned_existing_destination() {
    let (ticket, _server) = fixture().await;
    let destination = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    let cache_root = roots
        .caches
        .join(&ticket.runtime.as_ref().unwrap().artifact.manifest.id);
    fs::create_dir_all(cache_root.join("engine")).unwrap();
    fs::write(cache_root.join("engine/keep"), b"preserve").unwrap();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    assert!(command(&ticket, &roots, state).await.is_err());
    assert_eq!(
        fs::read(cache_root.join("engine/keep")).unwrap(),
        b"preserve"
    );
}

#[tokio::test]
async fn cache_root_symlink_is_rejected_without_creating_foreign_directories() {
    let (ticket, _server) = fixture().await;
    let destination = TempDir::new();
    let foreign = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    std::os::unix::fs::symlink(&foreign.0, &roots.caches).unwrap();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    assert!(command(&ticket, &roots, state).await.is_err());
    assert_eq!(fs::read_dir(&foreign.0).unwrap().count(), 0);
}

#[tokio::test]
async fn legacy_ticket_and_observation_defaults_remain_compatible() {
    let (ticket, _) = fixture().await;
    let mut value = serde_json::to_value(&ticket).unwrap();
    value.as_object_mut().unwrap().remove("runtime");
    let legacy: Ticket = serde_json::from_value(value).unwrap();
    assert!(legacy.runtime.is_none());
    let mut observation = serde_json::to_value(State::new(&legacy)).unwrap();
    for name in [
        "artifact_bytes_downloaded",
        "artifact_bytes_total",
        "artifact_network_bytes_downloaded",
        "artifact_volume_bytes_per_s",
        "artifacts",
        "inference_verified",
    ] {
        observation.as_object_mut().unwrap().remove(name);
    }
    let state: State = serde_json::from_value(observation).unwrap();
    assert!(state.artifacts.is_empty());
    assert_eq!(state.artifact_bytes_total, 0);
    assert!(!state.inference_verified);
    let destination = TempDir::new();
    assert!(
        command(
            &legacy,
            &RootPaths::under(&destination.0),
            Arc::new(RwLock::new(state))
        )
        .await
        .unwrap()
        .is_none()
    );
}

#[tokio::test]
async fn actual_host_compatibility_checks_architecture_and_every_gpu() {
    let (mut ticket, _) = fixture().await;
    let runtime = ticket.runtime.as_mut().unwrap();
    assert!(compatibility::check(runtime, "wrong-architecture", &[]).is_err());
    runtime.caches[0]
        .artifact
        .manifest
        .metadata
        .compatibility
        .insert("gpu_compute_capability".into(), "08.09".into());
    assert!(compatibility::check(runtime, std::env::consts::ARCH, &["8.9".into()]).is_ok());
    assert!(
        compatibility::check(
            runtime,
            std::env::consts::ARCH,
            &["8.9".into(), "8.0".into()]
        )
        .is_err()
    );
    assert!(compatibility::check(runtime, std::env::consts::ARCH, &[]).is_err());
    runtime.caches[0]
        .artifact
        .manifest
        .metadata
        .compatibility
        .remove("gpu_compute_capability");
    runtime
        .artifact
        .manifest
        .metadata
        .compatibility
        .insert("gpu_compute_capability".into(), "8.9".into());
    assert!(compatibility::check(runtime, std::env::consts::ARCH, &["8.0".into()]).is_err());
    assert!(compatibility::check(runtime, std::env::consts::ARCH, &["8.9".into()]).is_ok());
    runtime
        .artifact
        .manifest
        .metadata
        .compatibility
        .remove("gpu_compute_capability");
    runtime.cache_export = Some(crate::export::CacheExportConfig {
        name: "compiled".into(),
        compatibility: BTreeMap::from([("gpu_compute_capability".into(), "8.9".into())]),
        drain_timeout_s: 60,
        shutdown_timeout_s: 30,
    });
    assert!(compatibility::check(runtime, std::env::consts::ARCH, &["8.0".into()]).is_err());
    assert!(compatibility::check(runtime, std::env::consts::ARCH, &["8.9".into()]).is_ok());
}

#[tokio::test]
async fn trusted_cache_copy_rejects_a_symlink_in_a_reused_seed() {
    let (mut ticket, _server) = fixture().await;
    ticket.startup.mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
    let destination = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    command(&ticket, &roots, Arc::new(RwLock::new(State::new(&ticket))))
        .await
        .unwrap();
    let cache_root = roots
        .caches
        .join(&ticket.runtime.as_ref().unwrap().artifact.manifest.id);
    let seed_file = cache_root.join(".seeds/engine/vllm/kernels.bin");
    let outside = destination.0.join("outside");
    fs::write(&outside, b"do not copy").unwrap();
    fs::remove_file(&seed_file).unwrap();
    std::os::unix::fs::symlink(&outside, &seed_file).unwrap();
    fs::remove_dir_all(cache_root.join("engine")).unwrap();
    let error = command(&ticket, &roots, Arc::new(RwLock::new(State::new(&ticket))))
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("regular file"));
    assert!(!cache_root.join("engine").exists());
    assert_eq!(fs::read(outside).unwrap(), b"do not copy");
}

#[tokio::test]
async fn configured_zero_runtime_retries_preserves_first_failed_attempt() {
    let (mut ticket, server) = fixture().await;
    ticket.concurrency = 1;
    let runtime = ticket.runtime.as_mut().unwrap();
    runtime.hydration = Some(RuntimeHydrationOptions { retries: 0 });
    server.corrupt_first.store(true, Ordering::Relaxed);
    let destination = TempDir::new();
    let roots = RootPaths::under(&destination.0);
    assert!(
        command(&ticket, &roots, Arc::new(RwLock::new(State::new(&ticket))))
            .await
            .is_err()
    );
    assert_eq!(server.calls.load(Ordering::Relaxed), 1);
    assert!(!roots.runtime.exists());

    let mut invalid = ticket.clone();
    invalid.runtime.as_mut().unwrap().hydration = Some(RuntimeHydrationOptions { retries: 11 });
    assert!(
        invalid
            .runtime
            .as_ref()
            .unwrap()
            .validate_launch(switchyard_artifacts::HydrationMode::Verified)
            .is_err()
    );
}
