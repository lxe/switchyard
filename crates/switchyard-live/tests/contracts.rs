//! Exercise the live scheduler at real HTTP boundaries without renting hardware.
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, Response, StatusCode},
    routing::get,
};
use serde_json::json;
use std::{
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};
use switchyard_core::{FleetView, Policy, Scheduler, Scheduling};
use switchyard_live::config::DeploymentConfig;
use switchyard_live::{LiveRuntime, WorkCall, planner::Message};
use switchyard_provider::{Launch, Offering, Pod, PodStatus, Provider, storage::Storage};

#[derive(Clone)]
struct MockProvider {
    url: String,
    pods: Arc<Mutex<Vec<Pod>>>,
    unknown: bool,
    bake: Option<MockBake>,
    rejected_allocations: Arc<Mutex<VecDeque<u16>>>,
    allocation_attempts: Arc<Mutex<Vec<String>>>,
    create_delay_ms: Arc<AtomicU64>,
    agent_failure: Arc<Mutex<Option<String>>>,
    state_delay_ms: Arc<AtomicU64>,
    state_reads: Arc<AtomicUsize>,
}
impl Provider for MockProvider {
    fn endpoint_url(&self, _: &Pod, _: u16) -> String {
        self.url.clone()
    }
    async fn offers(&self) -> switchyard_provider::Result<Vec<Offering>> {
        Ok(vec![Offering {
            gpu: "mock".into(),
            gpu_memory_gb: 48,
            data_center: "local".into(),
            availability: "HIGH".into(),
            gpu_usd_per_hour: 1.0,
            disk_usd_per_hour: 0.01,
            quoted_at_unix_s: 0,
        }])
    }
    async fn create(&self, launch: Launch) -> switchyard_provider::Result<Pod> {
        self.allocation_attempts
            .lock()
            .unwrap()
            .push(launch.name.clone());
        tokio::time::sleep(std::time::Duration::from_millis(
            self.create_delay_ms.load(Ordering::SeqCst),
        ))
        .await;
        if let Some(status) = self.rejected_allocations.lock().unwrap().pop_front() {
            return Err(switchyard_provider::Error::AllocationRejected { status });
        }
        if let Some(bake) = &self.bake {
            let mut state = bake.state.lock().unwrap();
            state.token = Some(launch.container.environment["SWITCHYARD_AGENT_TOKEN"].clone());
            state.events.push("provision".into());
        }
        let mut pods = self.pods.lock().unwrap();
        let pod = Pod {
            id: format!("pod{}", pods.len()),
            name: launch.name.clone(),
            status: PodStatus::Running,
            cost: 1.0,
            data_center: Some("local".into()),
        };
        pods.push(pod.clone());
        if self.unknown {
            Err(switchyard_provider::Error::UnknownLaunch { name: launch.name })
        } else {
            Ok(pod)
        }
    }
    async fn get(&self, id: &str) -> switchyard_provider::Result<Option<Pod>> {
        Ok(self
            .pods
            .lock()
            .unwrap()
            .iter()
            .find(|p| p.id == id)
            .cloned())
    }
    async fn list(&self) -> switchyard_provider::Result<Vec<Pod>> {
        Ok(self.pods.lock().unwrap().clone())
    }
    async fn terminate(&self, id: &str) -> switchyard_provider::Result<()> {
        if let Some(bake) = &self.bake {
            bake.state.lock().unwrap().events.push("terminate".into());
        }
        self.pods
            .lock()
            .unwrap()
            .iter_mut()
            .find(|p| p.id == id)
            .unwrap()
            .status = PodStatus::Terminated;
        Ok(())
    }
}

// Optional cache-export capability extends the same provider, tokenizer, and
// inference harness. Only the host's export/status and object-store HTTP are mocked.
#[derive(Clone, Default)]
struct MockBake {
    fail_export: bool,
    fail_publication: bool,
    state: Arc<Mutex<BakeState>>,
}
#[derive(Default)]
struct BakeState {
    manifest: Option<switchyard_artifacts::Manifest>,
    chunks: BTreeMap<String, Vec<u8>>,
    objects: BTreeMap<String, Vec<u8>>,
    tickets: Vec<switchyard_agent::Ticket>,
    deleted: Vec<String>,
    token: Option<String>,
    requests: Vec<serde_json::Value>,
    events: Vec<String>,
    completion_delay_ms: u64,
}
#[derive(Clone)]
struct HttpFixture {
    bad_usage: bool,
    model_manifest: serde_json::Value,
    bake: Option<MockBake>,
    agent_failure: Arc<Mutex<Option<String>>>,
    state_delay_ms: Arc<AtomicU64>,
    state_reads: Arc<AtomicUsize>,
}

async fn object_get(State(state): State<HttpFixture>, Path(key): Path<String>) -> Response<Body> {
    if let Some(bake) = &state.bake
        && let Some(bytes) = bake.state.lock().unwrap().objects.get(&key)
    {
        return Response::builder()
            .header("content-type", "application/json")
            .body(Body::from(bytes.clone()))
            .unwrap();
    }
    if key.starts_with("registry/") || key.starts_with("learned/") {
        return Response::builder().status(404).body(Body::empty()).unwrap();
    }
    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&state.model_manifest).unwrap(),
        ))
        .unwrap()
}
async fn object_put(
    State(state): State<HttpFixture>,
    Path(key): Path<String>,
    bytes: axum::body::Bytes,
) -> StatusCode {
    if let Some(bake) = &state.bake {
        if bake.fail_publication && !key.starts_with("tickets/") {
            return StatusCode::BAD_REQUEST;
        }
        let mut state = bake.state.lock().unwrap();
        if key.starts_with("tickets/") {
            state.events.push("ticket-put".into());
            state.tickets.push(serde_json::from_slice(&bytes).unwrap());
        } else {
            state.events.push(
                if key.starts_with("registry/") {
                    "registry-put"
                } else {
                    "artifact-put"
                }
                .into(),
            );
        }
        state.objects.insert(key, bytes.to_vec());
    }
    StatusCode::NO_CONTENT
}
async fn object_delete(State(state): State<HttpFixture>, Path(key): Path<String>) -> StatusCode {
    if let Some(bake) = &state.bake {
        let mut state = bake.state.lock().unwrap();
        state.events.push("ticket-delete".into());
        state.objects.remove(&key);
        state.deleted.push(key);
    }
    StatusCode::NO_CONTENT
}
fn export_status(state: &HttpFixture, headers: &HeaderMap, start: bool) -> Response<Body> {
    let Some(bake) = &state.bake else {
        return Response::builder().status(404).body(Body::empty()).unwrap();
    };
    let mut state = bake.state.lock().unwrap();
    if headers.get("authorization").and_then(|v| v.to_str().ok())
        != state
            .token
            .as_ref()
            .map(|v| format!("Bearer {v}"))
            .as_deref()
    {
        return Response::builder().status(401).body(Body::empty()).unwrap();
    }
    state.events.push(
        if start {
            "export-start"
        } else {
            "export-status"
        }
        .into(),
    );
    let status = switchyard_agent::export::ExportState {
        phase: if start {
            switchyard_agent::export::ExportPhase::Packing
        } else if bake.fail_export {
            switchyard_agent::export::ExportPhase::Failed
        } else {
            switchyard_agent::export::ExportPhase::Ready
        },
        manifest: if start || bake.fail_export {
            None
        } else {
            state.manifest.clone()
        },
        error: bake
            .fail_export
            .then(|| "fixture engine cache preparation failed".into()),
    };
    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&status).unwrap()))
        .unwrap()
}
async fn export_start(State(state): State<HttpFixture>, headers: HeaderMap) -> Response<Body> {
    export_status(&state, &headers, true)
}
async fn export_observe(State(state): State<HttpFixture>, headers: HeaderMap) -> Response<Body> {
    export_status(&state, &headers, false)
}
async fn export_chunk(
    State(state): State<HttpFixture>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> Response<Body> {
    let Some(bake) = &state.bake else {
        return Response::builder().status(404).body(Body::empty()).unwrap();
    };
    let mut state = bake.state.lock().unwrap();
    if headers.get("authorization").and_then(|v| v.to_str().ok())
        != state
            .token
            .as_ref()
            .map(|v| format!("Bearer {v}"))
            .as_deref()
    {
        return Response::builder().status(401).body(Body::empty()).unwrap();
    }
    state.events.push(format!("export-chunk:{name}"));
    let Some(chunk) = state
        .manifest
        .as_ref()
        .unwrap()
        .chunks
        .iter()
        .find(|c| c.name == name)
    else {
        return Response::builder().status(404).body(Body::empty()).unwrap();
    };
    Response::builder()
        .header("content-length", chunk.size)
        .header("x-artifact-sha256", chunk.sha256.clone())
        .body(Body::from(state.chunks[&name].clone()))
        .unwrap()
}

type Fixture = (
    LiveRuntime<MockProvider>,
    MockProvider,
    PathBuf,
    tokio::task::JoinHandle<()>,
);

async fn fixture(bad_usage: bool, unknown: bool) -> Fixture {
    fixture_with_scheduling(bad_usage, unknown, None).await
}

async fn fixture_with_scheduling(
    bad_usage: bool,
    unknown: bool,
    scheduling: Option<Scheduling>,
) -> Fixture {
    fixture_with_cache_export(bad_usage, unknown, scheduling, None).await
}

async fn fixture_with_cache_export(
    bad_usage: bool,
    unknown: bool,
    scheduling: Option<Scheduling>,
    bake: Option<MockBake>,
) -> Fixture {
    fixture_internal(bad_usage, unknown, scheduling, bake, false).await
}
async fn fixture_internal(
    bad_usage: bool,
    unknown: bool,
    scheduling: Option<Scheduling>,
    bake: Option<MockBake>,
    registry: bool,
) -> Fixture {
    let out = std::env::temp_dir().join(format!(
        "switchyard-live-test-{}",
        switchyard_live::runtime::nonce().unwrap()
    ));
    let revision = "a".repeat(40);
    let dir = out.join("tokenizers").join(&revision);
    std::fs::create_dir_all(&dir).unwrap();
    let tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("hello".into(), 1),
                    ("world".into(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
    std::fs::write(dir.join("chat_template.jinja"), "{{ messages[0].content }}").unwrap();
    let chunk_hash = "b".repeat(64);
    let archive_hash = format!(
        "{:x}",
        <sha2::Sha256 as sha2::Digest>::digest(chunk_hash.as_bytes())
    );
    let manifest = json!({"format":"shards-v1","compress":"none","total_size":1,"raw_size":1,"sha256":archive_hash,
        "chunks":[{"name":"chunk","size":1,"raw":1,"shard":0,"seq":0,"sha256":chunk_hash}],"files":{"config.json":{"size":1}}});
    let agent_failure = Arc::new(Mutex::new(None));
    let state_delay_ms = Arc::new(AtomicU64::new(0));
    let state_reads = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(
            "/bucket/{*key}",
            get(object_get).put(object_put).delete(object_delete),
        )
        .route(
            "/state",
            get(|State(state): State<HttpFixture>| async move {
                state.state_reads.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(std::time::Duration::from_millis(
                    state.state_delay_ms.load(Ordering::Relaxed),
                ))
                .await;
                let error = state.agent_failure.lock().unwrap().clone();
                Json(switchyard_agent::State {
                    model_allocation: Default::default(),
                    model_diagnostics: None,
                    weight_loading: None,
                    model_transport: None,
                    artifact_bytes_downloaded: 0,
                    artifact_network_bytes_downloaded: 0,
                    artifact_bytes_total: 0,
                    artifact_volume_bytes_per_s: None,
                    artifact_network_bytes_per_s: None,
                    hydration_mode: Default::default(),
                    model_storage: None,
                    runtime_storage: None,
                    dependency_budget: None,
                    startup_milestones: vec![],
                    artifacts: vec![],
                    inference_verified: true,
                    model: "tiny".into(),
                    revision: "a".repeat(40),
                    phase: if error.is_some() {
                        switchyard_agent::Phase::Failed
                    } else {
                        switchyard_agent::Phase::Ready
                    },
                    bytes_downloaded: 1,
                    bytes_total: 1,
                    hydration_seconds: 1.0,
                    volume_bytes_per_s: Some(100_000_000.0),
                    model_cached: false,
                    image_cached: None,
                    error,
                })
            }),
        )
        .route(
            "/tokenize",
            axum::routing::post(|| async { Json(json!({"count":1})) }),
        )
        .route("/v1/chat/completions", axum::routing::post(completion))
        .route("/cache/export", get(export_observe).post(export_start))
        .route("/cache/export/chunks/{name}", get(export_chunk))
        .with_state(HttpFixture {
            bad_usage,
            model_manifest: manifest,
            bake: bake.clone(),
            agent_failure: agent_failure.clone(),
            state_delay_ms: state_delay_ms.clone(),
            state_reads: state_reads.clone(),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut config:DeploymentConfig=serde_json::from_value(json!({
        "compute":{"api_url":url,"key_env":"MOCK_KEY","cloud":"SECURE","gpu_types":["mock"],"data_centers":[],"min_cuda":"12.9",
            "disk_gb":80,"disk_usd_per_gb_month":0.1,"billing_hours_per_month":730.0},
        "bootstrap":{"image":"fixture","agent_url":"https://example.com/agent","agent_sha256":"a".repeat(64),"ticket_prefix":"tickets"},
        "storage":{"endpoint":url,"region":"auto","bucket":"bucket","access_key_env":"MOCK_AK","secret_key_env":"MOCK_SK","url_lifetime_s":3600},
        "models":[{"id":"tiny","repository":"fixture/tiny","revision":revision,"archive_prefix":"models/tiny","archive_sha256":archive_hash,"engine":{"max_model_len":100,"max_num_seqs":8,"gpu_memory_utilization":0.8,"dtype":"bfloat16","language_model_only":true,"extra_args":[]},
            "prior":{"single_output_tps":100.0,"aggregate_output_tps":1000.0,"input_work_per_token":0.0,"startup_s":1.0,"kv_tokens":1000}}],
        "limits":{"max_instances":2,"max_usd_per_hour":3.0,"max_run_usd":1.0,"run_timeout_s":20,"startup_timeout_s":10,"request_timeout_s":5,"hydration_concurrency":2,"hydration_stall_s":2,"poll_ms":100}
    })).unwrap();
    if let Some(scheduling) = scheduling {
        config.scheduling = scheduling;
    }
    if let Some(bake) = &bake {
        use std::os::unix::fs::PermissionsExt;
        use switchyard_artifacts::{Metadata, PackOptions, pack};
        let source = out.join("runtime-source");
        std::fs::create_dir_all(source.join("bin")).unwrap();
        std::fs::write(source.join("bin/serve"), b"#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(
            source.join("bin/serve"),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        if registry {
            config.models[0].engine = serde_json::from_value(json!({"kind":"native","max_model_len":100,"max_num_seqs":8,"gpu_memory_utilization":0.0,"dtype":"f32","language_model_only":true,"extra_args":[],"native":{"max_weight_bytes":1000,"minimum_gpu_memory_bytes":1000,"queue_capacity":8,"batch_wait_ms":0,"cuda_ordinal":0}})).unwrap();
            config.bootstrap.native_cache_registry = Some(serde_json::from_value(json!({"name":"gpu-kernels","registry_prefix":"registry/v1","artifact_prefix":"learned/v1","compatibility":{"native_driver_cache_layout":"native-cuda-driver-family-v1"},"publication_timeout_s":1})).unwrap());
        }
        let mut runtime_labels =
            BTreeMap::from([("architecture".into(), std::env::consts::ARCH.into())]);
        if registry {
            runtime_labels.insert("engine_sha256".into(), "c".repeat(64));
        }
        let runtime_manifest = pack(
            &source,
            &out.join("runtime-packed"),
            Metadata {
                kind: "runtime".into(),
                compatibility: runtime_labels,
                attributes: BTreeMap::new(),
            },
            PackOptions::default(),
        )
        .unwrap();
        config.bootstrap.runtime = Some(switchyard_live::artifacts::ServingRuntimeConfig {
            components: vec![],
            hydration: None,
            artifact: switchyard_live::artifacts::ArtifactRef {
                prefix: "runtimes/fixture".into(),
                id: runtime_manifest.id.clone(),
            },
            compatibility: runtime_manifest.metadata.compatibility.clone(),
            program: "bin/serve".into(),
            args: vec!["--fixture".into()],
            environment: if registry {
                BTreeMap::from([("NATIVE_ENGINE_CACHE_POLICY".into(), "opportunistic".into())])
            } else {
                BTreeMap::from([("VLLM_FIXTURE".into(), "1".into())])
            },
        });
        config.limits.max_instances = 1;
        config.limits.run_timeout_s = 60;
        let compatibility = if registry {
            config
                .bootstrap
                .native_cache_registry
                .as_ref()
                .unwrap()
                .binding(&config.execution(), &config.models[0], "mock")
                .unwrap()
                .compatibility
        } else {
            switchyard_live::artifacts::cache_requirements(
                &bake_compatibility(),
                "mock",
                config.bootstrap.runtime.as_ref().unwrap(),
                &config.models[0],
            )
            .unwrap()
        };
        let source = out.join("cache-source");
        std::fs::create_dir_all(source.join("torch")).unwrap();
        std::fs::write(
            source.join("torch/kernel.cubin"),
            b"prepared fixture kernel",
        )
        .unwrap();
        std::fs::write(source.join("torch/graph.json"), b"{\"captured\":true}").unwrap();
        let manifest = pack(
            &source,
            &out.join("cache-packed"),
            Metadata {
                kind: "cache".into(),
                compatibility,
                attributes: BTreeMap::new(),
            },
            PackOptions {
                target_shard_bytes: 1,
                ..PackOptions::default()
            },
        )
        .unwrap();
        assert!(manifest.chunks.len() >= 2);
        let mut state = bake.state.lock().unwrap();
        state.objects.insert(
            "runtimes/fixture/manifest.json".into(),
            serde_json::to_vec(&runtime_manifest).unwrap(),
        );
        state.chunks = manifest
            .chunks
            .iter()
            .map(|chunk| {
                (
                    chunk.name.clone(),
                    std::fs::read(out.join("cache-packed").join(&chunk.name)).unwrap(),
                )
            })
            .collect();
        state.manifest = Some(manifest);
    }
    let provider = MockProvider {
        url,
        pods: Arc::new(Mutex::new(vec![])),
        unknown,
        bake,
        rejected_allocations: Arc::new(Mutex::new(VecDeque::new())),
        allocation_attempts: Arc::new(Mutex::new(vec![])),
        create_delay_ms: Arc::new(AtomicU64::new(0)),
        agent_failure,
        state_delay_ms,
        state_reads,
    };
    let storage = Storage::with_keys(&config.storage, "fake".into(), "fake".into()).unwrap();
    let runtime = LiveRuntime::new(
        provider.clone(),
        storage,
        config.execution(),
        Policy::Adaptive,
        &out,
    )
    .await
    .unwrap();
    (runtime, provider, out, server)
}
async fn completion(
    State(state): State<HttpFixture>,
    Json(request): Json<serde_json::Value>,
) -> Response<Body> {
    let delay = if let Some(bake) = &state.bake {
        let mut state = bake.state.lock().unwrap();
        state.requests.push(request);
        state.events.push("completion".into());
        state.completion_delay_ms
    } else {
        0
    };
    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
    let bad = state.bad_usage;
    let chunks = [
        json!({"choices":[{"delta":{"content":"world"},"finish_reason":null}]}),
        json!({"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":if bad {2}else {1},"completion_tokens":1}}),
    ];
    let mut bytes = chunks
        .iter()
        .map(|v| format!("data: {v}\n\n"))
        .collect::<String>();
    bytes.push_str("data: [DONE]\n\n");
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from(bytes))
        .unwrap()
}
fn calls(count: usize) -> Vec<WorkCall> {
    (0..count)
        .map(|_| WorkCall {
            model: "tiny".into(),
            messages: vec![Message {
                role: "user".into(),
                content: "hello".into(),
            }],
            planned_output: 1,
            max_output: 2,
            arrival_delay_s: 0.0,
            class: "standard".into(),
            depends_on: vec![],
        })
        .collect()
}

#[tokio::test]
async fn allocation_rejection_replans_without_losing_hot_capacity_or_launch_ownership() {
    struct RejectExpansion {
        attempted: bool,
        inner: switchyard_scheduler::BuiltinScheduler,
        cooldown_observed: Arc<Mutex<bool>>,
    }
    impl Scheduler for RejectExpansion {
        fn name(&self) -> &str {
            "rejected-expansion"
        }
        fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<switchyard_core::Action> {
            if !self.attempted {
                self.attempted = true;
                return vec![
                    switchyard_core::Action::Provision { profile: 0 },
                    switchyard_core::Action::Route {
                        request: 0,
                        node: fleet.nodes.len(),
                        service_weight: 1.0,
                    },
                ];
            }
            *self.cooldown_observed.lock().unwrap() |= fleet
                .provisioning_cooldowns
                .get(&0)
                .is_some_and(|until| *until > fleet.now);
            self.inner.decide(fleet)
        }
    }
    let scheduling = Scheduling {
        retry_cooldown_s: 0.5,
        plan_interval_s: 0.01,
        ..Scheduling::default()
    };
    let (mut runtime, provider, out, server) =
        fixture_with_scheduling(false, false, Some(scheduling)).await;
    provider.rejected_allocations.lock().unwrap().push_back(400);
    let cold = runtime
        .run_workload("rejected-cold", &calls(1), 200.0, 10.0)
        .await
        .unwrap();
    assert_eq!(cold.completed, 1);
    assert_eq!(cold.invalid_actions, 0);
    assert_eq!(provider.allocation_attempts.lock().unwrap().len(), 2);
    assert_eq!(provider.pods.lock().unwrap().len(), 1);

    provider.rejected_allocations.lock().unwrap().push_back(403);
    let cooldown_observed = Arc::new(Mutex::new(false));
    runtime = runtime.with_scheduler(RejectExpansion {
        attempted: false,
        inner: switchyard_scheduler::BuiltinScheduler::new(Policy::WarmOnly),
        cooldown_observed: cooldown_observed.clone(),
    });
    let hot = runtime
        .run_workload("rejected-expansion", &calls(32), 200.0, 10.0)
        .await
        .unwrap();
    assert_eq!(hot.completed, 32);
    assert_eq!(hot.output_tokens, 32);
    assert_eq!(
        hot.invalid_actions, 0,
        "discard routes to an unallocated node"
    );
    assert!(*cooldown_observed.lock().unwrap());
    {
        let attempts = provider.allocation_attempts.lock().unwrap();
        assert_eq!(attempts.len(), 3);
        assert_eq!(
            attempts
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
    }
    let rejected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("allocation-rejections.json")).unwrap())
            .unwrap();
    assert_eq!(rejected[0]["status"], 400);
    assert_eq!(rejected[1]["status"], 403);
    let pending: Vec<switchyard_live::runtime::PendingLaunch> =
        serde_json::from_slice(&std::fs::read(out.join("pending-launches.json")).unwrap()).unwrap();
    assert!(pending.iter().all(|launch| launch.resolved));
    runtime.cleanup().await.unwrap();
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|pod| pod.status.is_terminated())
    );
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn live_controller_receives_configured_scheduling_and_custom_service_classes() {
    struct Observer {
        inner: switchyard_scheduler::BuiltinScheduler,
        observations: Arc<Mutex<Vec<serde_json::Value>>>,
    }
    impl Scheduler for Observer {
        fn name(&self) -> &str {
            "live-scheduling-observer"
        }
        fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<switchyard_core::Action> {
            self.observations
                .lock()
                .unwrap()
                .push(serde_json::to_value(fleet.scheduling).unwrap());
            self.inner.decide(fleet)
        }
    }
    let mut scheduling = Scheduling {
        maximum_startup_s: 300.0,
        minimum_hydration_bytes_per_s: 5_000_000.0,
        retry_cooldown_s: 17.0,
        plan_interval_s: 0.25,
        ..Scheduling::default()
    };
    let mut custom = scheduling.classes["standard"].clone();
    custom.weight = 7.0;
    scheduling.classes.insert("custom".into(), custom);
    let expected = serde_json::to_value(&scheduling).unwrap();
    let (runtime, _, out, server) = fixture_with_scheduling(false, false, Some(scheduling)).await;
    let observations = Arc::new(Mutex::new(vec![]));
    let mut runtime = runtime.with_scheduler(Observer {
        inner: switchyard_scheduler::BuiltinScheduler::new(Policy::Adaptive),
        observations: observations.clone(),
    });
    let mut workload = calls(1);
    workload[0].class = "custom".into();
    let report = runtime
        .run_workload("configured-scheduling", &workload, 200.0, 10.0)
        .await
        .unwrap();
    assert_eq!(report.completed, 1);
    assert_eq!(report.invalid_actions, 0);
    {
        let seen = observations.lock().unwrap();
        assert!(!seen.is_empty());
        assert!(seen.iter().all(|value| *value == expected));
    }
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("configuration.json")).unwrap()).unwrap();
    assert_eq!(receipt["scheduling"], expected);
    runtime.cleanup().await.unwrap();
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn single_call_does_not_duplicate_cold_capacity_and_hot_burst_conserves_usage() {
    let (mut runtime, provider, out, server) = fixture(false, false).await;
    let cold = runtime
        .run_workload("cold", &calls(1), 200.0, 10.0)
        .await
        .unwrap();
    assert_eq!(cold.launches, 1);
    assert_eq!(cold.completed, 1);
    assert_eq!(cold.invalid_actions, 0);
    // Evidence is readable while the runtime is still live, before cleanup
    // writes its aggregate summaries. Interrupted runs retain these records.
    let decisions = std::fs::read_to_string(out.join("scheduler-decisions.ndjson")).unwrap();
    let decisions = decisions
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert!(decisions.iter().any(|record| {
        record["actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|action| action["kind"] == "provision")
    }));
    assert!(decisions.iter().any(|record| {
        record["actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|action| action["kind"] == "route")
    }));
    let observations = std::fs::read_to_string(out.join("agent-observations.ndjson")).unwrap();
    assert!(observations.lines().all(|line| {
        let observation = serde_json::from_str::<serde_json::Value>(line).unwrap();
        observation["state"].is_object()
            && observation["start_s"].as_f64().unwrap() <= observation["finish_s"].as_f64().unwrap()
            && observation["finish_s"] == observation["at_s"]
    }));
    assert!(!out.join("agent-observations.json").exists());
    runtime.set_policy(Policy::WarmOnly);
    let burst = runtime
        .run_workload("burst", &calls(32), 200.0, 10.0)
        .await
        .unwrap();
    assert_eq!(burst.output_tokens, 32);
    assert_eq!(burst.completed, 32);
    assert_eq!(burst.launches, 0);
    assert!(
        burst
            .completions
            .iter()
            .all(|c| c.first_content_s >= c.started_s && c.completed_s >= c.first_content_s)
    );
    runtime.cleanup().await.unwrap();
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|p| p.status.is_terminated())
    );
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}
#[tokio::test]
async fn host_failure_preserves_observed_completions_and_pending_calls() {
    struct FailAfterCompletion {
        inner: switchyard_scheduler::BuiltinScheduler,
        agent_failure: Arc<Mutex<Option<String>>>,
    }
    impl Scheduler for FailAfterCompletion {
        fn name(&self) -> &str {
            "host-failure-fixture"
        }
        fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<switchyard_core::Action> {
            if fleet
                .states
                .values()
                .any(|state| state.state == switchyard_core::scheduler::RequestStatus::Done)
            {
                *self.agent_failure.lock().unwrap() = Some("fixture cache rejected".into());
            }
            self.inner.decide(fleet)
        }
    }
    let (runtime, provider, out, server) = fixture(false, false).await;
    let mut runtime = runtime.with_scheduler(FailAfterCompletion {
        inner: switchyard_scheduler::BuiltinScheduler::new(Policy::Adaptive),
        agent_failure: provider.agent_failure.clone(),
    });
    let mut workload = calls(2);
    workload[1].arrival_delay_s = 5.0;
    let result = runtime
        .run_workload("host-failure", &workload, 200.0, 10.0)
        .await;
    assert!(
        matches!(result, Err(switchyard_live::Error::Request(error)) if error == "fixture cache rejected")
    );
    let partial: serde_json::Value = serde_json::from_slice(
        &std::fs::read(out.join("workloads/host-failure-partial.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(partial["requests"], 2);
    assert_eq!(partial["completed"], 1);
    assert_eq!(partial["completions"][0]["request"], 0);
    assert_eq!(partial["completions"][0]["output_tokens"], 1);
    assert_eq!(partial["states"]["0"], "Done");
    assert_eq!(partial["states"]["1"], "Future");
    assert!(partial["estimated_rental_usd"].as_f64().unwrap() > 0.0);
    assert!(!out.join("workloads/host-failure.json").exists());
    runtime.cleanup().await.unwrap();
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|pod| pod.status.is_terminated())
    );
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn incorrect_usage_is_rejected_and_owned_resources_are_released() {
    let (mut runtime, provider, out, server) = fixture(true, false).await;
    let report = runtime
        .run_workload("bad-usage", &calls(1), 200.0, 10.0)
        .await
        .unwrap();
    assert_eq!(report.failed, 1);
    assert_eq!(report.output_tokens, 0);
    assert_eq!(report.completed, 0);
    runtime.cleanup().await.unwrap();
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|p| p.status.is_terminated())
    );
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}
#[tokio::test]
async fn uncertain_allocation_is_journaled_before_creation_and_cleaned_by_exact_name() {
    let (mut runtime, provider, out, server) = fixture(false, true).await;
    assert!(
        runtime
            .run_workload("unknown", &calls(1), 200.0, 10.0)
            .await
            .is_err()
    );
    assert!(out.join("pending-launches.json").exists());
    runtime.cleanup().await.unwrap();
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|p| p.status.is_terminated())
    );
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[test]
fn checked_in_fleet_and_workloads_deserialize_and_validate() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config: DeploymentConfig = serde_json::from_slice(
        &std::fs::read(root.join("configs/runpod/qwen-small.json")).unwrap(),
    )
    .unwrap();
    config.validate().unwrap();
    let cases: switchyard_live::cli::Experiment =
        serde_json::from_slice(&std::fs::read(root.join("configs/runpod/gauntlet.json")).unwrap())
            .unwrap();
    assert!(cases.cases.iter().any(|c| c.calls == 1000));
    assert_eq!(config.models.len(), 3);
    assert!(config.scheduling.maximum_startup_s < config.limits.startup_timeout_s as f64);
    assert_eq!(config.scheduling.minimum_hydration_bytes_per_s, 5_000_000.0);
    assert_eq!(config.scheduling.retry_cooldown_s, 60.0);
}

#[test]
fn legacy_ownership_journals_recover_with_normalized_lifecycle_and_field_names() {
    let quote = Offering {
        gpu: "fixture".into(),
        gpu_memory_gb: 48,
        data_center: "local".into(),
        availability: "HIGH".into(),
        gpu_usd_per_hour: 1.0,
        disk_usd_per_hour: 0.01,
        quoted_at_unix_s: 0,
    };
    for (legacy, expected) in [
        ("PROVISIONING", PodStatus::Starting),
        ("STARTING", PodStatus::Starting),
        ("RUNNING", PodStatus::Running),
        ("EXITED", PodStatus::Stopped),
        ("ERROR", PodStatus::Failed),
        ("TERMINATED", PodStatus::Terminated),
        ("DELETED", PodStatus::Terminated),
        ("UNRECOGNIZED", PodStatus::Unknown),
    ] {
        let journal = json!([{
            "pod": {"id":"legacy123","name":"switchyard-legacy-owned","status":legacy,"cost":1.0,"dataCenterId":"local"},
            "allocated_s":0.0,"ended_s":null,"profile":0,"quoted":quote,
        }]);
        let recovered: Vec<switchyard_live::runtime::OwnedPod> =
            serde_json::from_value(journal).unwrap();
        assert_eq!(recovered[0].pod.status, expected);
        assert_eq!(recovered[0].pod.data_center.as_deref(), Some("local"));
        let serialized = serde_json::to_value(recovered).unwrap();
        assert_eq!(serialized[0]["pod"]["data_center"], "local");
        assert!(serialized[0]["pod"].get("dataCenterId").is_none());
        assert_eq!(
            serialized[0]["pod"]["status"],
            serde_json::to_value(expected).unwrap()
        );
    }
}

struct AllocateTwo(bool);
impl switchyard_core::scheduler::Scheduler for AllocateTwo {
    fn name(&self) -> &str {
        "cleanup-fixture"
    }
    fn decide(
        &mut self,
        _: &switchyard_core::scheduler::FleetView<'_>,
    ) -> Vec<switchyard_core::scheduler::Action> {
        if self.0 {
            return vec![];
        }
        self.0 = true;
        vec![switchyard_core::scheduler::Action::Provision { profile: 0 }; 2]
    }
}

#[tokio::test]
async fn receipt_filesystem_failure_cannot_interrupt_gpu_termination() {
    let (runtime, provider, out, server) = fixture(false, false).await;
    let mut runtime = runtime.with_scheduler(AllocateTwo(false));
    let report = runtime
        .run_workload("cutoff", &calls(1), 200.0, 0.5)
        .await
        .unwrap();
    assert!(report.budget_exceeded);
    assert_eq!(provider.pods.lock().unwrap().len(), 2);
    std::fs::remove_dir_all(&out).unwrap();
    std::fs::write(&out, b"simulate broken receipt directory").unwrap();
    assert!(runtime.cleanup().await.is_err());
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|p| p.status.is_terminated())
    );
    server.abort();
    std::fs::remove_file(out).unwrap();
}

#[tokio::test]
async fn expired_workload_does_not_leave_stale_admission_reservations() {
    let (mut runtime, provider, out, server) = fixture(false, false).await;
    runtime
        .run_workload("warmup", &calls(1), 200.0, 10.0)
        .await
        .unwrap();
    runtime.set_policy(Policy::WarmOnly);
    let cutoff = runtime
        .run_workload("cutoff", &calls(32), 200.0, 0.15)
        .await
        .unwrap();
    assert!(cutoff.budget_exceeded);
    let next = runtime
        .run_workload("next", &calls(1), 200.0, 10.0)
        .await
        .unwrap();
    assert_eq!(next.completed, 1);
    assert_eq!(next.invalid_actions, 0);
    runtime.cleanup().await.unwrap();
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|p| p.status.is_terminated())
    );
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

fn bake_compatibility() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("architecture".into(), std::env::consts::ARCH.into()),
        ("gpu_compute_capability".into(), "8.9".into()),
    ])
}
fn bake_plan(arrival_interval_s: f64) -> switchyard_live::bake::CacheBakePlan {
    switchyard_live::bake::CacheBakePlan {
        model: "tiny".into(),
        gpu: "mock".into(),
        name: "compiled".into(),
        compatibility: bake_compatibility(),
        warmup: switchyard_live::workloads::Experiment {
            cases: vec![switchyard_live::workloads::Case {
                name: "warm-cache".into(),
                policy: Policy::WarmOnly,
                calls: 2,
                prompt: "hello".into(),
                prompt_variants: vec![],
                planned_output: 1,
                max_output: 2,
                arrival_interval_s,
                desired_output_tps: 200.0,
                completion_budget_s: 45.0,
                serial_dependencies: false,
                system_prompt: None,
                unique_prefix: false,
                output_bounds: vec![],
            }],
        },
        export_timeout_s: 5,
        max_artifact_bytes: 1 << 20,
        download_concurrency: 2,
    }
}
fn assert_bake_cleanup(provider: &MockProvider, bake: &MockBake, out: &std::path::Path) {
    let pods = provider.pods.lock().unwrap();
    assert_eq!(pods.len(), 1, "cache preparation owns exactly one host");
    assert!(pods[0].status.is_terminated());
    let owned: Vec<switchyard_live::runtime::OwnedPod> =
        serde_json::from_slice(&std::fs::read(out.join("owned-pods.json")).unwrap()).unwrap();
    assert_eq!(owned.len(), 1);
    assert!(owned[0].ended_s.is_some());
    let state = bake.state.lock().unwrap();
    assert!(!state.deleted.is_empty());
    assert!(state.deleted.iter().all(|key| key.starts_with("tickets/")));
    assert!(state.objects.keys().all(|key| !key.starts_with("tickets/")));
    let terminated = state
        .events
        .iter()
        .position(|event| event == "terminate")
        .unwrap();
    let deleted = state
        .events
        .iter()
        .position(|event| event == "ticket-delete")
        .unwrap();
    assert!(
        terminated < deleted,
        "termination is confirmed before ticket removal"
    );
}

#[tokio::test]
async fn cache_bake_keeps_one_host_across_idle_gap_and_exports_exact_compatible_artifact() {
    use sha2::{Digest, Sha256};
    let bake = MockBake::default();
    let (mut runtime, provider, out, server) =
        fixture_with_cache_export(false, false, None, Some(bake.clone())).await;
    // The dedicated preparation host must survive the router's 30-second idle
    // timeout while waiting for the next announced warmup request.
    let plan = bake_plan(31.0);
    let prepared = runtime.bake_cache(&plan, "caches/fixture").await.unwrap();
    let manifest: switchyard_artifacts::Manifest =
        serde_json::from_slice(&std::fs::read(prepared.directory.join("manifest.json")).unwrap())
            .unwrap();
    manifest.validate().unwrap();
    assert_eq!(manifest.id, prepared.manifest.id);
    assert_eq!(manifest.metadata.kind, "cache");
    assert_eq!(prepared.configuration.name, "compiled");
    assert_eq!(prepared.configuration.gpu, "mock");
    assert_eq!(prepared.configuration.artifact.id, manifest.id);
    assert_eq!(
        prepared.configuration.artifact.prefix,
        format!("caches/fixture/{}", manifest.id)
    );
    assert_eq!(
        prepared.configuration.compatibility,
        manifest.metadata.compatibility
    );
    prepared.configuration.validate().unwrap();
    let config: switchyard_live::config::ExecutionConfig =
        serde_json::from_slice(&std::fs::read(out.join("configuration.json")).unwrap()).unwrap();
    let serving = config.bootstrap.runtime.as_ref().unwrap();
    let model = &config.models[0];
    assert_eq!(manifest.metadata.compatibility, switchyard_live::artifacts::cache_requirements(
        &plan.compatibility, "mock", serving, model,
    ).unwrap());
    assert_eq!(
        manifest.metadata.compatibility["runtime_id"],
        serving.artifact.id
    );
    assert_eq!(
        manifest.metadata.compatibility["model_revision"],
        "a".repeat(40)
    );
    assert_eq!(
        manifest.metadata.compatibility["gpu_compute_capability"],
        "8.9"
    );
    for (path, expected) in [
        ("torch/kernel.cubin", b"prepared fixture kernel".as_slice()),
        ("torch/graph.json", b"{\"captured\":true}".as_slice()),
    ] {
        let Some(switchyard_artifacts::Entry::File { size, sha256, .. }) =
            manifest.resolve_entry(path).unwrap()
        else {
            panic!("exported cache file is absent");
        };
        assert_eq!(*size, expected.len() as u64);
        assert_eq!(*sha256, format!("{:x}", Sha256::digest(expected)));
    }
    {
        let state = bake.state.lock().unwrap();
        assert_eq!(state.requests.len(), 2);
        assert!(
            state
                .requests
                .iter()
                .all(|request| request["model"] == "tiny" && request["max_tokens"] == 2)
        );
        assert_eq!(state.tickets.len(), 1);
        let export = state.tickets[0]
            .runtime
            .as_ref()
            .unwrap()
            .cache_export
            .as_ref()
            .unwrap();
        assert_eq!(export.name, plan.name);
        assert_eq!(export.compatibility, manifest.metadata.compatibility);
        let export_start = state
            .events
            .iter()
            .position(|event| event == "export-start")
            .unwrap();
        assert_eq!(
            state.events[..export_start]
                .iter()
                .filter(|event| *event == "completion")
                .count(),
            2
        );
        assert!(state.events.iter().any(|event| event == "export-status"));
        for chunk in &manifest.chunks {
            let bytes = std::fs::read(prepared.directory.join(&chunk.name)).unwrap();
            assert_eq!(bytes, state.chunks[&chunk.name]);
            assert_eq!(bytes.len() as u64, chunk.size);
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), chunk.sha256);
            assert!(
                state
                    .events
                    .contains(&format!("export-chunk:{}", chunk.name))
            );
        }
    }
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("workloads/warm-cache.json")).unwrap())
            .unwrap();
    assert_eq!(report["policy"], "warm_only");
    assert_eq!(report["completed"], 2);
    assert_eq!(report["output_tokens"], 2);
    assert_eq!(report["failed"], 0);
    assert_eq!(report["launches"], 0);
    assert_eq!(report["releases"], 0);
    assert!(report["elapsed_s"].as_f64().unwrap() >= 31.0);
    // Preparation leaves cleanup to its caller; a returned artifact is durable
    // local input for publication only after that owned host is released.
    assert_eq!(provider.pods.lock().unwrap()[0].status, PodStatus::Running);
    runtime.cleanup().await.unwrap();
    assert_bake_cleanup(&provider, &bake, &out);
    assert!(prepared.directory.join("manifest.json").is_file());
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn cache_bake_export_failure_retains_receipt_and_cleans_host_without_publishing_manifest() {
    let bake = MockBake {
        fail_export: true,
        ..MockBake::default()
    };
    let (mut runtime, provider, out, server) =
        fixture_with_cache_export(false, false, None, Some(bake.clone())).await;
    let result = runtime.bake_cache(&bake_plan(0.0), "caches/fixture").await;
    runtime.cleanup().await.unwrap();
    assert!(
        matches!(result, Err(switchyard_live::Error::Request(message)) if message.contains("cache export failed"))
    );
    assert_bake_cleanup(&provider, &bake, &out);
    let status: switchyard_agent::export::ExportState =
        serde_json::from_slice(&std::fs::read(out.join("cache-export-status.json")).unwrap())
            .unwrap();
    assert_eq!(status.phase, switchyard_agent::export::ExportPhase::Failed);
    assert!(status.manifest.is_none());
    assert!(status.error.is_some());
    assert!(!out.join("cache-artifact").exists());
    {
        let state = bake.state.lock().unwrap();
        assert_eq!(state.requests.len(), 2);
        assert!(
            !state
                .events
                .iter()
                .any(|event| event.starts_with("export-chunk:"))
        );
    }
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn cache_bake_rejects_oversized_warmup_before_allocating_a_host() {
    let bake = MockBake::default();
    let (mut runtime, provider, out, server) =
        fixture_with_cache_export(false, false, None, Some(bake.clone())).await;
    let mut plan = bake_plan(0.0);
    plan.warmup.cases[0].max_output = 101; // Fixture model context is 100 tokens.
    let result = runtime.bake_cache(&plan, "caches/fixture").await;
    assert!(matches!(result, Err(switchyard_live::Error::Config(_))));
    assert!(provider.pods.lock().unwrap().is_empty());
    {
        let state = bake.state.lock().unwrap();
        assert!(state.tickets.is_empty());
        assert!(state.requests.is_empty());
        assert!(state.events.is_empty());
    }
    assert!(!out.join("cache-artifact").exists());
    runtime.cleanup().await.unwrap();
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn cache_bake_rejects_missing_standard_class_before_any_paid_allocation() {
    let mut scheduling = Scheduling::default();
    assert!(scheduling.classes.remove("standard").is_some());
    let bake = MockBake::default();
    let (mut runtime, provider, out, server) =
        fixture_with_cache_export(false, false, Some(scheduling), Some(bake.clone())).await;
    let result = runtime.bake_cache(&bake_plan(0.0), "caches/fixture").await;
    assert!(
        matches!(result, Err(switchyard_live::Error::Config(message)) if message.contains("standard scheduling class"))
    );
    assert!(provider.pods.lock().unwrap().is_empty());
    {
        let state = bake.state.lock().unwrap();
        assert!(state.tickets.is_empty());
        assert!(state.requests.is_empty());
        assert!(state.events.is_empty());
    }
    assert!(!out.join("cache-bake-plan.json").exists());
    assert!(!out.join("cache-artifact").exists());
    runtime.cleanup().await.unwrap();
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn native_registry_learns_real_work_after_cleanup_and_reuses_next_launch() {
    let bake = MockBake::default();
    let (mut runtime, provider, out, server) =
        fixture_internal(false, false, None, Some(bake.clone()), true).await;
    let report = runtime
        .run_workload("real-client", &calls(1), 1.0, 5.0)
        .await
        .unwrap();
    assert_eq!(report.completed, 1);
    {
        let state = bake.state.lock().unwrap();
        let runtime = state.tickets[0].runtime.as_ref().unwrap();
        assert!(runtime.caches.is_empty());
        assert_eq!(runtime.cache_export.as_ref().unwrap().name, "gpu-kernels");
        assert!(!state.events.iter().any(|event| event == "export-start"));
    }
    runtime.cleanup().await.unwrap();
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|pod| pod.status.is_terminated())
    );
    {
        let state = bake.state.lock().unwrap();
        let terminate = state
            .events
            .iter()
            .position(|event| event == "terminate")
            .unwrap();
        let publish = state
            .events
            .iter()
            .position(|event| event == "artifact-put")
            .unwrap();
        let record = state
            .events
            .iter()
            .position(|event| event == "registry-put")
            .unwrap();
        assert!(terminate < publish && publish < record);
        assert_eq!(state.requests.len(), 1); // No synthetic warmup or readiness inference.
    }
    assert!(
        std::fs::read_to_string(out.join("native-cache-registry.ndjson"))
            .unwrap()
            .contains("\"phase\":\"miss\"")
    );
    let (mut next, next_provider, next_out, next_server) =
        fixture_internal(false, false, None, Some(bake.clone()), true).await;
    assert_eq!(
        next.run_workload("next-real-client", &calls(1), 1.0, 5.0)
            .await
            .unwrap()
            .completed,
        1
    );
    {
        let state = bake.state.lock().unwrap();
        let caches = &state.tickets[1].runtime.as_ref().unwrap().caches;
        assert_eq!(caches.len(), 1);
        assert_eq!(
            caches[0].artifact.manifest.id,
            state.manifest.as_ref().unwrap().id
        );
    }
    next.cleanup().await.unwrap();
    assert!(
        next_provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|pod| pod.status.is_terminated())
    );
    assert!(
        std::fs::read_to_string(next_out.join("native-cache-registry.ndjson"))
            .unwrap()
            .contains("\"phase\":\"hit\"")
    );
    server.abort();
    next_server.abort();
    std::fs::remove_dir_all(out).unwrap();
    std::fs::remove_dir_all(next_out).unwrap();
}

#[tokio::test]
async fn native_registry_collection_and_publication_failure_do_not_leak_owned_pods() {
    for fail_export in [true, false] {
        let bake = MockBake {
            fail_export,
            fail_publication: !fail_export,
            ..Default::default()
        };
        let (mut runtime, provider, out, server) =
            fixture_internal(false, false, None, Some(bake.clone()), true).await;
        assert_eq!(
            runtime
                .run_workload("real-client", &calls(1), 1.0, 5.0)
                .await
                .unwrap()
                .completed,
            1
        );
        runtime.cleanup().await.unwrap();
        assert!(
            provider
                .pods
                .lock()
                .unwrap()
                .iter()
                .all(|pod| pod.status.is_terminated())
        );
        let telemetry = std::fs::read_to_string(out.join("native-cache-registry.ndjson")).unwrap();
        assert!(telemetry.contains(if fail_export {
            "collection_failed"
        } else {
            "publication_"
        }));
        assert!(
            !bake
                .state
                .lock()
                .unwrap()
                .objects
                .keys()
                .any(|key| key.starts_with("registry/"))
        );
        server.abort();
        std::fs::remove_dir_all(out).unwrap();
    }
}

#[tokio::test]
async fn native_registry_bad_binding_is_a_visible_cold_jit_miss() {
    use switchyard_live::native_cache_registry::CacheRegistryEntry;
    let bake = MockBake::default();
    let (mut runtime, provider, out, server) =
        fixture_internal(false, false, None, Some(bake.clone()), true).await;
    let config: switchyard_live::config::ExecutionConfig =
        serde_json::from_slice(&std::fs::read(out.join("configuration.json")).unwrap()).unwrap();
    let registry = config.bootstrap.native_cache_registry.as_ref().unwrap();
    let mut binding = registry
        .binding(&config, &config.models[0], "mock")
        .unwrap();
    let key = format!("{}/{}.json", registry.registry_prefix, binding.identity);
    binding
        .compatibility
        .insert("gpu".into(), "different-gpu".into());
    {
        let mut state = bake.state.lock().unwrap();
        let id = state.manifest.as_ref().unwrap().id.clone();
        state.objects.insert(
            key,
            serde_json::to_vec(&CacheRegistryEntry {
                version: 1,
                binding,
                artifact: switchyard_live::artifacts::ArtifactRef {
                    id: id.clone(),
                    prefix: format!("learned/v1/{id}"),
                },
            })
            .unwrap(),
        );
    }
    assert_eq!(
        runtime
            .run_workload("real-client", &calls(1), 1.0, 5.0)
            .await
            .unwrap()
            .completed,
        1
    );
    assert!(
        bake.state.lock().unwrap().tickets[0]
            .runtime
            .as_ref()
            .unwrap()
            .caches
            .is_empty()
    );
    assert!(
        std::fs::read_to_string(out.join("native-cache-registry.ndjson"))
            .unwrap()
            .contains("lookup_failed")
    );
    runtime.cleanup().await.unwrap();
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|pod| pod.status.is_terminated())
    );
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn native_registry_does_not_export_after_canceling_an_active_remote_request() {
    let bake = MockBake::default();
    let (mut runtime, provider, out, server) =
        fixture_internal(false, false, None, Some(bake.clone()), true).await;
    assert_eq!(
        runtime
            .run_workload("completed", &calls(1), 1.0, 5.0)
            .await
            .unwrap()
            .completed,
        1
    );
    bake.state.lock().unwrap().completion_delay_ms = 1000;
    let canceled = runtime
        .run_workload("canceled", &calls(1), 1.0, 0.05)
        .await
        .unwrap();
    assert!(canceled.budget_exceeded);
    runtime.cleanup().await.unwrap();
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|pod| pod.status.is_terminated())
    );
    assert!(
        !bake
            .state
            .lock()
            .unwrap()
            .events
            .iter()
            .any(|event| event == "export-start")
    );
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn slow_passive_observation_does_not_block_hot_dispatch_or_cleanup() {
    let (mut runtime, provider, out, server) = fixture(false, false).await;
    let warm = runtime
        .run_workload("warm", &calls(1), 200.0, 5.0)
        .await
        .unwrap();
    assert_eq!(warm.completed, 1);
    runtime.set_policy(Policy::WarmOnly);
    let before = provider.state_reads.load(Ordering::Relaxed);
    provider.state_delay_ms.store(3000, Ordering::Relaxed);
    let started = std::time::Instant::now();
    let hot = runtime
        .run_workload("slow-state-hot", &calls(32), 200.0, 1.5)
        .await
        .unwrap();
    assert_eq!(hot.completed, 32);
    assert!(!hot.budget_exceeded);
    assert!(started.elapsed().as_secs_f64() < 1.5);
    assert_eq!(
        provider.state_reads.load(Ordering::Relaxed),
        before + 1,
        "only one outstanding observation per host"
    );
    let cleanup = std::time::Instant::now();
    runtime.cleanup().await.unwrap();
    assert!(
        cleanup.elapsed().as_secs_f64() < 1.0,
        "cleanup cancels the outstanding HTTP read"
    );
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|pod| pod.status.is_terminated())
    );
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}

#[tokio::test]
async fn delayed_expansion_does_not_block_hot_dispatch_and_cleanup_drains_ownership() {
    struct Expand {
        once: bool,
        inner: switchyard_scheduler::BuiltinScheduler,
        pending_seen: Arc<AtomicUsize>,
    }
    impl Scheduler for Expand {
        fn name(&self) -> &str {
            "delayed-expansion"
        }
        fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<switchyard_core::Action> {
            self.pending_seen
                .fetch_max(fleet.pending_allocations.len(), Ordering::SeqCst);
            let mut actions = self.inner.decide(fleet);
            if !self.once {
                self.once = true;
                actions.insert(0, switchyard_core::Action::Provision { profile: 0 });
                actions.insert(0, switchyard_core::Action::Provision { profile: 0 });
            }
            actions
        }
    }
    let (mut runtime, provider, out, server) = fixture(false, false).await;
    runtime
        .run_workload("cold", &calls(1), 200.0, 10.0)
        .await
        .unwrap();
    provider.create_delay_ms.store(1500, Ordering::SeqCst);
    let pending_seen = Arc::new(AtomicUsize::new(0));
    runtime = runtime.with_scheduler(Expand {
        once: false,
        inner: switchyard_scheduler::BuiltinScheduler::new(Policy::WarmOnly),
        pending_seen: pending_seen.clone(),
    });
    let report = runtime
        .run_workload("hot-during-allocation", &calls(32), 200.0, 10.0)
        .await
        .unwrap();
    assert_eq!(report.completed, 32);
    assert_eq!(report.invalid_actions, 0);
    assert!(
        report.elapsed_s < 1.0,
        "hot work blocked by 1.5s allocation: {}",
        report.elapsed_s
    );
    assert_eq!(
        pending_seen.load(Ordering::SeqCst),
        1,
        "second pending launch must be rejected by max_instances"
    );
    runtime.cleanup().await.unwrap();
    assert_eq!(provider.allocation_attempts.lock().unwrap().len(), 2);
    assert!(
        provider
            .pods
            .lock()
            .unwrap()
            .iter()
            .all(|p| p.status.is_terminated())
    );
    let pending: Vec<switchyard_live::runtime::PendingLaunch> =
        serde_json::from_slice(&std::fs::read(out.join("pending-launches.json")).unwrap()).unwrap();
    assert_eq!(pending.len(), 2);
    assert!(pending.iter().all(|p| p.resolved));
    server.abort();
    std::fs::remove_dir_all(out).unwrap();
}
