//! Exercise the live scheduler at real HTTP boundaries without renting hardware.
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{Response, StatusCode},
    routing::get,
};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use switchyard_core::Policy;
use switchyard_live::config::DeploymentConfig;
use switchyard_live::{LiveRuntime, WorkCall, planner::Message};
use switchyard_provider::{Launch, Offering, Pod, PodStatus, Provider, storage::Storage};

#[derive(Clone)]
struct MockProvider {
    url: String,
    pods: Arc<Mutex<Vec<Pod>>>,
    unknown: bool,
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

async fn fixture(
    bad_usage: bool,
    unknown: bool,
) -> (
    LiveRuntime<MockProvider>,
    MockProvider,
    PathBuf,
    tokio::task::JoinHandle<()>,
) {
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
    let app = Router::new()
        .route(
            "/bucket/{*key}",
            get(move || {
                let m = manifest.clone();
                async move { Json(m) }
            })
            .put(|| async { StatusCode::NO_CONTENT })
            .delete(|| async { StatusCode::NO_CONTENT }),
        )
        .route(
            "/state",
            get(|| async {
                Json(switchyard_agent::State {
                    model: "tiny".into(),
                    revision: "a".repeat(40),
                    phase: switchyard_agent::Phase::Ready,
                    bytes_downloaded: 1,
                    bytes_total: 1,
                    hydration_seconds: 1.0,
                    volume_bytes_per_s: Some(100_000_000.0),
                    model_cached: false,
                    image_cached: None,
                    error: None,
                })
            }),
        )
        .route(
            "/tokenize",
            axum::routing::post(|| async { Json(json!({"count":1})) }),
        )
        .route("/v1/chat/completions", axum::routing::post(completion))
        .with_state(bad_usage);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let config:DeploymentConfig=serde_json::from_value(json!({
        "compute":{"api_url":url,"key_env":"MOCK_KEY","cloud":"SECURE","gpu_types":["mock"],"data_centers":[],"min_cuda":"12.9",
            "disk_gb":80,"disk_usd_per_gb_month":0.1,"billing_hours_per_month":730.0},
        "bootstrap":{"image":"fixture","agent_url":"https://example.com/agent","agent_sha256":"a".repeat(64),"ticket_prefix":"tickets"},
        "storage":{"endpoint":url,"region":"auto","bucket":"bucket","access_key_env":"MOCK_AK","secret_key_env":"MOCK_SK","url_lifetime_s":3600},
        "models":[{"id":"tiny","repository":"fixture/tiny","revision":revision,"archive_prefix":"models/tiny","archive_sha256":archive_hash,"engine":{"max_model_len":100,"max_num_seqs":8,"gpu_memory_utilization":0.8,"dtype":"bfloat16","language_model_only":true,"extra_args":[]},
            "prior":{"single_output_tps":100.0,"aggregate_output_tps":1000.0,"input_work_per_token":0.0,"startup_s":1.0,"kv_tokens":1000}}],
        "limits":{"max_instances":2,"max_usd_per_hour":3.0,"max_run_usd":1.0,"run_timeout_s":20,"startup_timeout_s":10,"request_timeout_s":5,"hydration_concurrency":2,"hydration_stall_s":2,"poll_ms":100}
    })).unwrap();
    let provider = MockProvider {
        url,
        pods: Arc::new(Mutex::new(vec![])),
        unknown,
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
async fn completion(State(bad): State<bool>) -> Response<Body> {
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
async fn single_call_does_not_duplicate_cold_capacity_and_hot_burst_conserves_usage() {
    let (mut runtime, provider, out, server) = fixture(false, false).await;
    let cold = runtime
        .run_workload("cold", &calls(1), 200.0, 10.0)
        .await
        .unwrap();
    assert_eq!(cold.launches, 1);
    assert_eq!(cold.completed, 1);
    assert_eq!(cold.invalid_actions, 0);
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
