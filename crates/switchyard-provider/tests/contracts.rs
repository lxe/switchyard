use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{Request, State},
    response::Response,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use switchyard_provider::{storage::Storage, *};

#[derive(Clone)]
struct Reply {
    status: u16,
    body: Value,
}
impl Reply {
    fn ok(body: Value) -> Self {
        Self { status: 200, body }
    }
}
#[derive(Debug)]
struct Captured {
    method: String,
    uri: String,
    headers: axum::http::HeaderMap,
    body: Vec<u8>,
}
#[derive(Clone)]
struct MockState {
    replies: Arc<Mutex<VecDeque<Reply>>>,
    captured: Arc<Mutex<Vec<Captured>>>,
}
struct Mock {
    url: String,
    state: MockState,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Mock {
    async fn start(replies: Vec<Reply>) -> Self {
        let state = MockState {
            replies: Arc::new(Mutex::new(replies.into())),
            captured: Arc::default(),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new().fallback(handler).with_state(state.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { url, state, task }
    }
    fn assert_finished(&self) {
        assert!(
            self.state.replies.lock().unwrap().is_empty(),
            "expected requests were not sent"
        );
    }
}
async fn handler(State(state): State<MockState>, request: Request) -> Response {
    let method = request.method().to_string();
    let uri = request.uri().to_string();
    let headers = request.headers().clone();
    let body = to_bytes(request.into_body(), 1024 * 1024)
        .await
        .unwrap()
        .to_vec();
    state.captured.lock().unwrap().push(Captured {
        method,
        uri,
        headers,
        body,
    });
    let reply = state
        .replies
        .lock()
        .unwrap()
        .pop_front()
        .expect("unexpected HTTP request");
    Response::builder()
        .status(reply.status)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&reply.body).unwrap()))
        .unwrap()
}

fn compute() -> ComputeConfig {
    ComputeConfig {
        api_url: "https://api.runpod.io/v2".into(),
        key_env: "RUNPOD_API_KEY".into(),
        cloud: "SECURE".into(),
        gpu_types: vec!["NVIDIA RTX A6000".into(), "NVIDIA GeForce RTX 4090".into()],
        data_centers: vec!["US-KS-2".into()],
        min_cuda: "12.9".into(),
        min_ram_per_gpu_gb: None,
        min_vcpu_count_per_gpu: None,
        disk_gb: 80,
        disk_usd_per_gb_month: 0.1,
        billing_hours_per_month: 730.0,
    }
}
fn storage(endpoint: &str) -> StorageConfig {
    StorageConfig {
        endpoint: endpoint.into(),
        region: "auto".into(),
        bucket: "bucket".into(),
        access_key_env: "R2_ACCESS_KEY".into(),
        secret_key_env: "R2_SECRET_KEY".into(),
        url_lifetime_s: 3600,
    }
}
fn launch() -> Launch {
    Launch {
        name: "switchyard-test-unique".into(),
        offering: Offering {
            gpu: "NVIDIA RTX A6000".into(),
            gpu_memory_gb: 48,
            data_center: "US-KS-2".into(),
            availability: "LOW".into(),
            gpu_usd_per_hour: 0.59,
            disk_usd_per_hour: 8.0 / 730.0,
            quoted_at_unix_s: 0,
        },
        container: ContainerSpec {
            image: "registry.example.test/generic-server:fixture".into(),
            disk_gb: 80,
            entrypoint: vec!["/bin/bash".into(), "-c".into()],
            command: vec!["exec /app/server --port 8080".into()],
            environment: std::collections::BTreeMap::from([(
                "APP_TOKEN".into(),
                "FAKE_PRIVATE_CONTAINER".into(),
            )]),
            ports: vec![
                Port {
                    number: 8080,
                    http: true,
                },
                Port {
                    number: 9000,
                    http: false,
                },
            ],
        },
    }
}
fn provider(endpoint: &str) -> RunPod {
    let mut settings = compute();
    settings.api_url = endpoint.into();
    RunPod::with_config(&settings, "FAKE_PRIVATE_KEY").unwrap()
}
fn pod(id: &str, name: &str) -> Value {
    json!({ "id":id,"name":name,"status":"RUNNING","cost":0.59,"dataCenterId":"US-KS-2",
        "env":{"SECRET":"FAKE_PRIVATE_RESPONSE"},"cmd":["FAKE_PRIVATE_RESPONSE"] })
}
fn inventory(pods: Vec<Value>) -> Reply {
    Reply::ok(json!({"pods":pods}))
}

#[tokio::test]
async fn catalog_filters_capacity_and_preserves_separate_disk_estimate() {
    let mock = Mock::start(vec![Reply::ok(json!({"gpus":[
        {"id":"NVIDIA RTX A6000","memory":48,"price":{"secure":0.59,"community":0.33},"dataCenters":[{"id":"US-KS-2","availability":"LOW"},{"id":"EU-RO-1","availability":"HIGH"}]},
        {"id":"NVIDIA GeForce RTX 4090","memory":24,"price":{"secure":0.89},"dataCenters":[{"id":"US-KS-2","availability":"NONE"}]},
        {"id":"unrequested","memory":80,"price":{"secure":0.01},"dataCenters":[{"id":"US-KS-2","availability":"HIGH"}]}
    ]}))]).await;
    let api = provider(&mock.url);
    let offers = api.offers().await.unwrap();
    assert_eq!(offers.len(), 1);
    assert_eq!(offers[0].gpu_memory_gb, 48);
    assert_eq!(offers[0].gpu_usd_per_hour, 0.59);
    assert!((offers[0].disk_usd_per_hour - 8.0 / 730.0).abs() < 1e-12);
    let captured = mock.state.captured.lock().unwrap();
    let query: std::collections::BTreeMap<_, _> =
        reqwest::Url::parse(&format!("{}{}", mock.url, captured[0].uri))
            .unwrap()
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
    assert_eq!(query["cloud"], "SECURE");
    assert_eq!(query["minCudaVersion"], "12.9");
    assert_eq!(query["include"], "AVAILABILITY");
    assert_eq!(query["count"], "1");
    assert_eq!(
        captured[0].headers["authorization"],
        "Bearer FAKE_PRIVATE_KEY"
    );
    assert_eq!(captured[0].headers["user-agent"], "Switchyard/0.1");
    mock.assert_finished();
}

#[tokio::test]
async fn launch_contract_keeps_credentials_out_of_retained_metadata() {
    let mock = Mock::start(vec![
        inventory(vec![]),
        Reply::ok(pod("owned123", "switchyard-test-unique")),
    ])
    .await;
    let api = provider(&mock.url);
    let created = api.create(launch()).await.unwrap();
    assert_eq!(
        api.endpoint_url(&created, 8080),
        "https://owned123-8080.proxy.runpod.net"
    );
    assert!(
        !serde_json::to_string(&created)
            .unwrap()
            .contains("FAKE_PRIVATE")
    );
    assert!(!format!("{created:?}").contains("FAKE_PRIVATE"));
    let captured = mock.state.captured.lock().unwrap();
    assert_eq!(captured[1].method, "POST");
    assert_eq!(captured[1].uri, "/pods");
    let body: Value = serde_json::from_slice(&captured[1].body).unwrap();
    assert_eq!(body["gpu"]["count"], 1);
    assert_eq!(body["gpu"]["minCudaVersion"], "12.9");
    assert!(body["gpu"].get("minRamPerGpu").is_none());
    assert!(body["gpu"].get("minVcpuCountPerGpu").is_none());
    assert_eq!(body["dataCenterIds"], json!(["US-KS-2"]));
    assert_eq!(body["ports"], json!(["8080/http", "9000/tcp"]));
    assert_eq!(body["startSsh"], false);
    assert_eq!(body["disk"], 80);
    assert_eq!(
        body["image"],
        "registry.example.test/generic-server:fixture"
    );
    assert_eq!(body["env"], json!({"APP_TOKEN":"FAKE_PRIVATE_CONTAINER"}));
    assert_eq!(body["entrypoint"], json!(["/bin/bash", "-c"]));
    assert_eq!(body["cmd"], json!(["exec /app/server --port 8080"]));
    mock.assert_finished();
}

#[tokio::test]
async fn launch_maps_host_placement_floors_without_inventing_container_options() {
    let mock = Mock::start(vec![
        inventory(vec![]),
        Reply::ok(pod("owned123", "switchyard-test-unique")),
    ])
    .await;
    let mut config = compute();
    config.api_url = mock.url.clone();
    config.min_ram_per_gpu_gb = Some(64);
    config.min_vcpu_count_per_gpu = Some(8);
    let api = RunPod::with_config(&config, "FAKE_PRIVATE_KEY").unwrap();
    api.create(launch()).await.unwrap();
    let captured = mock.state.captured.lock().unwrap();
    let body: Value = serde_json::from_slice(&captured[1].body).unwrap();
    assert_eq!(body["gpu"]["minRamPerGpu"], 64);
    assert_eq!(body["gpu"]["minVcpuCountPerGpu"], 8);
    assert!(body.get("shmSize").is_none());
    mock.assert_finished();
    config.min_ram_per_gpu_gb = Some(0);
    assert!(config.validate().is_err());
    config.min_ram_per_gpu_gb = None;
    config.min_vcpu_count_per_gpu = Some(0);
    assert!(config.validate().is_err());
}

#[tokio::test]
async fn lost_create_response_reconciles_eventual_inventory_without_second_post() {
    let mock = Mock::start(vec![
        inventory(vec![]),
        Reply {
            status: 503,
            body: json!({"error":"FAKE_PRIVATE_TICKET"}),
        },
        inventory(vec![]),
        inventory(vec![pod("owned123", "switchyard-test-unique")]),
    ])
    .await;
    let api = provider(&mock.url);
    assert_eq!(api.create(launch()).await.unwrap().id, "owned123");
    assert_eq!(
        mock.state
            .captured
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.method == "POST")
            .count(),
        1
    );
    mock.assert_finished();
}

#[tokio::test]
async fn malformed_success_with_ambiguous_inventory_preserves_unknown_launch() {
    let mock = Mock::start(vec![
        inventory(vec![]),
        Reply::ok(json!({"unexpected":"FAKE_PRIVATE_TICKET"})),
        inventory(vec![
            pod("owned123", "switchyard-test-unique"),
            pod("owned456", "switchyard-test-unique"),
        ]),
    ])
    .await;
    let api = provider(&mock.url);
    let error = api.create(launch()).await.unwrap_err();
    assert!(matches!(&error, Error::UnknownLaunch {name} if name=="switchyard-test-unique"));
    assert!(!error.to_string().contains("FAKE_PRIVATE"));
    mock.assert_finished();
}

#[tokio::test]
async fn preexisting_name_never_allocates_or_adopts_a_pod() {
    let mock = Mock::start(vec![inventory(vec![pod(
        "foreign",
        "switchyard-test-unique",
    )])])
    .await;
    let api = provider(&mock.url);
    assert!(matches!(api.create(launch()).await, Err(Error::Config(_))));
    assert!(
        mock.state
            .captured
            .lock()
            .unwrap()
            .iter()
            .all(|r| r.method == "GET")
    );
    mock.assert_finished();
}

#[tokio::test]
async fn missing_pod_reads_and_deletes_are_idempotent() {
    let mock = Mock::start(vec![
        Reply {
            status: 404,
            body: json!({}),
        },
        Reply {
            status: 404,
            body: json!({}),
        },
    ])
    .await;
    let api = provider(&mock.url);
    assert!(api.get("gone123").await.unwrap().is_none());
    api.terminate("gone123").await.unwrap();
    assert!(matches!(
        api.get("../foreign?secret").await,
        Err(Error::Config(_))
    ));
    mock.assert_finished();
}

#[tokio::test]
async fn provider_errors_do_not_echo_auth_urls_or_response_bodies() {
    let mock = Mock::start(vec![Reply {
        status: 403,
        body: json!({"error":"FAKE_PRIVATE_RESPONSE"}),
    }])
    .await;
    let api = provider(&mock.url);
    let error = api.offers().await.unwrap_err();
    assert_eq!(error.to_string(), "provider returned HTTP 403 for catalog");
    assert!(!format!("{error:?}").contains("FAKE_PRIVATE"));
    mock.assert_finished();
}

#[tokio::test]
async fn signed_object_requests_bind_payload_and_escape_reserved_key_characters() {
    let mock = Mock::start(vec![
        Reply::ok(json!({})),
        Reply::ok(json!({"ok":true})),
        Reply {
            status: 404,
            body: json!({}),
        },
    ])
    .await;
    let store = Storage::with_keys(
        &storage(&mock.url),
        "example-access".into(),
        "example-secret".into(),
    )
    .unwrap();
    store
        .put_json("models/a b+c%.json", &json!({"payload":"opaque"}))
        .await
        .unwrap();
    assert_eq!(
        store.get_json::<Value>("models/a b+c%.json").await.unwrap(),
        json!({"ok":true})
    );
    store.delete("models/a b+c%.json").await.unwrap();
    let captured = mock.state.captured.lock().unwrap();
    assert_eq!(
        captured
            .iter()
            .map(|r| r.method.as_str())
            .collect::<Vec<_>>(),
        ["PUT", "GET", "DELETE"]
    );
    for request in captured.iter() {
        assert_eq!(request.uri, "/bucket/models/a%20b%2Bc%25.json");
        let expected = format!("{:x}", Sha256::digest(&request.body));
        assert_eq!(request.headers["x-amz-content-sha256"], expected);
        let auth = request.headers["authorization"].to_str().unwrap();
        assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=example-access/"));
        assert!(auth.contains("/auto/s3/aws4_request"));
        assert!(auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"));
    }
    mock.assert_finished();
}

#[tokio::test]
async fn storage_errors_do_not_echo_signed_access_or_body() {
    let mock = Mock::start(vec![Reply {
        status: 403,
        body: json!({"error":"FAKE_PRIVATE_RESPONSE"}),
    }])
    .await;
    let store = Storage::with_keys(
        &storage(&mock.url),
        "example-access".into(),
        "example-secret".into(),
    )
    .unwrap();
    let error = store.get_json::<Value>("object").await.unwrap_err();
    assert_eq!(
        error.to_string(),
        "provider returned HTTP 403 for object request"
    );
    mock.assert_finished();
}

#[test]
fn endpoint_configuration_rejects_embedded_credentials_and_remote_cleartext() {
    for endpoint in [
        "https://key:secret@objects.example.test",
        "https://objects.example.test?secret=x",
        "http://objects.example.test",
    ] {
        assert!(storage(endpoint).validate().is_err());
        let mut bad = compute();
        bad.api_url = endpoint.into();
        assert!(bad.validate().is_err());
        assert!(
            Storage::with_keys(
                &storage(endpoint),
                "example-access".into(),
                "example-secret".into()
            )
            .is_err()
        );
        assert!(RunPod::with_key(endpoint, "fake").is_err());
    }
}

#[tokio::test]
async fn direct_rejected_creation_does_not_reconcile_or_repeat_paid_mutation() {
    let mock = Mock::start(vec![
        inventory(vec![]),
        Reply {
            status: 400,
            body: json!({"error":"FAKE_PRIVATE_TICKET"}),
        },
    ])
    .await;
    let api = provider(&mock.url);
    assert!(matches!(
        api.create(launch()).await,
        Err(Error::Http { status: 400, .. })
    ));
    assert_eq!(
        mock.state
            .captured
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.method == "POST")
            .count(),
        1
    );
    mock.assert_finished();
}

#[tokio::test]
async fn billing_request_keeps_provider_credentials_in_headers() {
    let mock = Mock::start(vec![Reply::ok(
        json!({"records":[{"podId":"owned123","totalAmount":0.01}]}),
    )])
    .await;
    let api = provider(&mock.url);
    let billing = api
        .billing("2026-10-10T00:00:00Z", "2026-10-11T00:00:00Z")
        .await
        .unwrap();
    assert_eq!(billing["records"][0]["podId"], "owned123");
    let captured = mock.state.captured.lock().unwrap();
    let url = reqwest::Url::parse(&format!("{}{}", mock.url, captured[0].uri)).unwrap();
    let pairs: std::collections::BTreeMap<_, _> = url
        .query_pairs()
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    assert_eq!(url.path(), "/billing/pods");
    assert!(!pairs.contains_key("grouping"));
    assert_eq!(pairs["bucketSize"], "hour");
    assert_eq!(pairs["startTime"], "2026-10-10T00:00:00Z");
    assert!(!captured[0].uri.contains("FAKE_PRIVATE"));
    mock.assert_finished();
}

#[test]
fn local_ipv6_http_endpoints_support_contract_testing() {
    let mut cfg = compute();
    cfg.api_url = "http://[::1]:8080/v2".into();
    cfg.validate().unwrap();
}

#[tokio::test]
async fn rest_only_client_refuses_capacity_discovery_and_allocation() {
    let mock = Mock::start(vec![inventory(vec![])]).await;
    let api = RunPod::with_key(&mock.url, "FAKE_PRIVATE_KEY").unwrap();
    assert!(matches!(api.offers().await, Err(Error::Config(_))));
    assert!(matches!(api.create(launch()).await, Err(Error::Config(_))));
    assert!(api.list().await.unwrap().is_empty());
    assert_eq!(mock.state.captured.lock().unwrap().len(), 1);
    mock.assert_finished();
}

#[tokio::test]
async fn allocation_rejects_a_container_disk_different_from_its_quote() {
    let mock = Mock::start(vec![]).await;
    let api = provider(&mock.url);
    let mut request = launch();
    request.container.disk_gb = 81;
    assert!(matches!(api.create(request).await, Err(Error::Config(_))));
    assert!(mock.state.captured.lock().unwrap().is_empty());
    mock.assert_finished();
}

#[test]
fn container_validation_checks_opaque_fields_without_engine_assumptions() {
    let mut container = launch().container;
    container.command = vec!["custom-server".into(), "--host".into(), "0.0.0.0".into()];
    container
        .environment
        .insert("MULTILINE_CONFIG".into(), "line one\nline two".into());
    container.validate().unwrap();
    let mut bad = container.clone();
    bad.ports.push(Port {
        number: 8080,
        http: true,
    });
    assert!(bad.validate().is_err());
    let mut bad = container.clone();
    bad.ports[0].number = 0;
    assert!(bad.validate().is_err());
    let mut bad = container.clone();
    bad.command.push("embedded\0nul".into());
    assert!(bad.validate().is_err());
    let mut bad = container;
    bad.environment.insert("BAD NAME".into(), "private".into());
    assert!(bad.validate().is_err());
}

#[test]
fn capacity_and_storage_validate_their_own_configuration_fields() {
    compute().validate().unwrap();
    storage("https://objects.example.test").validate().unwrap();
    let mut bad = compute();
    bad.disk_gb = 0;
    assert!(bad.validate().is_err());
    let mut bad = compute();
    bad.gpu_types.clear();
    assert!(bad.validate().is_err());
    let mut bad = compute();
    bad.disk_usd_per_gb_month = f64::NAN;
    assert!(bad.validate().is_err());
    let mut bad = compute();
    bad.billing_hours_per_month = 0.0;
    assert!(bad.validate().is_err());
    let mut bad = storage("https://objects.example.test");
    bad.bucket = "..".into();
    assert!(bad.validate().is_err());
    let mut bad = storage("https://objects.example.test");
    bad.access_key_env = "BAD NAME".into();
    assert!(bad.validate().is_err());
    let mut bad = storage("https://objects.example.test");
    bad.url_lifetime_s = 59;
    assert!(bad.validate().is_err());
}

#[tokio::test]
async fn vendor_lifecycle_states_are_normalized_before_leaving_the_adapter() {
    let cases = [
        ("PROVISIONING", PodStatus::Starting),
        ("STARTING", PodStatus::Starting),
        ("RUNNING", PodStatus::Running),
        ("EXITED", PodStatus::Stopped),
        ("ERROR", PodStatus::Failed),
        ("TERMINATED", PodStatus::Terminated),
        ("DELETED", PodStatus::Terminated),
        ("FUTURE_VENDOR_STATE", PodStatus::Unknown),
    ];
    let replies = cases
        .iter()
        .map(|(status, _)| {
            let mut wire = pod("owned123", "owned-name");
            wire["status"] = json!(status);
            Reply::ok(wire)
        })
        .collect();
    let mock = Mock::start(replies).await;
    let api = provider(&mock.url);
    for (_, expected) in cases {
        let observed = api.get("owned123").await.unwrap().unwrap();
        assert_eq!(observed.status, expected);
        assert_eq!(
            observed.status.is_terminated(),
            expected == PodStatus::Terminated
        );
        let serialized = serde_json::to_value(observed).unwrap();
        assert_eq!(serialized["data_center"], "US-KS-2");
        assert!(serialized.get("dataCenterId").is_none());
        assert_eq!(
            serialized["status"],
            serde_json::to_value(expected).unwrap()
        );
    }
    mock.assert_finished();
}

#[test]
fn generic_status_serialization_and_legacy_aliases_are_unambiguous() {
    let cases = [
        (PodStatus::Starting, "starting", "PROVISIONING"),
        (PodStatus::Running, "running", "RUNNING"),
        (PodStatus::Stopped, "stopped", "EXITED"),
        (PodStatus::Failed, "failed", "ERROR"),
        (PodStatus::Terminated, "terminated", "TERMINATED"),
        (PodStatus::Unknown, "unknown", "FUTURE_VENDOR_STATE"),
    ];
    for (expected, normal, legacy) in cases {
        assert_eq!(serde_json::to_value(expected).unwrap(), json!(normal));
        assert_eq!(
            serde_json::from_value::<PodStatus>(json!(normal)).unwrap(),
            expected
        );
        let mut wire = pod("legacy123", "legacy-name");
        wire["status"] = json!(legacy);
        let recovered: Pod = serde_json::from_value(wire).unwrap();
        assert_eq!(recovered.status, expected);
        assert_eq!(recovered.data_center.as_deref(), Some("US-KS-2"));
    }
    for status in [
        PodStatus::Starting,
        PodStatus::Running,
        PodStatus::Stopped,
        PodStatus::Failed,
        PodStatus::Unknown,
    ] {
        assert!(!status.is_terminated());
    }
    assert!(serde_json::from_str::<Pod>(r#"{"id":"x","name":"x","status":"running","cost":0,"data_center":"a","dataCenterId":"b"}"#).is_err());
}

#[tokio::test]
async fn empty_vendor_status_is_invalid_rather_than_confirmed_termination() {
    let mut wire = pod("owned123", "owned-name");
    wire["status"] = json!("");
    let mock = Mock::start(vec![Reply::ok(wire)]).await;
    assert!(matches!(
        provider(&mock.url).get("owned123").await,
        Err(Error::Response(_))
    ));
    mock.assert_finished();
}

#[tokio::test]
async fn empty_container_entrypoint_and_command_preserve_image_defaults() {
    let mock = Mock::start(vec![
        inventory(vec![]),
        Reply::ok(pod("owned123", "switchyard-test-unique")),
    ])
    .await;
    let api = provider(&mock.url);
    let mut request = launch();
    request.container.entrypoint.clear();
    request.container.command.clear();
    api.create(request).await.unwrap();
    let captured = mock.state.captured.lock().unwrap();
    let body: Value = serde_json::from_slice(&captured[1].body).unwrap();
    assert!(body.get("entrypoint").is_none());
    assert!(body.get("cmd").is_none());
    mock.assert_finished();
}
