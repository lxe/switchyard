//! Contract tests exercise the same Axum router used by the server. They use a
//! manual virtual clock and analytical hardware; no real provider is contacted.
use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use switchyard_gauntlet::CloudConfig;
use switchyard_gauntlet::engine::CloudSimulator;
use switchyard_gauntlet::http::router;
use switchyard_gauntlet::{CapacityPoint, HostProfile, Profile, Startup, Workload};
use tokio::sync::Mutex;
use tower::ServiceExt;

fn hardware() -> Profile {
    Profile {
        id: "test-gpu".into(),
        model: "test-model".into(),
        workload: Workload::Text,
        gpu: "analytical-test-accelerator".into(),
        usd_per_hour: 3.6,
        startup: Startup {
            allocation_s: 0.0,
            container_s: 0.0,
            weights_s: 0.0,
            engine_s: 3.0,
            jitter: 0.0,
            restore_samples_s: vec![],
            build_samples_s: vec![],
        },
        network_s: 0.0,
        network_bytes_per_s: 100_000_000.0,
        max_in_flight: 2,
        kv_tokens: 10_000,
        input_work_per_token: 0.0,
        image_work_per_megapixel: 0.0,
        adapter_load_s: 0.0,
        multi_adapter_factor: 1.0,
        adapter_cache_slots: 2,
        minimum_bill_s: 1.0,
        curve: vec![
            CapacityPoint {
                in_flight: 1,
                work_units_per_s: 100.0,
            },
            CapacityPoint {
                in_flight: 2,
                work_units_per_s: 100.0,
            },
        ],
        evidence: "analytical HTTP contract fixture".into(),
        source: "tests/http_contract.rs".into(),
        host: HostProfile::default(),
    }
}

fn api_with(profile: Profile) -> Router {
    api_with_profiles(vec![profile])
}

fn api_with_profiles(profiles: Vec<Profile>) -> Router {
    let mut config = CloudConfig::default();
    config.scheduling.probe.minimum_window_s = 1.0;
    router(Arc::new(Mutex::new(
        CloudSimulator::with_config(profiles, config, 7).unwrap(),
    )))
}

fn api() -> Router {
    api_with(hardware())
}

struct Reply {
    status: StatusCode,
    location: Option<String>,
    body: Value,
    bytes: usize,
}

async fn raw_call(app: &Router, method: &str, path: &str, body: Body, json_type: bool) -> Reply {
    let mut builder = Request::builder().method(method).uri(path);
    if json_type {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
    }
    let response = app
        .clone()
        .oneshot(builder.body(body).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let location = response
        .headers()
        .get(header::LOCATION)
        .map(|value| value.to_str().unwrap().to_owned());
    let bytes = to_bytes(response.into_body(), 128 * 1024).await.unwrap();
    Reply {
        status,
        location,
        body: if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        },
        bytes: bytes.len(),
    }
}

async fn call(app: &Router, method: &str, path: &str, body: Option<Value>) -> Reply {
    let json_type = body.is_some();
    let body = body
        .map(|value| Body::from(serde_json::to_vec(&value).unwrap()))
        .unwrap_or_else(Body::empty);
    raw_call(app, method, path, body, json_type).await
}

async fn provision(app: &Router) -> usize {
    let reply = call(
        app,
        "POST",
        "/v1/instances",
        Some(json!({ "profile_id": "test-gpu" })),
    )
    .await;
    assert_eq!(reply.status, StatusCode::CREATED, "{}", reply.body);
    let id = reply.body["id"].as_u64().unwrap() as usize;
    assert_eq!(reply.location, Some(format!("/v1/instances/{id}")));
    id
}

async fn advance(app: &Router, seconds: f64) -> f64 {
    let reply = call(
        app,
        "POST",
        "/control/advance",
        Some(json!({ "seconds": seconds })),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    reply.body["time_s"].as_f64().unwrap()
}

fn inference() -> Value {
    json!({ "model": "test-model", "input_tokens": 10, "planned_output": 100, "max_output": 100 })
}

fn assert_error(reply: &Reply, status: StatusCode, code: &str) {
    assert_eq!(reply.status, status, "{}", reply.body);
    assert_eq!(reply.body["error"]["code"], code);
    assert!(reply.body["error"]["message"].as_str().is_some());
}

fn assert_no_private_fields(value: &Value) {
    match value {
        Value::Object(fields) => {
            for (name, value) in fields {
                assert!(
                    !matches!(
                        name.as_str(),
                        "actual_output"
                            | "physical_bandwidth"
                            | "sampled_startup_s"
                            | "setup_until_s"
                    ),
                    "private simulator field escaped through HTTP: {name}"
                );
                assert_no_private_fields(value);
            }
        }
        Value::Array(values) => values.iter().for_each(assert_no_private_fields),
        _ => {}
    }
}

#[tokio::test]
async fn lifecycle_requires_readiness_and_explicit_clock_advancement() {
    let app = api();
    let offerings = call(&app, "GET", "/v1/offerings", None).await;
    assert_eq!(offerings.status, StatusCode::OK);
    assert!(offerings.body.to_string().contains("test-gpu"));
    let empty = call(&app, "GET", "/v1/instances", None).await;
    assert_eq!(empty.body["instances"], json!([]));
    let id = provision(&app).await;
    let instance_path = format!("/v1/instances/{id}");
    let pending = call(&app, "GET", &instance_path, None).await;
    assert_eq!(pending.status, StatusCode::OK);
    assert_eq!(pending.body["id"], id);
    assert_eq!(pending.body["status"], "starting");
    assert_eq!(pending.body["ready_at_s"], Value::Null);
    assert_eq!(pending.body["host"]["probe"], Value::Null);
    assert_no_private_fields(&pending.body);
    let time = call(&app, "GET", "/control/time", None).await;
    assert_eq!(time.body, json!({ "time_s": 0.0 }));
    let too_soon = call(
        &app,
        "POST",
        &format!("{instance_path}/inference"),
        Some(inference()),
    )
    .await;
    assert_error(&too_soon, StatusCode::CONFLICT, "conflict");
    assert_eq!(advance(&app, 3.0).await, 3.0);
    let accepted = call(
        &app,
        "POST",
        &format!("{instance_path}/inference"),
        Some(inference()),
    )
    .await;
    assert_eq!(accepted.status, StatusCode::ACCEPTED, "{}", accepted.body);
    let request_id = accepted.body["id"].as_u64().unwrap();
    let request_path = format!("/v1/requests/{request_id}");
    assert_eq!(accepted.location, Some(request_path.clone()));
    let active = call(&app, "GET", &request_path, None).await;
    assert_eq!(active.status, StatusCode::OK);
    assert_eq!(active.body["id"], request_id);
    assert_no_private_fields(&active.body);
    advance(&app, 5.0).await;
    let completed = call(&app, "GET", &request_path, None).await;
    assert_eq!(completed.body["status"], "completed");
    assert_no_private_fields(&completed.body);
    let released = call(&app, "DELETE", &instance_path, None).await;
    assert_eq!(released.status, StatusCode::NO_CONTENT);
    assert_eq!(released.bytes, 0);
    let after_release = call(
        &app,
        "POST",
        &format!("{instance_path}/inference"),
        Some(inference()),
    )
    .await;
    assert_error(&after_release, StatusCode::CONFLICT, "conflict");
}

#[tokio::test]
async fn capacity_is_admitted_atomically_without_oversubscribing_the_instance() {
    let app = api();
    let id = provision(&app).await;
    advance(&app, 3.0).await;
    let path = format!("/v1/instances/{id}/inference");
    let (first, second) = tokio::join!(
        call(&app, "POST", &path, Some(inference())),
        call(&app, "POST", &path, Some(inference()))
    );
    assert_eq!(first.status, StatusCode::ACCEPTED);
    assert_eq!(second.status, StatusCode::ACCEPTED);
    assert_ne!(first.body["id"], second.body["id"]);
    let full = call(&app, "POST", &path, Some(inference())).await;
    assert_error(&full, StatusCode::TOO_MANY_REQUESTS, "capacity_exceeded");
    advance(&app, 5.0).await;
    let available = call(&app, "POST", &path, Some(inference())).await;
    assert_eq!(available.status, StatusCode::ACCEPTED);
}

#[tokio::test]
async fn hard_output_limit_reserves_memory_independently_of_the_estimate() {
    let mut profile = hardware();
    profile.kv_tokens = 150;
    let app = api_with(profile);
    let id = provision(&app).await;
    advance(&app, 3.0).await;
    let path = format!("/v1/instances/{id}/inference");
    let spec = json!({ "model": "test-model", "input_tokens": 10, "planned_output": 1, "max_output": 100 });
    let first = call(&app, "POST", &path, Some(spec.clone())).await;
    assert_eq!(first.status, StatusCode::ACCEPTED);
    let second = call(&app, "POST", &path, Some(spec)).await;
    assert_error(&second, StatusCode::TOO_MANY_REQUESTS, "capacity_exceeded");
}

#[tokio::test]
async fn clients_cannot_supply_sampled_truth_or_unknown_fields() {
    let app = api();
    let id = provision(&app).await;
    advance(&app, 3.0).await;
    for field in ["actual_output", "physical_bandwidth", "startup_duration_s"] {
        let mut spec = inference();
        spec[field] = json!(1);
        let reply = call(
            &app,
            "POST",
            &format!("/v1/instances/{id}/inference"),
            Some(spec),
        )
        .await;
        assert_error(&reply, StatusCode::UNPROCESSABLE_ENTITY, "invalid_json");
    }
    let accepted = call(
        &app,
        "POST",
        &format!("/v1/instances/{id}/inference"),
        Some(inference()),
    )
    .await;
    assert_eq!(accepted.status, StatusCode::ACCEPTED);
    assert_eq!(
        accepted.body["id"], 0,
        "rejections must not create requests"
    );
}

#[tokio::test]
async fn missing_resources_invalid_routes_and_methods_have_json_errors() {
    let app = api();
    for (method, path, body) in [
        ("GET", "/v1/instances/999", None),
        ("DELETE", "/v1/instances/999", None),
        ("POST", "/v1/instances/999/probes", None),
        ("POST", "/v1/instances/999/inference", Some(inference())),
        ("GET", "/v1/requests/999", None),
        ("GET", "/missing", None),
        (
            "POST",
            "/v1/instances",
            Some(json!({ "profile_id": "missing" })),
        ),
    ] {
        let reply = call(&app, method, path, body).await;
        assert_error(&reply, StatusCode::NOT_FOUND, "not_found");
    }
    let wrong_method = call(&app, "PATCH", "/control/time", None).await;
    assert_error(
        &wrong_method,
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
    );
    let invalid_path = call(&app, "GET", "/v1/instances/not-a-number", None).await;
    assert_error(&invalid_path, StatusCode::BAD_REQUEST, "invalid_path");
}

#[tokio::test]
async fn malformed_json_wrong_content_type_and_oversized_bodies_are_bounded() {
    let app = api();
    let malformed = raw_call(&app, "POST", "/v1/instances", Body::from("{"), true).await;
    assert_error(&malformed, StatusCode::BAD_REQUEST, "invalid_json");
    let wrong_type = raw_call(&app, "POST", "/v1/instances", Body::from("{}"), false).await;
    assert_error(
        &wrong_type,
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "invalid_json",
    );
    let oversized = call(
        &app,
        "POST",
        "/v1/instances",
        Some(json!({ "profile_id": "x".repeat(70 * 1024) })),
    )
    .await;
    assert_error(&oversized, StatusCode::PAYLOAD_TOO_LARGE, "invalid_json");
}

#[tokio::test]
async fn invalid_time_and_overflowing_token_bounds_do_not_mutate_clock() {
    let app = api();
    let negative = call(
        &app,
        "POST",
        "/control/advance",
        Some(json!({ "seconds": -1.0 })),
    )
    .await;
    assert_error(
        &negative,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_request",
    );
    assert_eq!(advance(&app, 0.0).await, 0.0);
    let id = provision(&app).await;
    advance(&app, 3.0).await;
    let overflow = call(
        &app,
        "POST",
        &format!("/v1/instances/{id}/inference"),
        Some(json!({ "model": "test-model", "input_tokens": u64::MAX,
                     "planned_output": 100, "max_output": u64::MAX })),
    )
    .await;
    assert_error(
        &overflow,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_request",
    );
    let time = call(&app, "GET", "/control/time", None).await;
    assert_eq!(time.body["time_s"], 3.0);
}

#[tokio::test]
async fn probe_receipts_are_asynchronous_and_release_cancels_pending_instance() {
    let app = api();
    let id = provision(&app).await;
    let path = format!("/v1/instances/{id}");
    advance(&app, 1.0).await;
    let probe = call(&app, "POST", &format!("{path}/probes"), None).await;
    assert_eq!(probe.status, StatusCode::ACCEPTED);
    assert_eq!(probe.body, json!({ "id": id, "status": "probe_pending" }));
    let duplicate = call(&app, "POST", &format!("{path}/probes"), None).await;
    assert_error(&duplicate, StatusCode::CONFLICT, "conflict");
    let released = call(&app, "DELETE", &path, None).await;
    assert_eq!(released.status, StatusCode::NO_CONTENT);
    advance(&app, 10.0).await;
    let late_probe = call(&app, "POST", &format!("{path}/probes"), None).await;
    assert_error(&late_probe, StatusCode::CONFLICT, "conflict");
    let after = call(&app, "GET", &path, None).await;
    assert_no_private_fields(&after.body);
    assert_eq!(after.body["status"], "released");
    assert_eq!(after.body["host"]["probe"], Value::Null);
}

#[tokio::test]
async fn probe_result_is_visible_only_after_its_completion_time() {
    let app = api();
    let id = provision(&app).await;
    let path = format!("/v1/instances/{id}");
    advance(&app, 1.0).await;
    let probe = call(&app, "POST", &format!("{path}/probes"), None).await;
    assert_eq!(probe.status, StatusCode::ACCEPTED);
    advance(&app, 0.5).await;
    let during = call(&app, "GET", &path, None).await;
    assert_eq!(during.body["host"]["probe_pending"], true);
    assert_eq!(during.body["host"]["probe"], Value::Null);
    assert_eq!(during.body["ready_at_s"], Value::Null);
    advance(&app, 0.5).await;
    let after = call(&app, "GET", &path, None).await;
    assert_eq!(after.body["host"]["probe_pending"], false);
    assert!(after.body["host"]["probe"].is_object());
    assert_eq!(after.body["host"]["probe"]["at_s"], 2.0);
    assert_eq!(after.body["host"]["probe"]["started_s"], 1.0);
    assert!(
        after.body["host"]["probe"]["received_bytes"]
            .as_u64()
            .unwrap()
            >= 8 * 1024 * 1024
    );
    assert_eq!(after.body["ready_at_s"], Value::Null);
    let too_soon = call(&app, "POST", &format!("{path}/probes"), None).await;
    assert_error(&too_soon, StatusCode::CONFLICT, "conflict");
    advance(&app, 1.0).await;
    let refresh = call(&app, "POST", &format!("{path}/probes"), None).await;
    assert_eq!(refresh.status, StatusCode::ACCEPTED);
    advance(&app, 1.0).await;
    let refreshed = call(&app, "GET", &path, None).await;
    assert_eq!(refreshed.body["host"]["probe"]["at_s"], 4.0);
}

#[tokio::test]
async fn manual_http_replay_is_identical_for_an_identical_seed() {
    async fn replay(app: &Router) -> Vec<Value> {
        let id = provision(app).await;
        let path = format!("/v1/instances/{id}");
        let mut result = vec![call(app, "GET", &path, None).await.body];
        advance(app, 3.0).await;
        let receipt = call(app, "POST", &format!("{path}/inference"), Some(inference())).await;
        let request_id = receipt.body["id"].as_u64().unwrap();
        advance(app, 0.5).await;
        result.push(
            call(app, "GET", &format!("/v1/requests/{request_id}"), None)
                .await
                .body,
        );
        advance(app, 5.0).await;
        result.push(
            call(app, "GET", &format!("/v1/requests/{request_id}"), None)
                .await
                .body,
        );
        result.push(call(app, "GET", &path, None).await.body);
        result
    }
    let a = replay(&api()).await;
    let b = replay(&api()).await;
    assert_eq!(a, b);
}

#[tokio::test]
async fn rejected_capacity_traffic_does_not_change_accepted_output_samples() {
    async fn replay(include_rejected_traffic: bool) -> Value {
        let app = api();
        let id = provision(&app).await;
        advance(&app, 3.0).await;
        let path = format!("/v1/instances/{id}/inference");
        let spec = json!({ "model": "test-model", "input_tokens": 10,
                           "planned_output": 1000, "max_output": 2000 });
        for _ in 0..2 {
            let accepted = call(&app, "POST", &path, Some(spec.clone())).await;
            assert_eq!(accepted.status, StatusCode::ACCEPTED);
        }
        if include_rejected_traffic {
            for _ in 0..10 {
                let rejected = call(&app, "POST", &path, Some(spec.clone())).await;
                assert_error(
                    &rejected,
                    StatusCode::TOO_MANY_REQUESTS,
                    "capacity_exceeded",
                );
            }
        }
        advance(&app, 60.0).await;
        let accepted = call(&app, "POST", &path, Some(spec)).await;
        assert_eq!(accepted.status, StatusCode::ACCEPTED);
        assert_eq!(accepted.body["id"], 2);
        advance(&app, 60.0).await;
        call(&app, "GET", "/v1/requests/2", None).await.body
    }
    assert_eq!(replay(false).await, replay(true).await);
}

fn chat() -> Value {
    json!({ "model": "test-model", "messages": [{ "role": "user", "content": "hello" }],
            "max_completion_tokens": 20, "simulator": { "input_tokens": 10, "planned_output": 20 } })
}

async fn wait_for_request(app: &Router, id: usize) {
    for _ in 0..100 {
        if call(app, "GET", &format!("/v1/requests/{id}"), None)
            .await
            .status
            == StatusCode::OK
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("chat request {id} was not accepted");
}

#[tokio::test]
async fn instance_endpoint_queues_chat_until_ready_then_returns_openai_json() {
    let app = api();
    let id = provision(&app).await;
    let instance = call(&app, "GET", &format!("/v1/instances/{id}"), None).await;
    let base = instance.body["openai_base_path"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(base, format!("/v1/instances/{id}/openai/v1"));
    let models = call(&app, "GET", &format!("{base}/models"), None).await;
    assert_eq!(models.body["object"], "list");
    assert_eq!(models.body["data"][0]["id"], "test-model");
    let task_app = app.clone();
    let task = tokio::spawn(async move {
        call(
            &task_app,
            "POST",
            &format!("{base}/chat/completions"),
            Some(chat()),
        )
        .await
    });
    wait_for_request(&app, 0).await;
    assert!(!task.is_finished());
    advance(&app, 8.0).await;
    let reply = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    assert_eq!(reply.body["object"], "chat.completion");
    assert_eq!(reply.body["model"], "test-model");
    assert_eq!(reply.body["choices"][0]["message"]["role"], "assistant");
    assert!(
        !reply.body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .is_empty()
    );
    assert_eq!(reply.body["usage"]["prompt_tokens"], 10);
    let output = reply.body["usage"]["completion_tokens"].as_u64().unwrap();
    assert!((1..=20).contains(&output));
    assert_eq!(reply.body["usage"]["total_tokens"], 10 + output);
}

#[tokio::test]
async fn sse_waits_for_first_token_and_finishes_with_usage_and_done() {
    let app = api();
    let id = provision(&app).await;
    let mut body = chat();
    body["stream"] = json!(true);
    body["stream_options"] = json!({ "include_usage": true });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/instances/{id}/openai/v1/chat/completions"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    let mut stream = response.into_body().into_data_stream();
    assert!(
        tokio::time::timeout(Duration::from_millis(20), stream.next())
            .await
            .is_err()
    );
    advance(&app, 3.5).await;
    let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let mut text = String::from_utf8(first.to_vec()).unwrap();
    assert!(text.contains("assistant"));
    advance(&app, 5.0).await;
    while let Some(frame) = tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
    {
        text.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
    }
    let events: Vec<_> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"));
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|value| serde_json::from_str(value).unwrap())
        .collect();
    assert!(chunks.iter().any(|chunk| {
        chunk["choices"][0]["delta"]["content"]
            .as_str()
            .is_some_and(|text| !text.is_empty())
    }));
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["choices"][0]["finish_reason"].is_string())
    );
    let usage = chunks.last().unwrap();
    assert_eq!(usage["choices"], json!([]));
    assert!(usage["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn queued_chat_release_returns_an_error_instead_of_waiting_forever() {
    let app = api();
    let id = provision(&app).await;
    let task_app = app.clone();
    let task = tokio::spawn(async move {
        call(
            &task_app,
            "POST",
            &format!("/v1/instances/{id}/openai/v1/chat/completions"),
            Some(chat()),
        )
        .await
    });
    wait_for_request(&app, 0).await;
    let released = call(&app, "DELETE", &format!("/v1/instances/{id}"), None).await;
    assert_eq!(released.status, StatusCode::NO_CONTENT);
    let reply = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap();
    assert_error(&reply, StatusCode::SERVICE_UNAVAILABLE, "inference_failed");
}

#[tokio::test]
async fn chat_extension_rejects_private_state_and_completion_limits_are_bounded() {
    let app = api();
    let id = provision(&app).await;
    let path = format!("/v1/instances/{id}/openai/v1/chat/completions");
    let mut private = chat();
    private["simulator"]["actual_output"] = json!(1);
    let reply = call(&app, "POST", &path, Some(private)).await;
    assert_error(&reply, StatusCode::UNPROCESSABLE_ENTITY, "invalid_json");
    for limit in [0, 65_537] {
        let mut body = chat();
        body["max_completion_tokens"] = json!(limit);
        let reply = call(&app, "POST", &path, Some(body)).await;
        assert_error(&reply, StatusCode::UNPROCESSABLE_ENTITY, "invalid_request");
    }
}

#[tokio::test]
async fn stable_frontdoor_provisions_a_cold_model_and_returns_completion() {
    let app = api();
    let models = call(&app, "GET", "/v1/models", None).await;
    assert_eq!(models.body["data"][0]["id"], "test-model");
    let task_app = app.clone();
    let task =
        tokio::spawn(
            async move { call(&task_app, "POST", "/v1/chat/completions", Some(chat())).await },
        );
    wait_for_request(&app, 0).await;
    assert!(!task.is_finished());
    advance(&app, 0.25).await;
    let instances = call(&app, "GET", "/v1/instances", None).await;
    assert!(!instances.body["instances"].as_array().unwrap().is_empty());
    assert_eq!(instances.body["instances"][0]["profile_id"], "test-gpu");
    advance(&app, 20.0).await;
    let reply = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    assert_eq!(reply.body["model"], "test-model");
}

#[tokio::test]
async fn frontdoor_routes_a_multi_model_burst_to_compatible_instances() {
    let first = hardware();
    let mut second = hardware();
    second.id = "other-gpu".into();
    second.model = "other-model".into();
    let app = api_with_profiles(vec![first, second]);
    let models = call(&app, "GET", "/v1/models", None).await;
    assert_eq!(models.body["data"].as_array().unwrap().len(), 2);
    let mut tasks = vec![];
    for index in 0..10 {
        let task_app = app.clone();
        let mut body = chat();
        let model = if index % 2 == 0 {
            "test-model"
        } else {
            "other-model"
        };
        body["model"] = json!(model);
        tasks.push((
            model,
            tokio::spawn(async move {
                call(&task_app, "POST", "/v1/chat/completions", Some(body)).await
            }),
        ));
    }
    wait_for_request(&app, 9).await;
    advance(&app, 20.0).await;
    for (model, task) in tasks {
        let reply = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert_eq!(reply.body["model"], model);
    }
    let instances = call(&app, "GET", "/v1/instances", None).await;
    let entries = instances.body["instances"].as_array().unwrap();
    assert!(entries.iter().any(|node| node["profile_id"] == "test-gpu"));
    assert!(entries.iter().any(|node| node["profile_id"] == "other-gpu"));
}

#[tokio::test]
async fn sse_follows_delivered_tokens_when_gpu_generation_finishes_before_egress() {
    let mut profile = hardware();
    profile.network_s = 1.0;
    profile.network_bytes_per_s = 40.0; // Ten synthetic tokens per second.
    let app = api_with(profile);
    let id = provision(&app).await;
    let mut body = chat();
    body["stream"] = json!(true);
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/instances/{id}/openai/v1/chat/completions"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let mut stream = response.into_body().into_data_stream();
    advance(&app, 4.5).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(20), stream.next())
            .await
            .is_err()
    );
    advance(&app, 0.75).await;
    let receipt = call(&app, "GET", "/v1/requests/0", None).await.body;
    assert_eq!(receipt["status"], "active");
    let delivered = receipt["generated_tokens"].as_u64().unwrap();
    assert!((1..20).contains(&delivered));
    let role = tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(std::str::from_utf8(&role).unwrap().contains("assistant"));
    let content = tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let event = std::str::from_utf8(&content)
        .unwrap()
        .trim()
        .strip_prefix("data: ")
        .unwrap();
    let chunk: Value = serde_json::from_str(event).unwrap();
    assert_eq!(
        chunk["choices"][0]["delta"]["content"]
            .as_str()
            .unwrap()
            .matches(" tok")
            .count() as u64,
        delivered
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(20), stream.next())
            .await
            .is_err()
    );
    advance(&app, 10.0).await;
    while tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .is_some()
    {}
}
