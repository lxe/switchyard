//! Generic cloud-provider HTTP boundary for the simulated fleet.
//!
//! All lifecycle state comes from `CloudSimulator`'s observable snapshot. The
//! API accepts output estimates and limits; sampled output length and physical
//! startup state remain private to the simulator. The control routes expose a
//! virtual clock so clients can replay scheduling experiments deterministically.
use crate::config::{CloudConfig, Profile, Workload};
use crate::engine::{CloudSimulator, InferenceSpec};
use axum::extract::rejection::{JsonRejection, PathRejection};
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::error::Error;
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

type SharedCloud = Arc<Mutex<CloudSimulator>>;
type ApiResult = Result<Response, ApiError>;

/// Construct the provider API without opening a socket or advancing time.
///
/// `GET /v1/instances` returns `{ "instances": [...] }`. A single instance or
/// request is returned directly, with the same numeric `id` as its creation
/// receipt. `POST /control/advance` accepts `{ "seconds": 1.0 }` and returns
/// `{ "time_s": 1.0 }`; advancing by zero is valid.
pub fn router(cloud: SharedCloud) -> Router {
    Router::new()
        .route("/v1/offerings", get(offerings))
        .route("/v1/instances", get(instances).post(provision))
        .route("/v1/instances/{id}", get(instance).delete(release))
        .route("/v1/instances/{id}/probes", post(probe))
        .route("/v1/instances/{id}/inference", post(inference))
        .route("/v1/requests/{id}", get(request))
        .route("/v1/models", get(catalog_models))
        .route("/v1/chat/completions", post(routed_chat_completion))
        .route("/v1/instances/{id}/openai/v1/models", get(models))
        .route(
            "/v1/instances/{id}/openai/v1/chat/completions",
            post(chat_completion),
        )
        .route("/control/advance", post(advance))
        .route("/control/time", get(time))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(DefaultBodyLimit::max(64 * 1024))
        .with_state(cloud)
}

/// Serve the fleet API and advance its virtual clock at `clock_rate` seconds
/// per real second. A rate of zero leaves advancement to the control API.
///
/// The caller chooses the bind address. The CLI defaults to a loopback address.
/// Clock and HTTP futures share the same lock, making every mutation atomic.
pub async fn serve(
    profiles: Vec<Profile>,
    seed: u64,
    bind: SocketAddr,
    clock_rate: f64,
) -> Result<(), Box<dyn Error>> {
    serve_with_config(profiles, CloudConfig::default(), seed, bind, clock_rate).await
}

/// Serve with explicit provider/environment and routing-runtime settings.
pub async fn serve_with_config(
    profiles: Vec<Profile>,
    config: CloudConfig,
    seed: u64,
    bind: SocketAddr,
    clock_rate: f64,
) -> Result<(), Box<dyn Error>> {
    if !clock_rate.is_finite() || clock_rate < 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "clock rate must be finite and nonnegative",
        )
        .into());
    }
    let simulator = CloudSimulator::with_config(profiles, config, seed)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    let cloud = Arc::new(Mutex::new(simulator));
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let server = axum::serve(listener, router(cloud.clone()))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .into_future();
    if clock_rate == 0.0 {
        server.await?;
    } else {
        // Neither branch owns a spawned task: cancellation of serve also drops
        // the clock future, so a stopped server cannot keep mutating the fleet.
        tokio::select! {
            result = server => result?,
            result = run_clock(cloud, clock_rate) => {
                let message = result.err().unwrap_or_else(|| "simulation clock stopped".into());
                return Err(std::io::Error::other(message).into());
            }
        }
    }
    Ok(())
}

async fn run_clock(cloud: SharedCloud, clock_rate: f64) -> Result<(), String> {
    let mut interval = tokio::time::interval(Duration::from_millis(50));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut previous = tokio::time::Instant::now();
    loop {
        interval.tick().await;
        let now = tokio::time::Instant::now();
        let elapsed = (now - previous).as_secs_f64() * clock_rate;
        previous = now;
        cloud.lock().await.advance(elapsed)?;
    }
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn engine(message: String) -> Self {
        let normalized = message.to_ascii_lowercase();
        if normalized.starts_with("invalid") || normalized.starts_with("validation") {
            Self::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", message)
        } else if normalized.contains("not found") || normalized.starts_with("unknown") {
            Self::new(StatusCode::NOT_FOUND, "not_found", message)
        } else if normalized.starts_with("capacity")
            || normalized.contains("capacity exceeded")
            || normalized.contains("queue full")
        {
            Self::new(StatusCode::TOO_MANY_REQUESTS, "capacity_exceeded", message)
        } else if normalized.contains("not ready")
            || normalized.contains("released")
            || normalized.contains("already")
            || normalized.contains("pending")
            || normalized.contains("unavailable")
            || normalized.contains("busy")
        {
            Self::new(StatusCode::CONFLICT, "conflict", message)
        } else {
            Self::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", message)
        }
    }

    fn json(rejection: JsonRejection) -> Self {
        Self::new(rejection.status(), "invalid_json", rejection.body_text())
    }

    fn path(rejection: PathRejection) -> Self {
        Self::new(rejection.status(), "invalid_path", rejection.body_text())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({ "error": { "code": self.code, "message": self.message, "type": self.code, "param": null } })),
        )
            .into_response()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProvisionSpec {
    profile_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AdvanceSpec {
    seconds: f64,
}

// An explicit DTO is intentional: adding a private field to InferenceSpec must
// not silently expand the client contract. Unknown fields, including an actual
// output length, are rejected before calling the engine.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InferenceBody {
    model: String,
    input_tokens: u64,
    planned_output: u64,
    max_output: u64,
    #[serde(default)]
    upload_bytes: u64,
    #[serde(default = "text_workload")]
    workload: Workload,
    #[serde(default)]
    image_megapixels: f64,
    #[serde(default)]
    adapter: Option<String>,
    #[serde(default = "standard_class")]
    class: String,
    #[serde(default)]
    cached_fraction: f64,
}

fn text_workload() -> Workload {
    Workload::Text
}

fn standard_class() -> String {
    "standard".into()
}

impl From<InferenceBody> for InferenceSpec {
    fn from(body: InferenceBody) -> Self {
        Self {
            model: body.model,
            input_tokens: body.input_tokens,
            planned_output: body.planned_output,
            max_output: body.max_output,
            upload_bytes: body.upload_bytes,
            workload: body.workload,
            image_megapixels: body.image_megapixels,
            adapter: body.adapter,
            class: body.class,
            cached_fraction: body.cached_fraction,
        }
    }
}

fn visible_instance(snapshot: &Value, id: usize) -> Result<Value, ApiError> {
    snapshot
        .get("instances")
        .and_then(Value::as_array)
        .and_then(|instances| {
            instances
                .iter()
                .find(|instance| instance.get("id").and_then(Value::as_u64) == Some(id as u64))
        })
        .cloned()
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found",
                format!("unknown instance {id}"),
            )
        })
}

async fn offerings(State(cloud): State<SharedCloud>) -> Response {
    Json(cloud.lock().await.offerings()).into_response()
}

async fn instances(State(cloud): State<SharedCloud>) -> Response {
    let snapshot = cloud.lock().await.snapshot();
    let mut instances = snapshot["instances"].clone();
    if let Some(entries) = instances.as_array_mut() {
        for entry in entries {
            add_endpoint(entry);
        }
    }
    Json(json!({ "instances": instances })).into_response()
}

async fn instance(
    State(cloud): State<SharedCloud>,
    path: Result<Path<usize>, PathRejection>,
) -> ApiResult {
    let Path(id) = path.map_err(ApiError::path)?;
    let snapshot = cloud.lock().await.snapshot();
    let mut instance = visible_instance(&snapshot, id)?;
    add_endpoint(&mut instance);
    Ok(Json(instance).into_response())
}

async fn provision(
    State(cloud): State<SharedCloud>,
    body: Result<Json<ProvisionSpec>, JsonRejection>,
) -> ApiResult {
    let Json(spec) = body.map_err(ApiError::json)?;
    let id = cloud
        .lock()
        .await
        .provision(&spec.profile_id)
        .map_err(ApiError::engine)?;
    Ok((
        StatusCode::CREATED,
        [(header::LOCATION, format!("/v1/instances/{id}"))],
        Json(json!({ "id": id, "openai_base_path": endpoint(id) })),
    )
        .into_response())
}

async fn release(
    State(cloud): State<SharedCloud>,
    path: Result<Path<usize>, PathRejection>,
) -> ApiResult {
    let Path(id) = path.map_err(ApiError::path)?;
    cloud.lock().await.release(id).map_err(ApiError::engine)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

async fn probe(
    State(cloud): State<SharedCloud>,
    path: Result<Path<usize>, PathRejection>,
) -> ApiResult {
    let Path(id) = path.map_err(ApiError::path)?;
    cloud.lock().await.probe(id).map_err(ApiError::engine)?;
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({ "id": id, "status": "probe_pending" })),
    )
        .into_response())
}

async fn inference(
    State(cloud): State<SharedCloud>,
    path: Result<Path<usize>, PathRejection>,
    body: Result<Json<InferenceBody>, JsonRejection>,
) -> ApiResult {
    let Path(node) = path.map_err(ApiError::path)?;
    let Json(spec) = body.map_err(ApiError::json)?;
    let id = cloud
        .lock()
        .await
        .submit(node, spec.into())
        .map_err(ApiError::engine)?;
    Ok((
        StatusCode::ACCEPTED,
        [(header::LOCATION, format!("/v1/requests/{id}"))],
        Json(json!({ "id": id })),
    )
        .into_response())
}

async fn request(
    State(cloud): State<SharedCloud>,
    path: Result<Path<usize>, PathRejection>,
) -> ApiResult {
    let Path(id) = path.map_err(ApiError::path)?;
    let request = cloud.lock().await.request(id).map_err(ApiError::engine)?;
    Ok(Json(request).into_response())
}

async fn advance(
    State(cloud): State<SharedCloud>,
    body: Result<Json<AdvanceSpec>, JsonRejection>,
) -> ApiResult {
    let Json(spec) = body.map_err(ApiError::json)?;
    if !spec.seconds.is_finite() || spec.seconds < 0.0 {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "seconds must be finite and nonnegative",
        ));
    }
    let mut cloud = cloud.lock().await;
    cloud.advance(spec.seconds).map_err(ApiError::engine)?;
    Ok(Json(json!({ "time_s": cloud.clock() })).into_response())
}

async fn time(State(cloud): State<SharedCloud>) -> Response {
    Json(json!({ "time_s": cloud.lock().await.clock() })).into_response()
}

async fn not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "not_found", "unknown API route")
}

async fn method_not_allowed() -> ApiError {
    ApiError::new(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        "method is not supported for this route",
    )
}

fn endpoint(id: usize) -> String {
    format!("/v1/instances/{id}/openai/v1")
}

fn add_endpoint(instance: &mut Value) {
    if let Some(id) = instance.get("id").and_then(Value::as_u64) {
        instance["openai_base_path"] = json!(endpoint(id as usize));
    }
}

fn instance_profile(cloud: &CloudSimulator, id: usize) -> Result<(String, Workload), ApiError> {
    let instance = visible_instance(&cloud.snapshot(), id)?;
    let offerings = cloud.offerings();
    let profile = offerings["offerings"]
        .as_array()
        .and_then(|profiles| {
            profiles
                .iter()
                .find(|profile| profile["id"] == instance["profile_id"])
        })
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "instance offering is unavailable",
            )
        })?;
    let model = profile["model"].as_str().ok_or_else(|| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "offering has no model",
        )
    })?;
    let workload = serde_json::from_value(profile["workload"].clone()).map_err(|_| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "offering has no workload",
        )
    })?;
    Ok((model.to_owned(), workload))
}

async fn models(
    State(cloud): State<SharedCloud>,
    path: Result<Path<usize>, PathRejection>,
) -> ApiResult {
    let Path(id) = path.map_err(ApiError::path)?;
    let model = {
        let simulator = cloud.lock().await;
        instance_profile(&simulator, id)?.0
    };
    Ok(Json(json!({ "object": "list", "data": [{ "id": model, "object": "model", "created": 0, "owned_by": "simulator" }] })).into_response())
}

async fn catalog_models(State(cloud): State<SharedCloud>) -> Response {
    let offerings = cloud.lock().await.offerings();
    let models: std::collections::BTreeSet<_> = offerings["offerings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|profile| profile["model"].as_str())
        .collect();
    let data: Vec<_> = models.into_iter().map(|model| json!({ "id": model, "object": "model", "created": 0, "owned_by": "simulator" })).collect();
    Json(json!({ "object": "list", "data": data })).into_response()
}

fn catalog_workload(
    cloud: &CloudSimulator,
    model: &str,
    vision: bool,
) -> Result<Workload, ApiError> {
    let offerings = cloud.offerings();
    let matching: Vec<_> = offerings["offerings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|profile| profile["model"] == model)
        .collect();
    let preferred = if vision { "vision" } else { "text" };
    let profile = matching
        .iter()
        .find(|profile| profile["workload"] == preferred)
        .or_else(|| matching.first())
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "not_found", "unknown model"))?;
    serde_json::from_value(profile["workload"].clone()).map_err(|_| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "offering has no workload",
        )
    })
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SimulatorInputs {
    input_tokens: Option<u64>,
    planned_output: Option<u64>,
    #[serde(default)]
    upload_bytes: u64,
    #[serde(default)]
    cached_fraction: f64,
    #[serde(default)]
    image_megapixels: f64,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct ChatBody {
    model: String,
    messages: Vec<Value>,
    max_completion_tokens: Option<u64>,
    max_tokens: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    simulator: SimulatorInputs,
    n: Option<u64>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl ChatBody {
    fn validate(&self) -> Result<u64, ApiError> {
        let invalid =
            |message| ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", message);
        let max_output = self
            .max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(512);
        if !(1..=65_536).contains(&max_output) {
            return Err(invalid("max completion tokens must be between 1 and 65536"));
        }
        if self.messages.is_empty()
            || self.messages.iter().any(|message| {
                !message.is_object()
                    || !message["role"].is_string()
                    || !(message["content"].is_string()
                        || message["content"].is_array()
                        || message["content"].is_null())
            })
        {
            return Err(invalid(
                "messages must contain objects with a role and string, content-part array, or null content",
            ));
        }
        if self.n.is_some_and(|n| n != 1) {
            return Err(invalid("the simulator supports one completion per request"));
        }
        for name in self.extra.keys() {
            if !matches!(
                name.as_str(),
                "temperature"
                    | "top_p"
                    | "seed"
                    | "stop"
                    | "user"
                    | "frequency_penalty"
                    | "presence_penalty"
                    | "logprobs"
                    | "top_logprobs"
                    | "response_format"
                    | "metadata"
                    | "store"
                    | "reasoning_effort"
                    | "service_tier"
            ) {
                return Err(ApiError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "invalid_request",
                    format!("unsupported chat field: {name}"),
                ));
            }
        }
        Ok(max_output)
    }
}

async fn chat_completion(
    State(cloud): State<SharedCloud>,
    path: Result<Path<usize>, PathRejection>,
    body: Result<Json<ChatBody>, JsonRejection>,
) -> ApiResult {
    let Path(node) = path.map_err(ApiError::path)?;
    let Json(body) = body.map_err(ApiError::json)?;
    handle_chat(cloud, Some(node), body).await
}

async fn routed_chat_completion(
    State(cloud): State<SharedCloud>,
    body: Result<Json<ChatBody>, JsonRejection>,
) -> ApiResult {
    let Json(body) = body.map_err(ApiError::json)?;
    handle_chat(cloud, None, body).await
}

async fn handle_chat(cloud: SharedCloud, node: Option<usize>, body: ChatBody) -> ApiResult {
    let max_output = body.validate()?;
    let estimated_input = serde_json::to_vec(&body.messages)
        .expect("JSON values serialize")
        .len()
        .div_ceil(4)
        .max(1) as u64;
    let (id, created) = {
        let mut simulator = cloud.lock().await;
        let workload = match node {
            Some(node) => instance_profile(&simulator, node)?.1,
            None => {
                let vision = body.simulator.image_megapixels > 0.0
                    || body.messages.iter().any(|message| {
                        message["content"].as_array().is_some_and(|parts| {
                            parts.iter().any(|part| part["type"] == "image_url")
                        })
                    });
                catalog_workload(&simulator, &body.model, vision)?
            }
        };
        let spec = InferenceSpec {
            model: body.model.clone(),
            input_tokens: body.simulator.input_tokens.unwrap_or(estimated_input),
            planned_output: body.simulator.planned_output.unwrap_or(max_output),
            max_output,
            upload_bytes: body.simulator.upload_bytes,
            workload,
            image_megapixels: body.simulator.image_megapixels,
            adapter: None,
            class: "standard".into(),
            cached_fraction: body.simulator.cached_fraction,
        };
        let created = simulator.clock().floor() as u64;
        let id = match node {
            Some(node) => simulator.enqueue(node, spec),
            None => simulator.route(spec),
        }
        .map_err(ApiError::engine)?;
        (id, created)
    };
    if body.stream {
        let state = ChatStream {
            cloud,
            id,
            model: body.model,
            created,
            max_output,
            include_usage: body.stream_options.unwrap_or_default().include_usage,
            emitted: 0,
            role_sent: false,
            phase: StreamPhase::Tokens,
            usage: Value::Null,
        };
        let stream = futures_util::stream::unfold(state, next_chat_event);
        Ok(Sse::new(stream).into_response())
    } else {
        let receipt = wait_for_completion(&cloud, id).await?;
        let count = receipt["generated_tokens"].as_u64().unwrap_or(0);
        Ok(Json(json!({
            "id": format!("chatcmpl-sim-{id}"), "object": "chat.completion", "created": created,
            "model": body.model,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": synthetic_text(count) }, "finish_reason": finish_reason(count, max_output) }],
            "usage": openai_usage(&receipt["usage"]),
        })).into_response())
    }
}

async fn wait_for_completion(cloud: &SharedCloud, id: usize) -> Result<Value, ApiError> {
    loop {
        let receipt = cloud.lock().await.request(id).map_err(ApiError::engine)?;
        match receipt["status"].as_str() {
            Some("completed") => return Ok(receipt),
            Some("failed" | "cancelled") => {
                return Err(ApiError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "inference_failed",
                    "inference request failed or was cancelled",
                ));
            }
            _ => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    }
}

fn synthetic_text(tokens: u64) -> String {
    // Both completion handlers are bounded by the validated max_output ceiling.
    " tok".repeat(tokens.min(65_536) as usize)
}

fn finish_reason(tokens: u64, max_output: u64) -> &'static str {
    if tokens >= max_output {
        "length"
    } else {
        "stop"
    }
}

fn openai_usage(usage: &Value) -> Value {
    let prompt = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let completion = usage["completion_tokens"].as_u64().unwrap_or(0);
    json!({ "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt.saturating_add(completion) })
}

enum StreamPhase {
    Tokens,
    Usage,
    Done,
    End,
}

struct ChatStream {
    cloud: SharedCloud,
    id: usize,
    model: String,
    created: u64,
    max_output: u64,
    include_usage: bool,
    emitted: u64,
    role_sent: bool,
    phase: StreamPhase,
    usage: Value,
}

fn chat_chunk(state: &ChatStream, delta: Value, finish: Value) -> Value {
    json!({ "id": format!("chatcmpl-sim-{}", state.id), "object": "chat.completion.chunk", "created": state.created,
        "model": state.model, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })
}

async fn next_chat_event(
    mut state: ChatStream,
) -> Option<(Result<Event, std::convert::Infallible>, ChatStream)> {
    loop {
        match state.phase {
            StreamPhase::End => return None,
            StreamPhase::Done => {
                state.phase = StreamPhase::End;
                return Some((Ok(Event::default().data("[DONE]")), state));
            }
            StreamPhase::Usage => {
                let value = json!({ "id": format!("chatcmpl-sim-{}", state.id), "object": "chat.completion.chunk", "created": state.created,
                    "model": state.model, "choices": [], "usage": state.usage });
                state.phase = StreamPhase::Done;
                return Some((Ok(Event::default().data(value.to_string())), state));
            }
            StreamPhase::Tokens => {}
        }
        let receipt = state.cloud.lock().await.request(state.id);
        let receipt = match receipt {
            Ok(receipt) if !matches!(receipt["status"].as_str(), Some("failed" | "cancelled")) => {
                receipt
            }
            _ => {
                state.phase = StreamPhase::Done;
                return Some((Ok(Event::default().data(json!({ "error": { "code": "inference_failed", "message": "inference request failed or was cancelled" } }).to_string())), state));
            }
        };
        if !state.role_sent && receipt["first_token_at_s"].is_number() {
            state.role_sent = true;
            let chunk = chat_chunk(
                &state,
                json!({ "role": "assistant", "content": "" }),
                Value::Null,
            );
            return Some((Ok(Event::default().data(chunk.to_string())), state));
        }
        let generated = receipt["generated_tokens"]
            .as_u64()
            .unwrap_or(0)
            .min(state.max_output);
        if state.role_sent && generated > state.emitted {
            let count = (generated - state.emitted).min(128);
            state.emitted += count;
            let chunk = chat_chunk(
                &state,
                json!({ "content": synthetic_text(count) }),
                Value::Null,
            );
            return Some((Ok(Event::default().data(chunk.to_string())), state));
        }
        if receipt["status"] == "completed" {
            state.usage = openai_usage(&receipt["usage"]);
            state.phase = if state.include_usage {
                StreamPhase::Usage
            } else {
                StreamPhase::Done
            };
            let chunk = chat_chunk(
                &state,
                json!({}),
                json!(finish_reason(generated, state.max_output)),
            );
            return Some((Ok(Event::default().data(chunk.to_string())), state));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
