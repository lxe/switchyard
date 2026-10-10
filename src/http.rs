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
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use std::error::Error;
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::Mutex;

mod openai;
use openai::{
    add_endpoint, catalog_models, chat_completion, endpoint, models, routed_chat_completion,
};

type SharedCloud = Arc<Mutex<CloudSimulator>>;
type ApiResult = Result<Response, ApiError>;

#[derive(Clone, Default)]
struct Shutdown(Arc<AtomicBool>);

impl Shutdown {
    fn requested(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Construct the provider API without opening a socket or advancing time.
///
/// `GET /v1/instances` returns `{ "instances": [...] }`. A single instance or
/// request is returned directly, with the same numeric `id` as its creation
/// receipt. `POST /control/advance` accepts `{ "seconds": 1.0 }` and returns
/// `{ "time_s": 1.0 }`; advancing by zero is valid.
pub fn router(cloud: SharedCloud) -> Router {
    router_with_shutdown(cloud, Shutdown::default())
}

fn router_with_shutdown(cloud: SharedCloud, shutdown: Shutdown) -> Router {
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
        .layer(Extension(shutdown))
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
    let shutdown = Shutdown::default();
    let server = axum::serve(
        listener,
        router_with_shutdown(cloud.clone(), shutdown.clone()),
    )
    .with_graceful_shutdown(async move {
        let _ = tokio::signal::ctrl_c().await;
        shutdown.0.store(true, Ordering::Relaxed);
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
