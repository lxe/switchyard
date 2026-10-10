//! Synthetic OpenAI-compatible serving and streaming adapters.
use super::*;
use axum::response::sse::{Event, Sse};
use std::collections::BTreeMap;

pub(super) fn endpoint(id: usize) -> String {
    format!("/v1/instances/{id}/openai/v1")
}

pub(super) fn add_endpoint(instance: &mut Value) {
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

pub(super) async fn models(
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

pub(super) async fn catalog_models(State(cloud): State<SharedCloud>) -> Response {
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
    if matching.is_empty() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "unknown model",
        ));
    }
    if vision
        && !matching
            .iter()
            .any(|profile| profile["workload"] == "vision")
    {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "model has no configured vision offering",
        ));
    }
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
pub(super) struct ChatBody {
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
    fn vision_requested(&self) -> bool {
        self.simulator.image_megapixels > 0.0
            || self.messages.iter().any(|message| {
                message["content"]
                    .as_array()
                    .is_some_and(|parts| parts.iter().any(|part| part["type"] == "image_url"))
            })
    }

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

pub(super) async fn chat_completion(
    State(cloud): State<SharedCloud>,
    Extension(shutdown): Extension<Shutdown>,
    path: Result<Path<usize>, PathRejection>,
    body: Result<Json<ChatBody>, JsonRejection>,
) -> ApiResult {
    let Path(node) = path.map_err(ApiError::path)?;
    let Json(body) = body.map_err(ApiError::json)?;
    handle_chat(cloud, Some(node), body, shutdown).await
}

pub(super) async fn routed_chat_completion(
    State(cloud): State<SharedCloud>,
    Extension(shutdown): Extension<Shutdown>,
    body: Result<Json<ChatBody>, JsonRejection>,
) -> ApiResult {
    let Json(body) = body.map_err(ApiError::json)?;
    handle_chat(cloud, None, body, shutdown).await
}

async fn handle_chat(
    cloud: SharedCloud,
    node: Option<usize>,
    body: ChatBody,
    shutdown: Shutdown,
) -> ApiResult {
    let max_output = body.validate()?;
    let estimated_input = serde_json::to_vec(&body.messages)
        .expect("JSON values serialize")
        .len()
        .div_ceil(4)
        .max(1) as u64;
    let (id, created) = {
        let mut simulator = cloud.lock().await;
        let workload = match node {
            Some(node) => {
                let workload = instance_profile(&simulator, node)?.1;
                if body.vision_requested() && workload != Workload::Vision {
                    return Err(ApiError::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "invalid_request",
                        "instance has no configured vision offering",
                    ));
                }
                workload
            }
            None => catalog_workload(&simulator, &body.model, body.vision_requested())?,
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
            shutdown,
        };
        let stream = futures_util::stream::unfold(state, next_chat_event);
        Ok(Sse::new(stream).into_response())
    } else {
        let receipt = wait_for_completion(&cloud, id, &shutdown).await?;
        let count = receipt["generated_tokens"].as_u64().unwrap_or(0);
        Ok(Json(json!({
            "id": format!("chatcmpl-sim-{id}"), "object": "chat.completion", "created": created,
            "model": body.model,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": synthetic_text(count) }, "finish_reason": finish_reason(count, max_output) }],
            "usage": openai_usage(&receipt["usage"]),
        })).into_response())
    }
}

async fn wait_for_completion(
    cloud: &SharedCloud,
    id: usize,
    shutdown: &Shutdown,
) -> Result<Value, ApiError> {
    loop {
        if shutdown.requested() {
            return Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_shutdown",
                "server is shutting down",
            ));
        }
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
    shutdown: Shutdown,
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
        if state.shutdown.requested() {
            state.phase = StreamPhase::Done;
            return Some((Ok(Event::default().data(json!({ "error": { "code": "server_shutdown", "message": "server is shutting down" } }).to_string())), state));
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

#[cfg(test)]
mod shutdown_tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    async fn fixture(stream: bool) -> (SharedCloud, Shutdown, Response) {
        let profile = crate::scenarios::default_profiles().remove(0);
        let model = profile.model.clone();
        let profile_id = profile.id.clone();
        let cloud = Arc::new(Mutex::new(CloudSimulator::new(vec![profile], 7).unwrap()));
        let node = cloud.lock().await.provision(&profile_id).unwrap();
        let shutdown = Shutdown::default();
        let app = router_with_shutdown(cloud.clone(), shutdown.clone());
        let request = Request::builder().method("POST")
            .uri(format!("/v1/instances/{node}/openai/v1/chat/completions"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({"model": model, "messages": [{"role": "user", "content": "test"}], "max_tokens": 20, "stream": stream}).to_string())).unwrap();
        let response = tokio::spawn(app.oneshot(request));
        for _ in 0..100 {
            if cloud.lock().await.request(0).is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(cloud.lock().await.request(0).is_ok());
        shutdown.0.store(true, Ordering::Relaxed);
        let response = tokio::time::timeout(Duration::from_secs(1), response)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        (cloud, shutdown, response)
    }

    #[tokio::test]
    async fn manual_clock_json_wait_ends_when_server_shuts_down() {
        let (_cloud, _shutdown, response) = fixture(false).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert!(
            std::str::from_utf8(&body)
                .unwrap()
                .contains("server_shutdown")
        );
    }

    #[tokio::test]
    async fn manual_clock_stream_wait_ends_when_server_shuts_down() {
        let (_cloud, _shutdown, response) = fixture(true).await;
        let body =
            tokio::time::timeout(Duration::from_secs(1), to_bytes(response.into_body(), 4096))
                .await
                .unwrap()
                .unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("server_shutdown"));
        assert!(text.contains("[DONE]"));
    }
}
