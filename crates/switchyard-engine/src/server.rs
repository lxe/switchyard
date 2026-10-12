//! HTTP transport and bounded request admission. GPU work has one owning thread.
use crate::{
    Error, Result,
    adapter_transport::{self, LoadRequest, PreparedAdapter},
    adapters::{AdapterHandle, EngineRegistry, RegistryError, SharedCatalog},
    config::Config,
    engine::{Engine, Generate, Generation},
    multimodal::{ChatMessage, Geometry, InferenceRequest, prepare_chat},
    telemetry::Telemetry,
    tokenizer::Tokenizer,
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    net::SocketAddr,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Semaphore, mpsc, oneshot};

struct Job {
    adapter: Arc<AdapterHandle>,
    requests: Vec<InferenceRequest>,
    reply: Reply,
}
enum OwnerCommand {
    Infer(Job),
    Load {
        name: String,
        prepared: PreparedAdapter,
        reply: oneshot::Sender<std::result::Result<Arc<AdapterHandle>, RegistryError>>,
    },
    Unload {
        name: Option<String>,
        id: Option<u64>,
        reply: oneshot::Sender<std::result::Result<(), RegistryError>>,
    },
}

enum Reply {
    Complete(oneshot::Sender<Result<Vec<Generation>>>),
    Stream {
        tx: mpsc::Sender<String>,
        chat: bool,
        id: String,
        created: u64,
        model: String,
        include_usage: bool,
    },
}
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn completion_id() -> String {
    format!(
        "sy-{}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}
fn created() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
impl Reply {
    fn closed(&self) -> bool {
        match self {
            Self::Complete(tx) => tx.is_closed(),
            Self::Stream { tx, .. } => tx.is_closed(),
        }
    }
    fn complete(self, output: Result<Vec<Generation>>) {
        match self {
            Self::Complete(tx) => {
                let _ = tx.send(output);
            }
            Self::Stream {
                tx,
                chat,
                id,
                created,
                model,
                include_usage,
            } => match output {
                Ok(output) => {
                    if include_usage {
                        let prompt: usize = output.iter().map(|x| x.prompt_tokens).sum();
                        let completion: usize = output.iter().map(|x| x.token_ids.len()).sum();
                        let object = if chat {
                            "chat.completion.chunk"
                        } else {
                            "text_completion"
                        };
                        let usage = json!({"id":id,"created":created,"object":object,
                                "model":model,"choices":[],"usage":{"prompt_tokens":prompt,
                                "completion_tokens":completion,"total_tokens":prompt+completion}});
                        let _ = tx.try_send(usage.to_string());
                    }
                    let _ = tx.try_send("[DONE]".into());
                }
                Err(e) => {
                    let _ = tx.try_send(json!({"error":{"message":e.to_string()}}).to_string());
                }
            },
        }
    }
}
#[derive(Clone)]
struct App {
    config: Arc<Config>,
    tokenizer: Arc<Tokenizer>,
    tx: mpsc::Sender<OwnerCommand>,
    catalog: SharedCatalog,
    upload_limit: Arc<Semaphore>,
    telemetry: Telemetry,
    vocab_size: usize,
    vision_geometry: Option<Geometry>,
}
struct ApiError(Error, StatusCode);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.1,
            Json(json!({"error":{"message":self.0.to_string(),"type":"invalid_request_error"}})),
        )
            .into_response()
    }
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    model: String,
    prompt: Prompt,
    max_tokens: usize,
    #[serde(default)]
    temperature: f64,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    stream_options: StreamOptions,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatTemplateKwargs {
    #[serde(default)]
    enable_thinking: bool,
}
impl ChatTemplateKwargs {
    fn validate(&self) -> std::result::Result<(), ApiError> {
        if self.enable_thinking {
            return Err(ApiError(
                Error::Config("Thinking mode is unsupported".into()),
                StatusCode::BAD_REQUEST,
            ));
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Chat {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    #[serde(default)]
    temperature: f64,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: StreamOptions,
    #[serde(default)]
    chat_template_kwargs: ChatTemplateKwargs,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tokenize {
    model: String,
    messages: Vec<ChatMessage>,
    add_generation_prompt: bool,
    #[serde(default)]
    chat_template_kwargs: ChatTemplateKwargs,
}
async fn tokenize(
    State(app): State<App>,
    Json(request): Json<Tokenize>,
) -> std::result::Result<Json<Value>, ApiError> {
    let _adapter = app.check(&request.model, 0., false)?;
    request.chat_template_kwargs.validate()?;
    if !request.add_generation_prompt {
        return Err(ApiError(
            Error::Config("Expected add_generation_prompt=true".into()),
            StatusCode::BAD_REQUEST,
        ));
    }
    let prepared = app.prepare_chat(&request.messages, 0)?;
    let tokens = prepared.generation.prompt;
    Ok(Json(json!({"count":tokens.len(), "tokens":tokens})))
}

impl App {
    fn prepare_chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: usize,
    ) -> std::result::Result<InferenceRequest, ApiError> {
        prepare_chat(
            &self.tokenizer,
            messages,
            self.config
                .vision
                .as_ref()
                .zip(self.vision_geometry.as_ref()),
            self.config.max_context,
            max_tokens,
        )
        .map_err(|e| ApiError(e, StatusCode::BAD_REQUEST))
    }
    fn check(
        &self,
        model: &str,
        temperature: f64,
        _stream: bool,
    ) -> std::result::Result<Arc<AdapterHandle>, ApiError> {
        if temperature != 0. {
            return Err(ApiError(
                Error::Config("Expected temperature=0".into()),
                StatusCode::BAD_REQUEST,
            ));
        }
        self.catalog
            .read()
            .map_err(|_| {
                ApiError(
                    Error::Config("Adapter catalog unavailable".into()),
                    StatusCode::SERVICE_UNAVAILABLE,
                )
            })?
            .get(model)
            .cloned()
            .ok_or_else(|| {
                ApiError(
                    Error::Config("Unknown model or adapter".into()),
                    StatusCode::NOT_FOUND,
                )
            })
    }
    async fn submit(
        &self,
        adapter: Arc<AdapterHandle>,
        requests: Vec<InferenceRequest>,
    ) -> std::result::Result<Vec<Generation>, ApiError> {
        self.validate(&requests)?;
        if requests.is_empty() || requests.len() > self.config.max_batch_size {
            return Err(ApiError(
                Error::Config("Request batch exceeds configured bounds".into()),
                StatusCode::BAD_REQUEST,
            ));
        }
        let (tx, rx) = oneshot::channel();
        self.tx
            .try_send(OwnerCommand::Infer(Job {
                adapter,
                requests,
                reply: Reply::Complete(tx),
            }))
            .map_err(|_| {
                ApiError(
                    Error::Config("Inference queue full or unavailable".into()),
                    StatusCode::TOO_MANY_REQUESTS,
                )
            })?;
        rx.await
            .map_err(|_| {
                ApiError(
                    Error::Config("Inference owner stopped".into()),
                    StatusCode::SERVICE_UNAVAILABLE,
                )
            })?
            .map_err(|e| ApiError(e, StatusCode::BAD_REQUEST))
    }
    fn stream(
        &self,
        adapter: Arc<AdapterHandle>,
        requests: Vec<InferenceRequest>,
        chat: bool,
        include_usage: bool,
    ) -> std::result::Result<Response, ApiError> {
        self.validate(&requests)?;
        // One event per token, optional usage, and DONE: the buffer cannot grow
        // beyond the admitted output budget, and GPU execution never waits on it.
        let capacity = requests
            .iter()
            .map(|r| r.generation.max_tokens)
            .sum::<usize>()
            + 1
            + usize::from(include_usage);
        let (tx, rx) = mpsc::channel(capacity);
        self.tx
            .try_send(OwnerCommand::Infer(Job {
                adapter: adapter.clone(),
                requests,
                reply: Reply::Stream {
                    tx,
                    chat,
                    id: completion_id(),
                    created: created(),
                    model: adapter.name.clone(),
                    include_usage,
                },
            }))
            .map_err(|_| {
                ApiError(
                    Error::Config("Inference queue full or unavailable".into()),
                    StatusCode::TOO_MANY_REQUESTS,
                )
            })?;
        let stream = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|data| {
                (
                    Ok::<_, std::convert::Infallible>(Event::default().data(data)),
                    rx,
                )
            })
        });
        Ok(Sse::new(stream).into_response())
    }
    fn validate(&self, requests: &[InferenceRequest]) -> std::result::Result<(), ApiError> {
        if requests.is_empty()
            || requests.len() > self.config.max_batch_size
            || requests.iter().any(|request| {
                let r = &request.generation;
                r.prompt.is_empty()
                    || r.prompt
                        .iter()
                        .any(|&token| token as usize >= self.vocab_size)
                    || r.max_tokens == 0
                    || r.prompt
                        .len()
                        .checked_add(r.max_tokens)
                        .is_none_or(|n| n > self.config.max_context)
            })
        {
            return Err(ApiError(
                Error::Config("Invalid context, output or batch bounds".into()),
                StatusCode::BAD_REQUEST,
            ));
        }
        Ok(())
    }
}
fn registry_error(error: RegistryError) -> ApiError {
    let status = match error {
        RegistryError::NotFound(_) => StatusCode::NOT_FOUND,
        RegistryError::InUse(_) | RegistryError::AlreadyLoaded(_) => StatusCode::CONFLICT,
        RegistryError::Invalid(_) => StatusCode::BAD_REQUEST,
    };
    ApiError(Error::Config(error.to_string()), status)
}
fn queue_error() -> ApiError {
    ApiError(
        Error::Config("Inference queue full or unavailable".into()),
        StatusCode::TOO_MANY_REQUESTS,
    )
}
async fn list_models(State(app): State<App>) -> std::result::Result<Json<Value>, ApiError> {
    let catalog = app.catalog.read().map_err(|_| {
        ApiError(
            Error::Config("Adapter catalog unavailable".into()),
            StatusCode::SERVICE_UNAVAILABLE,
        )
    })?;
    let models: Vec<Value> = catalog.values().map(|handle| json!({
        "id":handle.name,"object":"model","created":created(),"owned_by":"switchyard",
        "root":app.config.model,"parent":if handle.id==0 {Value::Null} else {json!(app.config.model)},
        "adapter_revision":if handle.id==0 {Value::Null} else {json!(handle.revision)},
        "lora_int_id":handle.id,"lora_rank":handle.rank
    })).collect();
    Ok(Json(json!({"object":"list","data":models})))
}
impl App {
    async fn install(
        &self,
        name: String,
        prepared: PreparedAdapter,
    ) -> std::result::Result<Arc<AdapterHandle>, ApiError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .try_send(OwnerCommand::Load {
                name,
                prepared,
                reply: tx,
            })
            .map_err(|_| queue_error())?;
        rx.await
            .map_err(|_| {
                ApiError(
                    Error::Config("Inference owner stopped".into()),
                    StatusCode::SERVICE_UNAVAILABLE,
                )
            })?
            .map_err(registry_error)
    }
    fn upload_permit(&self) -> std::result::Result<tokio::sync::OwnedSemaphorePermit, ApiError> {
        self.upload_limit.clone().try_acquire_owned().map_err(|_| {
            ApiError(
                Error::Config("An adapter upload or attachment is already active".into()),
                StatusCode::TOO_MANY_REQUESTS,
            )
        })
    }
}
async fn load_adapter(
    State(app): State<App>,
    Json(request): Json<LoadRequest>,
) -> std::result::Result<Json<Value>, ApiError> {
    let config = app.config.lora.as_ref().ok_or_else(|| {
        ApiError(
            Error::Config("LoRA serving disabled".into()),
            StatusCode::NOT_FOUND,
        )
    })?;
    let prepared = adapter_transport::load(&request, config, &app.telemetry, app.upload_permit()?)
        .await
        .map_err(|error| ApiError(error, StatusCode::BAD_REQUEST))?;
    let handle = app.install(request.lora_name, prepared).await?;
    Ok(Json(
        json!({"status":"loaded","lora_name":handle.name,"lora_int_id":handle.id,"revision":handle.revision,"rank":handle.rank}),
    ))
}
async fn upload_adapter(
    State(app): State<App>,
    Path(name): Path<String>,
    multipart: Multipart,
) -> std::result::Result<Json<Value>, ApiError> {
    let config = app.config.lora.as_ref().ok_or_else(|| {
        ApiError(
            Error::Config("LoRA serving disabled".into()),
            StatusCode::NOT_FOUND,
        )
    })?;
    let prepared = adapter_transport::upload(
        &name,
        multipart,
        config,
        &app.telemetry,
        app.upload_permit()?,
    )
    .await
    .map_err(|error| ApiError(error, StatusCode::BAD_REQUEST))?;
    let handle = app.install(name, prepared).await?;
    Ok(Json(
        json!({"status":"loaded","lora_name":handle.name,"lora_int_id":handle.id,"revision":handle.revision,"rank":handle.rank}),
    ))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UnloadAdapter {
    lora_name: Option<String>,
    lora_int_id: Option<u64>,
}
async fn unload_adapter(
    State(app): State<App>,
    Json(request): Json<UnloadAdapter>,
) -> std::result::Result<Json<Value>, ApiError> {
    if request.lora_name.is_some() == request.lora_int_id.is_some() {
        return Err(ApiError(
            Error::Config("Provide lora_name or lora_int_id".into()),
            StatusCode::BAD_REQUEST,
        ));
    }
    let name = request.lora_name;
    let id = request.lora_int_id;
    let (tx, rx) = oneshot::channel();
    app.tx
        .try_send(OwnerCommand::Unload {
            name: name.clone(),
            id,
            reply: tx,
        })
        .map_err(|_| queue_error())?;
    rx.await
        .map_err(|_| {
            ApiError(
                Error::Config("Inference owner stopped".into()),
                StatusCode::SERVICE_UNAVAILABLE,
            )
        })?
        .map_err(registry_error)?;
    Ok(Json(
        json!({"status":"unloaded","lora_name":name,"lora_int_id":id}),
    ))
}

async fn completions(
    State(app): State<App>,
    Json(request): Json<Completion>,
) -> std::result::Result<Response, ApiError> {
    let adapter = app.check(&request.model, request.temperature, request.stream)?;
    let prompts = match request.prompt {
        Prompt::Text(x) => vec![
            app.tokenizer
                .encode(&x)
                .map_err(|e| ApiError(e, StatusCode::BAD_REQUEST))?,
        ],
        Prompt::Tokens(x) => vec![x],
        Prompt::Batch(x) => x,
    };
    let requests: Vec<InferenceRequest> = prompts
        .into_iter()
        .map(|prompt| {
            Generate {
                prompt,
                max_tokens: request.max_tokens,
                ignore_eos: request.ignore_eos,
            }
            .into()
        })
        .collect();
    app.validate(&requests)?;
    if request.stream {
        return app.stream(
            adapter,
            requests,
            false,
            request.stream_options.include_usage,
        );
    }
    let output = app.submit(adapter, requests).await?;
    let choices: Vec<Value> = output
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let text = app.tokenizer.decode(&x.token_ids)?;
            Ok(json!({"index": i, "text": text, "token_ids": x.token_ids,
            "token_latency_s": x.token_latency_s, "finish_reason": x.finish_reason}))
        })
        .collect::<Result<_>>()
        .map_err(|e| ApiError(e, StatusCode::INTERNAL_SERVER_ERROR))?;
    let prompt: usize = output.iter().map(|x| x.prompt_tokens).sum();
    let completion: usize = output.iter().map(|x| x.token_ids.len()).sum();
    Ok(Json(
        json!({"id":completion_id(),"created":created(),"object":"text_completion","model":request.model,"choices":choices,"usage":{"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":prompt+completion}}),
    ).into_response())
}
async fn chat(
    State(app): State<App>,
    Json(request): Json<Chat>,
) -> std::result::Result<Response, ApiError> {
    let adapter = app.check(&request.model, request.temperature, request.stream)?;
    request.chat_template_kwargs.validate()?;
    let requests = vec![app.prepare_chat(&request.messages, request.max_tokens)?];
    app.validate(&requests)?;
    if request.stream {
        return app.stream(
            adapter,
            requests,
            true,
            request.stream_options.include_usage,
        );
    }
    let output = app.submit(adapter, requests).await?;
    let x = &output[0];
    let content = app
        .tokenizer
        .decode(&x.token_ids)
        .map_err(|e| ApiError(e, StatusCode::INTERNAL_SERVER_ERROR))?;
    Ok(Json(
        json!({"id":completion_id(),"created":created(),"object":"chat.completion","model":request.model,"choices":[{"index":0,"message":{"role":"assistant","content":content},"token_ids":x.token_ids,"finish_reason":x.finish_reason}],"usage":{"prompt_tokens":x.prompt_tokens,"completion_tokens":x.token_ids.len(),"total_tokens":x.prompt_tokens+x.token_ids.len()}}),
    ).into_response())
}
async fn generate(
    State(app): State<App>,
    Json(requests): Json<Vec<Generate>>,
) -> std::result::Result<Json<Vec<Generation>>, ApiError> {
    Ok(Json(
        app.submit(
            app.check(&app.config.model, 0., false)?,
            requests.into_iter().map(Into::into).collect(),
        )
        .await?,
    ))
}
fn run_owner(
    mut rx: mpsc::Receiver<OwnerCommand>,
    mut registry: EngineRegistry,
    owner_tokenizer: Arc<Tokenizer>,
    max_batch: usize,
    batch_wait: Duration,
    owner_telemetry: Telemetry,
) {
    let mut pending = None;
    loop {
        let Some(command) = pending.take().or_else(|| rx.blocking_recv()) else {
            break;
        };
        let first = match command {
            OwnerCommand::Infer(job) => job,
            OwnerCommand::Load {
                name,
                prepared,
                reply,
            } => {
                if !reply.is_closed() {
                    let _ = reply.send(registry.load(name, &prepared.adapter));
                }
                continue;
            }
            OwnerCommand::Unload { name, id, reply } => {
                if !reply.is_closed() {
                    let result = match (name, id) {
                        (Some(name), None) => registry.unload(&name),
                        (None, Some(id)) => registry.unload_id(id),
                        _ => Err(RegistryError::Invalid(Error::Config(
                            "Provide one adapter identity".into(),
                        ))),
                    };
                    let _ = reply.send(result);
                }
                continue;
            }
        };
        if first.reply.closed() {
            continue;
        }
        let deadline = Instant::now() + batch_wait;
        let adapter_id = first.adapter.id;
        let mut size = first.requests.len();
        let mut jobs = vec![first];
        while size < max_batch {
            match rx.try_recv() {
                Ok(OwnerCommand::Infer(job)) if job.reply.closed() => continue,
                Ok(OwnerCommand::Infer(job))
                    if size + job.requests.len() > max_batch || job.adapter.id != adapter_id =>
                {
                    pending = Some(OwnerCommand::Infer(job));
                    break;
                }
                Ok(OwnerCommand::Infer(job)) => {
                    size += job.requests.len();
                    jobs.push(job);
                }
                Ok(command) => {
                    pending = Some(command);
                    break;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => break,
                Err(mpsc::error::TryRecvError::Empty) => {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    std::thread::sleep((deadline - now).min(Duration::from_micros(200)));
                }
            }
        }
        let mut requests = Vec::with_capacity(size);
        let mut destinations = Vec::with_capacity(size);
        for (j, job) in jobs.iter().enumerate() {
            for (row, r) in job.requests.iter().enumerate() {
                requests.push(r.clone());
                destinations.push((j, row));
            }
        }
        owner_telemetry.event("queue_batch", json!({"jobs": jobs.len(), "sequences":size,"adapter_id":adapter_id,"model":jobs[0].adapter.name}));
        let mut decoders: Vec<_> = (0..size)
            .map(|_| owner_tokenizer.tokenizer.decode_stream(true))
            .collect();
        let mut first_token = vec![true; size];
        let mut stream_errors = vec![None; jobs.len()];
        let engine = match registry.engine(adapter_id) {
            Ok(engine) => engine,
            Err(error) => {
                for job in jobs {
                    job.reply.complete(Err(Error::Config(error.to_string())));
                }
                continue;
            }
        };
        let generated = engine.generate_requests_observed(&requests, |i, token, elapsed, finished| {
            let (job_index, row) = destinations[i];
            let reply = &jobs[job_index].reply;
            if reply.closed() || stream_errors[job_index].is_some() {
                return false;
            }
            if let Reply::Stream { tx, chat, id, created, model, .. } = reply {
                let text = match decoders[i].step(token) {
                    Ok(text) => text.unwrap_or_default(),
                    Err(error) => {
                        stream_errors[job_index] = Some(error.to_string());
                        return false;
                    }
                };
                let reason = finished.then_some(
                    if token == engine.eos_token() && !requests[i].generation.ignore_eos {
                        "stop"
                    } else {
                        "length"
                    }
                );
                let choice = if *chat {
                    let mut delta = json!({"content": text});
                    if first_token[i] {
                        delta["role"] = json!("assistant");
                    }
                    json!({"index": row, "delta": delta, "token_ids": [token], "finish_reason": reason})
                } else {
                    json!({"index": row, "text": text, "token_ids": [token], "finish_reason": reason})
                };
                first_token[i] = false;
                let object = if *chat { "chat.completion.chunk" } else { "text_completion" };
                let event = json!({"id": id, "created": created, "object": object,
                    "model": model, "choices": [choice], "token_elapsed_s": elapsed});
                return tx.try_send(event.to_string()).is_ok();
            }
            true
        });
        match generated {
            Ok(results) => {
                let mut results = results.into_iter();
                for (j, job) in jobs.into_iter().enumerate() {
                    let output = results.by_ref().take(job.requests.len()).collect();
                    job.reply.complete(match stream_errors[j].take() {
                        Some(error) => Err(Error::Config(error)),
                        None => Ok(output),
                    });
                }
            }
            Err(error) => {
                let message = error.to_string();
                for job in jobs {
                    job.reply.complete(Err(Error::Config(message.clone())));
                }
            }
        }
    }
}

pub async fn serve(
    engine: Engine,
    config: Config,
    tokenizer: Tokenizer,
    bind: SocketAddr,
    telemetry: Telemetry,
) -> Result<()> {
    if config.vision.is_some() && !engine.supports_vision() {
        return Err(Error::Config(
            "Configured vision model was not loaded".into(),
        ));
    }
    let vision_geometry = config
        .vision
        .as_ref()
        .map(|_| Geometry::load(&config.model_root))
        .transpose()?;
    let body_limit = config.vision.as_ref().map_or(1024 * 1024, |vision| {
        1024 * 1024 + vision.max_images * vision.max_encoded_bytes.div_ceil(3) * 4
    });
    let vocab_size = engine.vocab_size();
    let tokenizer = Arc::new(tokenizer);
    let (tx, rx) = mpsc::channel::<OwnerCommand>(config.queue_capacity);
    let catalog: SharedCatalog = Arc::new(RwLock::new(Default::default()));
    let registry = EngineRegistry::new(engine, catalog.clone(), config.clone())?;
    let max_batch = config.max_batch_size;
    let batch_wait = Duration::from_millis(config.batch_wait_ms);
    let owner_tokenizer = tokenizer.clone();
    let owner_telemetry = telemetry.clone();
    let owner = std::thread::Builder::new()
        .name("switchyard-inference".into())
        .spawn(move || {
            run_owner(
                rx,
                registry,
                owner_tokenizer,
                max_batch,
                batch_wait,
                owner_telemetry,
            )
        })?;
    let app = App {
        config: Arc::new(config),
        tokenizer,
        tx,
        vocab_size,
        vision_geometry,
        catalog,
        upload_limit: Arc::new(Semaphore::new(1)),
        telemetry: telemetry.clone(),
    };
    let mut router = Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat))
        .route("/generate", post(generate))
        .route("/tokenize", post(tokenize))
        .layer(DefaultBodyLimit::max(body_limit));
    if let Some(lora) = &app.config.lora {
        let uploads = Router::new()
            .route("/v1/adapters/{name}", post(upload_adapter))
            .layer(DefaultBodyLimit::max(
                lora.max_upload_bytes as usize + 1024 * 1024,
            ));
        router = router
            .route(
                "/v1/load_lora_adapter",
                post(load_adapter).layer(DefaultBodyLimit::max(1024 * 1024)),
            )
            .route(
                "/v1/unload_lora_adapter",
                post(unload_adapter).layer(DefaultBodyLimit::max(1024 * 1024)),
            )
            .merge(uploads);
    }
    let router = router.with_state(app);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    telemetry.event(
        "http_listening",
        json!({"address":listener.local_addr()?.to_string()}),
    );
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    owner
        .join()
        .map_err(|_| Error::Config("Inference owner panicked".into()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn streamed_usage_counts_generated_ids_and_precedes_done() {
        let (tx, mut rx) = mpsc::channel(2);
        Reply::Stream {
            tx,
            chat: true,
            id: "test".into(),
            created: 0,
            model: "fixture".into(),
            include_usage: true,
        }
        .complete(Ok(vec![Generation {
            token_ids: vec![1, 2, 3],
            token_latency_s: vec![0.; 3],
            elapsed_s: 0.,
            prompt_tokens: 7,
            finish_reason: "length".into(),
        }]));
        let usage: Value = serde_json::from_str(&rx.recv().await.unwrap()).unwrap();
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(
            usage["usage"],
            json!({"prompt_tokens":7,"completion_tokens":3,"total_tokens":10})
        );
        assert_eq!(rx.recv().await.unwrap(), "[DONE]");
        assert!(rx.recv().await.is_none());
    }

    #[test]
    fn router_options_accept_supported_template_and_reject_unknown_fields() {
        let mut request = json!({"model":"fixture","messages":[{"role":"user","content":"hi"}],
            "max_tokens":8,"stream":true,"stream_options":{"include_usage":true},
            "chat_template_kwargs":{"enable_thinking":false}});
        let decoded: Chat = serde_json::from_value(request.clone()).unwrap();
        assert!(decoded.stream_options.include_usage);
        assert!(decoded.chat_template_kwargs.validate().is_ok());
        request["chat_template_kwargs"]["enable_thinking"] = json!(true);
        let decoded: Chat = serde_json::from_value(request.clone()).unwrap();
        assert!(decoded.chat_template_kwargs.validate().is_err());
        request["stream_options"]["unknown"] = json!(true);
        assert!(serde_json::from_value::<Chat>(request).is_err());
    }
}
