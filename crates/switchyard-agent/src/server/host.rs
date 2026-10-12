//! A physical host owns several independent serving engines and one transfer budget.
use super::*;
use crate::host::HostTicket;
use axum::{extract::Path, routing::any};
use std::collections::BTreeMap;

#[derive(Clone)]
struct HostApp {
    plan: switchyard_core::topology::HostPlan,
    endpoints: BTreeMap<String, App>,
}

fn router(app: HostApp) -> Router {
    Router::new()
        .route("/state", get(observe_host))
        .route("/endpoints", get(observe_host))
        .route("/endpoints/{id}/state", get(observe_endpoint))
        .route("/endpoints/{id}/probe", get(observe_endpoint))
        .route(
            "/endpoints/{id}/cache/export",
            get(export_status).post(export_start),
        )
        .route(
            "/endpoints/{id}/cache/export/chunks/{name}",
            get(export_chunk),
        )
        .route("/endpoints/{id}/{*path}", any(proxy_endpoint))
        .layer(axum::extract::DefaultBodyLimit::max(2 << 20))
        .layer(middleware::from_fn_with_state(
            app.clone(),
            authenticate_host,
        ))
        .with_state(app)
}

/// Start a configured resident set. Inference routing remains the caller's responsibility.
pub async fn serve_started(
    ticket: HostTicket,
    token: String,
    bind: std::net::SocketAddr,
    epoch: std::time::Instant,
) -> Result<()> {
    ticket.validate()?;
    if token.len() < 32 {
        return Err(Error::Manifest("invalid host authentication token".into()));
    }
    // This is admission against a fixed plan, never automatic model/KV sizing.
    check_gpu_memory(ticket.plan.gpu_memory_bytes).await?;
    serve_at(ticket, token, bind, epoch, None).await
}

async fn check_gpu_memory(required: u64) -> Result<()> {
    let output = tokio::time::timeout(
        Duration::from_secs(5),
        tokio::process::Command::new("nvidia-smi")
            .args([
                "--id=0",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ])
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| Error::Manifest("host GPU memory admission timed out".into()))??;
    let actual = std::str::from_utf8(&output.stdout)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .and_then(|n| n.checked_mul(1024 * 1024));
    if !output.status.success() || actual.is_none_or(|n| n < required) {
        return Err(Error::Manifest(
            "allocated GPU does not support the fixed host plan".into(),
        ));
    }
    Ok(())
}

async fn serve_at(
    ticket: HostTicket,
    token: String,
    bind: std::net::SocketAddr,
    epoch: std::time::Instant,
    root: Option<std::path::PathBuf>,
) -> Result<()> {
    ticket.validate()?;
    let root = root.or_else(|| ticket.storage_root.as_ref().map(std::path::PathBuf::from));
    let lifetime = ticket.endpoints[0].ticket.lifetime_s;
    let limit = Arc::new(tokio::sync::Semaphore::new(
        ticket.endpoints[0].ticket.concurrency,
    ));
    let runtime_gate = Arc::new(tokio::sync::Mutex::new(()));
    let mut model_gates = BTreeMap::<String, Arc<tokio::sync::Mutex<()>>>::new();
    let mut endpoints = BTreeMap::new();
    let mut tasks = tokio::task::JoinSet::new();
    let mut controls = Vec::new();
    for endpoint in ticket.endpoints {
        let model_gate = model_gates
            .entry(crate::host::model_identity(&endpoint.ticket))
            .or_default()
            .clone();
        let mut roots = crate::runtime::RootPaths::for_ticket(&endpoint.ticket)?;
        if let Some(root) = &root {
            roots.runtime = root.join("runtime");
            roots.chunks = root.join("chunks");
            roots.caches = root.join("cache");
        }
        // CUDA/compiler writes are private; immutable seeds are shared by artifact identity.
        // Runtime/components use the publication gate; weight replicas share a model gate.
        roots.cache_seeds = Some(roots.caches.join(".host-seeds"));
        // Endpoints share one runtime root, so each engine gets its own ready file.
        roots.ready = roots
            .runtime
            .with_file_name(format!("engine-ready-{}", endpoint.id));
        roots.caches = roots.caches.join("endpoints").join(&endpoint.id);
        let state = Arc::new(RwLock::new(State::new(&endpoint.ticket)));
        let admission = admission::Admission::default();
        let (control, commands) = tokio::sync::mpsc::channel(1);
        let export = cache_export::CacheExport::new(
            &endpoint.ticket,
            &roots,
            admission.clone(),
            control.clone(),
            state.clone(),
        )?;
        controls.push(control);
        let app = App {
            state: state.clone(),
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(3))
                .timeout(Duration::from_secs(lifetime))
                .build()
                .map_err(|_| Error::Transfer("host proxy client initialization".into()))?,
            authorization_hash: Sha256::digest(format!("Bearer {token}").as_bytes()).into(),
            engine_url: format!("http://127.0.0.1:{}", endpoint.port),
            admission,
            export,
        };
        let engine_url = app.engine_url.clone();
        endpoints.insert(endpoint.id, app);
        let resources = startup::PreparationResources {
            limit: limit.clone(),
            runtime_gate: Some(runtime_gate.clone()),
            model_gate: Some(model_gate),
        };
        tasks.spawn(async move {
            let result = async {
                let process = startup::prepare_with_resources(
                    &endpoint.ticket,
                    &roots,
                    state.clone(),
                    epoch,
                    resources,
                )
                .await?;
                monitor_engine(process, state.clone(), commands, &engine_url, epoch).await
            }
            .await;
            if let Err(error) = result {
                let mut state = state.write().await;
                state.phase = Phase::Failed;
                state.error = Some(error.to_string());
            }
        });
    }
    let app = HostApp {
        plan: ticket.plan,
        endpoints,
    };
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let (shutdown, signal) = tokio::sync::oneshot::channel::<()>();
    let server = axum::serve(listener, router(app.clone()))
        .with_graceful_shutdown(async {
            let _ = signal.await;
        })
        .into_future();
    tokio::pin!(server);
    let result = tokio::select! {
        result = &mut server => result.map_err(Error::from),
        _ = async { tokio::select! { _ = tokio::time::sleep(Duration::from_secs(lifetime)) => (), _ = shutdown_requested() => () } } => {
            for endpoint in app.endpoints.values() { if let Some(export) = &endpoint.export { export.cancel().await; } }
            tasks.abort_all();
            while tasks.join_next().await.is_some() {}
            let _ = shutdown.send(());
            match tokio::time::timeout(Duration::from_secs(5), &mut server).await { Ok(result) => result.map_err(Error::from), Err(_) => Ok(()) }
        }
    };
    for endpoint in app.endpoints.values() {
        if let Some(export) = &endpoint.export {
            export.cancel().await;
        }
    }
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    drop(controls);
    result
}

async fn authenticate_host(
    AxumState(app): AxumState<HostApp>,
    request: Request,
    next: Next,
) -> Response {
    authenticate(
        AxumState(
            app.endpoints
                .values()
                .next()
                .expect("validated host")
                .clone(),
        ),
        request,
        next,
    )
    .await
}
async fn observe_host(AxumState(app): AxumState<HostApp>) -> axum::Json<serde_json::Value> {
    let mut endpoints = vec![];
    for (id, endpoint) in &app.endpoints {
        endpoints.push(serde_json::json!({"id":id, "base_url":format!("/endpoints/{id}"), "state":endpoint.state.read().await.clone()}));
    }
    axum::Json(serde_json::json!({"plan":app.plan, "endpoints":endpoints}))
}
async fn observe_endpoint(AxumState(app): AxumState<HostApp>, Path(id): Path<String>) -> Response {
    match app.endpoints.get(&id) {
        Some(endpoint) => observe(AxumState(endpoint.clone())).await.into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
async fn proxy_endpoint(
    AxumState(app): AxumState<HostApp>,
    Path((id, path)): Path<(String, String)>,
    mut request: Request,
) -> Response {
    let Some(endpoint) = app.endpoints.get(&id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let query = request
        .uri()
        .query()
        .map(|q| format!("?{q}"))
        .unwrap_or_default();
    let Ok(uri) = format!("/{path}{query}").parse() else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    *request.uri_mut() = uri;
    proxy(AxumState(endpoint.clone()), request).await
}
async fn export_status(AxumState(app): AxumState<HostApp>, Path(id): Path<String>) -> Response {
    match app.endpoints.get(&id) {
        Some(endpoint) => cache_export::status(AxumState(endpoint.clone())).await,
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
async fn export_start(
    AxumState(app): AxumState<HostApp>,
    Path(id): Path<String>,
    request: Request,
) -> Response {
    match app.endpoints.get(&id) {
        Some(endpoint) => cache_export::start(AxumState(endpoint.clone()), request).await,
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
async fn export_chunk(
    AxumState(app): AxumState<HostApp>,
    Path((id, name)): Path<(String, String)>,
) -> Response {
    match app.endpoints.get(&id) {
        Some(endpoint) => cache_export::chunk(AxumState(endpoint.clone()), Path(name)).await,
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests;
