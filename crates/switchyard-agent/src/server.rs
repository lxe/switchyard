//! Authenticated lifecycle observations and streaming proxy to a loopback inference engine.

use crate::{Error, Phase, Result, State, Ticket};
use axum::{
    Router,
    body::Body,
    extract::{Request, State as AxumState},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::{future::IntoFuture, sync::Arc, time::Duration};
use tokio::sync::RwLock;

mod admission;
mod cache_export;
mod engine;
mod startup;

#[derive(Clone)]
struct App {
    state: Arc<RwLock<State>>,
    client: reqwest::Client,
    authorization_hash: [u8; 32],
    engine_url: String,
    admission: admission::Admission,
    export: Option<Arc<cache_export::CacheExport>>,
}

fn routes(app: App) -> Router {
    Router::new()
        .route("/state", get(observe))
        .route("/probe", get(observe))
        .route(
            "/cache/export",
            get(cache_export::status).post(cache_export::start),
        )
        .route("/cache/export/chunks/{name}", get(cache_export::chunk))
        .fallback(proxy)
        .layer(axum::extract::DefaultBodyLimit::max(2 << 20))
        .layer(middleware::from_fn_with_state(app.clone(), authenticate))
        .with_state(app)
}

/// Bind the control plane before starting hydration and enforce the ticket's agent lifetime.
///
/// Expiry stops the local engine and HTTP server. The provisioning client remains responsible
/// for terminating the cloud pod, because stopping this process does not stop rental billing.
pub async fn serve(ticket: Ticket, token: String, bind: std::net::SocketAddr) -> Result<()> {
    serve_started(ticket, token, bind, std::time::Instant::now()).await
}

/// Preserve the process-start epoch when the caller downloads a ticket before serving.
pub async fn serve_started(
    ticket: Ticket,
    token: String,
    bind: std::net::SocketAddr,
    epoch: std::time::Instant,
) -> Result<()> {
    if ticket.lifetime_s == 0 || token.len() < 32 {
        return Err(Error::Manifest(
            "invalid agent lifetime or authentication token".into(),
        ));
    }
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let roots = crate::runtime::RootPaths::for_ticket(&ticket)?;
    let admission = admission::Admission::default();
    let (control, commands) = tokio::sync::mpsc::channel(1);
    let export = cache_export::CacheExport::new(
        &ticket,
        &roots,
        admission.clone(),
        control.clone(),
        state.clone(),
    )?;
    // Ordinary serving tickets still keep the engine owner's channel open until server exit.
    let _control = control;
    let app = App {
        state: state.clone(),
        client: reqwest::Client::builder()
            .user_agent("Switchyard/0.1")
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(ticket.lifetime_s))
            .build()
            .map_err(|_| Error::Transfer("proxy client initialization".into()))?,
        authorization_hash: Sha256::digest(format!("Bearer {token}").as_bytes()).into(),
        engine_url: "http://127.0.0.1:8000".into(),
        admission,
        export: export.clone(),
    };
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let lifetime = ticket.lifetime_s;
    let engine = tokio::spawn(async move {
        if let Err(error) = run_engine(&ticket, state.clone(), commands, roots, epoch).await {
            let mut st = state.write().await;
            st.phase = Phase::Failed;
            st.error = Some(error.to_string());
        }
    });
    let (shutdown, signal) = tokio::sync::oneshot::channel::<()>();
    let server = axum::serve(listener, routes(app))
        .with_graceful_shutdown(async {
            let _ = signal.await;
        })
        .into_future();
    tokio::pin!(server);
    let result = tokio::select! {
        result = &mut server => result.map_err(Error::from),
        _ = tokio::time::sleep(Duration::from_secs(lifetime)) => {
            if let Some(export) = &export { export.cancel().await; }
            engine.abort();
            let _ = shutdown.send(());
            // Streaming clients cannot keep a billed endpoint's local agent alive indefinitely.
            match tokio::time::timeout(Duration::from_secs(5), &mut server).await {
                Ok(result) => result.map_err(Error::from),
                Err(_) => Ok(()),
            }
        },
    };
    if let Some(export) = &export {
        export.cancel().await;
    }
    engine.abort();
    let _ = engine.await;
    result
}

async fn authenticate(AxumState(app): AxumState<App>, request: Request, next: Next) -> Response {
    let provided = request
        .headers()
        .get("authorization")
        .map(|h| h.as_bytes())
        .unwrap_or_default();
    let hash: [u8; 32] = Sha256::digest(provided).into();
    let difference = hash
        .iter()
        .zip(app.authorization_hash)
        .fold(0, |difference, (a, b)| difference | (a ^ b));
    if difference != 0 {
        return (StatusCode::UNAUTHORIZED, "authentication required").into_response();
    }
    next.run(request).await
}

async fn observe(AxumState(app): AxumState<App>) -> axum::Json<State> {
    axum::Json(app.state.read().await.clone())
}

async fn proxy(AxumState(app): AxumState<App>, request: Request) -> Response {
    let path = request.uri().path();
    if !(path.starts_with("/v1/") || matches!(path, "/tokenize" | "/metrics" | "/health")) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(admission) = app.admission.enter().await else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "inference admission is closed",
        )
            .into_response();
    };
    if !matches!(app.state.read().await.phase, Phase::Ready) {
        return (StatusCode::SERVICE_UNAVAILABLE, "engine is unavailable").into_response();
    }
    let path_and_query = request
        .uri()
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or(path)
        .to_owned();
    let method = request.method().clone();
    let content_type = request.headers().get("content-type").cloned();
    let body = match tokio::time::timeout(
        Duration::from_secs(30),
        axum::body::to_bytes(request.into_body(), 2 << 20),
    )
    .await
    {
        Ok(Ok(body)) => body,
        Ok(Err(_)) => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
        Err(_) => return StatusCode::REQUEST_TIMEOUT.into_response(),
    };
    let mut upstream = app
        .client
        .request(method, format!("{}{path_and_query}", app.engine_url))
        .body(body);
    if let Some(content_type) = content_type {
        upstream = upstream.header("content-type", content_type);
    }
    // The cloud-facing agent token is never forwarded to the inference engine.
    let response = match upstream.send().await {
        Ok(response) => response,
        Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
    };
    let mut builder = Response::builder().status(response.status());
    for header in ["content-type", "cache-control"] {
        if let Some(value) = response.headers().get(header) {
            builder = builder.header(header, value);
        }
    }
    builder = builder.header("x-accel-buffering", "no");
    let stream = futures_util::stream::unfold(
        (Box::pin(response.bytes_stream()), admission),
        |(mut stream, admission)| async move {
            stream.next().await.map(|piece| {
                (
                    piece.map_err(|_| std::io::Error::other("engine response stream interrupted")),
                    (stream, admission),
                )
            })
        },
    );
    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

async fn run_engine(
    ticket: &Ticket,
    state: Arc<RwLock<State>>,
    mut commands: tokio::sync::mpsc::Receiver<engine::Stop>,
    roots: crate::runtime::RootPaths,
    epoch: std::time::Instant,
) -> Result<()> {
    let mut process = startup::prepare(ticket, &roots, state.clone(), epoch).await?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|_| Error::Transfer("health client initialization".into()))?;
    crate::startup::milestone(
        &state,
        epoch,
        crate::startup::StartupEvent::EngineHealthPollingStarted,
    )
    .await;
    loop {
        if process.child.try_wait()?.is_some() {
            return Err(Error::Transfer(
                "inference engine exited before readiness".into(),
            ));
        }
        if client
            .get("http://127.0.0.1:8000/health")
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    crate::startup::milestone(
        &state,
        epoch,
        crate::startup::StartupEvent::EngineHealthReady,
    )
    .await;
    {
        let mut st = state.write().await;
        st.phase = Phase::Ready;
    }
    tokio::select! {
        status = process.child.wait() => {
            Err(Error::Transfer(format!("inference engine exited: {}", status?)))
        },
        command = commands.recv() => {
            let Some(command) = command else { return Err(Error::Transfer("engine control closed".into())); };
            let result = process.stop(command.timeout).await;
            let failed = result.is_err();
            let _ = command.reply.send(result);
            if failed { Err(Error::Transfer("engine shutdown failed".into())) } else { Ok(()) }
        },
    }
}

#[cfg(test)]
mod tests;
