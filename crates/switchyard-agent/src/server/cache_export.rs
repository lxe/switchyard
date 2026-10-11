//! Ticket-selected export control and authenticated, manifest-scoped download handlers.
use super::{App, admission::Admission, engine};
use crate::{
    Error, Phase, Result, State, Ticket,
    export::{CacheExportConfig, ExportPhase, ExportState},
    runtime::RootPaths,
};
use axum::{
    body::Body,
    extract::{Path, Request, State as AxumState},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::{
    io::AsyncReadExt,
    sync::{Mutex, RwLock, oneshot},
    task::JoinHandle,
};

mod packing;

struct Job {
    status: ExportState,
    output: Option<Arc<packing::Output>>,
    task: Option<JoinHandle<()>>,
}
pub(super) struct CacheExport {
    config: CacheExportConfig,
    source: PathBuf,
    scratch: PathBuf,
    admission: Admission,
    engine: engine::Control,
    state: Arc<RwLock<State>>,
    job: Mutex<Job>,
}
impl CacheExport {
    pub(super) fn new(
        ticket: &Ticket,
        roots: &RootPaths,
        admission: Admission,
        engine: engine::Control,
        state: Arc<RwLock<State>>,
    ) -> Result<Option<Arc<Self>>> {
        let Some(runtime) = &ticket.runtime else {
            return Ok(None);
        };
        let Some(config) = &runtime.cache_export else {
            return Ok(None);
        };
        config.validate(&runtime.artifact.manifest.id)?;
        if config.compatibility.get("model_revision") != Some(&ticket.revision) {
            return Err(Error::Manifest(
                "cache export model revision differs from ticket".into(),
            ));
        }
        Ok(Some(Arc::new(Self {
            config: config.clone(),
            source: roots
                .caches
                .join(&runtime.artifact.manifest.id)
                .join(&config.name),
            scratch: roots.chunks.join("exports"),
            admission,
            engine,
            state,
            job: Mutex::new(Job {
                status: ExportState::default(),
                output: None,
                task: None,
            }),
        })))
    }
    pub(super) async fn cancel(&self) {
        let handle = self.job.lock().await.task.take();
        if let Some(handle) = handle {
            handle.abort();
            let _ = handle.await;
        }
        let mut job = self.job.lock().await;
        if matches!(
            job.status.phase,
            ExportPhase::Draining | ExportPhase::Stopping | ExportPhase::Packing
        ) {
            job.status.phase = ExportPhase::Failed;
            job.status.error = Some("cache export cancelled by agent shutdown".into());
            let mut state = self.state.write().await;
            state.phase = Phase::Failed;
            state.error = Some("cache export cancelled".into());
        }
    }
    async fn phase(&self, phase: ExportPhase) {
        self.job.lock().await.status.phase = phase;
    }
    async fn run(self: Arc<Self>) {
        let result = self.pack_after_stop().await;
        let mut job = self.job.lock().await;
        match result {
            Ok((output, manifest)) => {
                job.output = Some(Arc::new(output));
                job.status.manifest = Some(manifest);
                job.status.phase = ExportPhase::Ready;
            }
            Err(error) => {
                // Diagnostics name the failed operation and contain no engine body or paths.
                job.status.phase = ExportPhase::Failed;
                job.status.error = Some(error.to_string());
                let mut state = self.state.write().await;
                state.phase = Phase::Failed;
                state.error = Some("cache export failed".into());
            }
        }
    }
    async fn pack_after_stop(&self) -> Result<(packing::Output, switchyard_artifacts::Manifest)> {
        let _exclusive = tokio::time::timeout(
            Duration::from_secs(self.config.drain_timeout_s),
            self.admission.drain(),
        )
        .await
        .map_err(|_| Error::Transfer("cache export request drainage timed out".into()))?;
        self.phase(ExportPhase::Stopping).await;
        let (reply, result) = oneshot::channel();
        self.engine
            .send(engine::Stop {
                timeout: Duration::from_secs(self.config.shutdown_timeout_s),
                reply,
            })
            .await
            .map_err(|_| Error::Transfer("cache export engine owner unavailable".into()))?;
        tokio::time::timeout(
            Duration::from_secs(self.config.shutdown_timeout_s + 3),
            result,
        )
        .await
        .map_err(|_| Error::Transfer("cache export engine shutdown timed out".into()))?
        .map_err(|_| Error::Transfer("cache export engine shutdown interrupted".into()))??;
        self.state.write().await.inference_verified = false;
        self.phase(ExportPhase::Packing).await;
        packing::pack(
            self.source.clone(),
            self.scratch.clone(),
            self.config.compatibility.clone(),
        )
        .await
        .map_err(|_| Error::Transfer("cache export packing failed".into()))
    }
}

pub(super) async fn start(AxumState(app): AxumState<App>, request: Request) -> Response {
    let Some(export) = &app.export else {
        return StatusCode::NOT_FOUND.into_response();
    };
    // There is deliberately no HTTP configuration payload.
    match axum::body::to_bytes(request.into_body(), 0).await {
        Ok(body) if body.is_empty() => {}
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "cache export requires an empty body",
            )
                .into_response();
        }
    }
    let mut job = export.job.lock().await;
    if job.status.phase == ExportPhase::Idle {
        let mut state = export.state.write().await;
        if !matches!(state.phase, Phase::Ready) {
            return StatusCode::CONFLICT.into_response();
        }
        export.admission.close();
        state.phase = Phase::Exporting;
        job.status.phase = ExportPhase::Draining;
        job.task = Some(tokio::spawn(export.clone().run()));
    }
    let code = match job.status.phase {
        ExportPhase::Ready => StatusCode::OK,
        ExportPhase::Failed => StatusCode::CONFLICT,
        _ => StatusCode::ACCEPTED,
    };
    (code, axum::Json(job.status.clone())).into_response()
}

pub(super) async fn status(AxumState(app): AxumState<App>) -> Response {
    let Some(export) = &app.export else {
        return StatusCode::NOT_FOUND.into_response();
    };
    axum::Json(export.job.lock().await.status.clone()).into_response()
}

pub(super) async fn chunk(AxumState(app): AxumState<App>, Path(name): Path<String>) -> Response {
    let Some(export) = &app.export else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let (output, chunk) = {
        let job = export.job.lock().await;
        if job.status.phase != ExportPhase::Ready {
            return StatusCode::CONFLICT.into_response();
        }
        let Some(chunk) = job
            .status
            .manifest
            .as_ref()
            .and_then(|manifest| manifest.chunks.iter().find(|chunk| chunk.name == name))
        else {
            return StatusCode::NOT_FOUND.into_response();
        };
        (job.output.as_ref().unwrap().clone(), chunk.clone())
    };
    let file = match tokio::fs::File::open(output.chunks().join(&chunk.name)).await {
        Ok(file)
            if file
                .metadata()
                .await
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() == chunk.size) =>
        {
            file
        }
        _ => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    // Output ownership follows the streaming body, including cancellation/disconnection.
    let stream = futures_util::stream::try_unfold(
        (file, chunk.size, output),
        |(mut file, remaining, output)| async move {
            if remaining == 0 {
                return Ok(None);
            }
            let mut bytes = vec![0_u8; remaining.min(64 * 1024) as usize];
            let amount = file.read(&mut bytes).await?;
            if amount == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "artifact chunk stream interrupted",
                ));
            }
            bytes.truncate(amount);
            Ok(Some((bytes, (file, remaining - amount as u64, output))))
        },
    );
    Response::builder()
        .header("content-type", "application/octet-stream")
        .header("content-length", chunk.size)
        .header("x-artifact-sha256", chunk.sha256)
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[cfg(test)]
mod tests;
