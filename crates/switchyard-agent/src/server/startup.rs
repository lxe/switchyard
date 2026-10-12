//! Overlap dependency hydration and publish the engine's ready file.
//!
//! The agent creates the file named by `SWITCHYARD_READY_FILE` once the model
//! and runtime dependencies are published. An early-launched engine may do
//! weight-independent preparation and then wait for that file.
use super::engine;
use crate::{
    Error, Phase, Result, State, Ticket,
    runtime::RootPaths,
    startup::{StartupEvent, milestone},
};
use std::{path::Path, process::Stdio, sync::Arc, time::Instant};
use tokio::{
    process::Command,
    sync::{RwLock, Semaphore, oneshot},
};

/// Remove a ready file left by an earlier engine before any new engine can observe it.
pub(super) fn clear_ready(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Create the ready file atomically: a waiter sees either no file or the published one.
pub(super) fn publish_ready(path: &Path) -> Result<()> {
    let (Some(parent), Some(name)) = (path.parent(), path.file_name()) else {
        return Err(Error::Manifest("invalid engine ready file path".into()));
    };
    std::fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".{}.tmp", name.to_string_lossy()));
    std::fs::write(&temporary, b"")?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

pub(super) async fn prepare(
    ticket: &Ticket,
    roots: &RootPaths,
    state: Arc<RwLock<State>>,
    epoch: Instant,
) -> Result<engine::Process> {
    let resources = PreparationResources {
        limit: Arc::new(Semaphore::new(ticket.concurrency)),
        runtime_gate: None,
        model_gate: None,
    };
    prepare_with_resources(ticket, roots, state, epoch, resources).await
}

pub(super) struct PreparationResources {
    pub(super) limit: Arc<Semaphore>,
    pub(super) runtime_gate: Option<Arc<tokio::sync::Mutex<()>>>,
    pub(super) model_gate: Option<Arc<tokio::sync::Mutex<()>>>,
}

pub(super) async fn prepare_with_resources(
    ticket: &Ticket,
    roots: &RootPaths,
    state: Arc<RwLock<State>>,
    epoch: Instant,
    resources: PreparationResources,
) -> Result<engine::Process> {
    if ticket.source.extent().is_some() {
        return extents::prepare(ticket, roots, state, epoch).await;
    }
    clear_ready(&roots.ready)?;
    // This runs before either branch can transfer, including on a tiny Docker /dev/shm.
    crate::hydrate::preflight(ticket, &roots.models, &state).await?;
    crate::runtime::preflight(ticket, roots, &state).await?;
    let limit = resources.limit;
    let (ready, receiver) = oneshot::channel();
    let runtime = async {
        let _guard = match &resources.runtime_gate {
            Some(gate) => Some(gate.lock().await),
            None => None,
        };
        milestone(&state, epoch, StartupEvent::RuntimeHydrationStarted).await;
        if ticket.startup.early_launch {
            let (boot_tx, boot_rx) = oneshot::channel();
            let (dependencies_tx, dependencies_rx) = oneshot::channel();
            let dependencies = async {
                crate::runtime::command_staged_with_limit(
                    ticket,
                    roots,
                    state.clone(),
                    limit.clone(),
                    Some(boot_tx),
                )
                .await?;
                milestone(&state, epoch, StartupEvent::RuntimeHydrationFinished).await;
                let _ = dependencies_tx.send(());
                Ok::<_, Error>(())
            };
            let launch = async {
                let mut command = boot_rx
                    .await
                    .map_err(|_| Error::Transfer("boot runtime readiness cancelled".into()))?;
                milestone(&state, epoch, StartupEvent::RuntimeBootReady).await;
                command.stdin(Stdio::null());
                let mut process = engine::Process::spawn(&mut command)?;
                milestone(&state, epoch, StartupEvent::EngineSpawned).await;
                tokio::select! {
                    ready = async {
                        tokio::try_join!(
                            async { receiver.await.map_err(|_| Error::Transfer("model readiness gate cancelled".into())) },
                            async { dependencies_rx.await.map_err(|_| Error::Transfer("runtime dependencies readiness cancelled".into())) }
                        )?;
                        Ok::<_, Error>(())
                    } => ready?,
                    status = process.child.wait() => {
                        status?;
                        return Err(Error::Transfer("cooperative engine exited before readiness".into()));
                    }
                }
                publish_ready(&roots.ready)?;
                milestone(&state, epoch, StartupEvent::ModelGateReleased).await;
                milestone(&state, epoch, StartupEvent::EngineGateReleased).await;
                Ok::<_, Error>(process)
            };
            let ((), process) = tokio::try_join!(dependencies, launch)?;
            Ok::<_, Error>(Launch::Running(process))
        } else {
            let command = crate::runtime::command_preflighted_with_limit(
                ticket,
                roots,
                state.clone(),
                limit.clone(),
            )
            .await?;
            milestone(&state, epoch, StartupEvent::RuntimeHydrationFinished).await;
            let command = command.unwrap_or_else(|| {
                let mut command = Command::new("vllm");
                command.args(&ticket.engine_args);
                crate::runtime::child_environment(&mut command);
                command.env(crate::runtime::READY_FILE, &roots.ready);
                command
            });
            Ok(Launch::Prepared(command))
        }
    };
    let model = async {
        let _guard = match &resources.model_gate {
            Some(gate) => Some(gate.lock().await),
            None => None,
        };
        milestone(&state, epoch, StartupEvent::ModelHydrationStarted).await;
        crate::hydrate::hydrate_with_limit(ticket, &roots.models, state.clone(), limit.clone())
            .await?;
        milestone(&state, epoch, StartupEvent::ModelHydrationFinished).await;
        let _ = ready.send(());
        Ok::<_, Error>(())
    };
    let (launch, ()) = tokio::try_join!(runtime, model)?;
    state.write().await.phase = Phase::Initializing;
    match launch {
        Launch::Running(process) => Ok(process),
        Launch::Prepared(mut command) => {
            // Everything is published already, so a launch script that always waits proceeds.
            publish_ready(&roots.ready)?;
            let process = engine::Process::spawn(&mut command)?;
            milestone(&state, epoch, StartupEvent::EngineSpawned).await;
            Ok(process)
        }
    }
}

enum Launch {
    Prepared(Command),
    Running(engine::Process),
}
#[cfg(test)]
mod tests;

mod extents;
