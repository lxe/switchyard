//! Overlap dependency hydration and optionally release a cooperative engine's stdin gate.
use super::engine;
use crate::{
    Error, Phase, Result, State, Ticket,
    runtime::RootPaths,
    startup::{StartupEvent, milestone},
};
use std::{process::Stdio, sync::Arc, time::Instant};
use tokio::{
    io::AsyncWriteExt,
    process::Command,
    sync::{RwLock, Semaphore, oneshot},
};

const MODEL_READY: &[u8] = b"switchyard-model-ready-v1\n";

pub(super) async fn prepare(
    ticket: &Ticket,
    roots: &RootPaths,
    state: Arc<RwLock<State>>,
    epoch: Instant,
) -> Result<engine::Process> {
    if ticket.source.extent().is_some() {
        return extents::prepare(ticket, roots, state, epoch).await;
    }
    // This runs before either branch can transfer, including on a tiny Docker /dev/shm.
    crate::hydrate::preflight(ticket, &roots.models, &state).await?;
    crate::runtime::preflight(ticket, roots, &state).await?;
    let limit = Arc::new(Semaphore::new(ticket.concurrency));
    let (ready, receiver) = oneshot::channel();
    let runtime = async {
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
                command.stdin(Stdio::piped());
                let mut process = engine::Process::spawn(&mut command)?;
                milestone(&state, epoch, StartupEvent::EngineSpawned).await;
                // Retain stdin outside Child while monitoring exit: Child::wait closes its pipe.
                let mut stdin = process.child.stdin.take().ok_or_else(|| {
                    Error::Transfer("engine model gate stdin is unavailable".into())
                })?;
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
                stdin.write_all(MODEL_READY).await?;
                stdin.shutdown().await?;
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
                command
            });
            Ok(Launch::Prepared(command))
        }
    };
    let model = async {
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
async fn release(process: &mut engine::Process) -> Result<()> {
    if process.child.try_wait()?.is_some() {
        return Err(Error::Transfer(
            "cooperative engine exited before model readiness".into(),
        ));
    }
    let mut stdin = process
        .child
        .stdin
        .take()
        .ok_or_else(|| Error::Transfer("engine model gate stdin is unavailable".into()))?;
    stdin
        .write_all(MODEL_READY)
        .await
        .map_err(|_| Error::Transfer("engine model gate delivery failed".into()))?;
    stdin
        .shutdown()
        .await
        .map_err(|_| Error::Transfer("engine model gate shutdown failed".into()))?;
    Ok(())
}

#[cfg(test)]
mod tests;

mod extents;
