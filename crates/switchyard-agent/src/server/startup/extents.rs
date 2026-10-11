//! Metadata gate and live tensor bridge for an extent-backed model.
use super::{Stdio, engine};
use crate::{
    Error, Phase, Result, State, Ticket,
    runtime::RootPaths,
    startup::{StartupEvent, milestone},
};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use switchyard_artifacts::{
    Entry,
    extents::{ExtentHydration, ExtentOptions},
};
use switchyard_weights::{WeightBridge, write_vllm_bundle};
use tokio::{
    io::AsyncWriteExt,
    process::Command,
    sync::{RwLock, Semaphore, watch},
};

struct Sidecar {
    root: PathBuf,
}
impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn sidecar(root: &Path) -> Result<Arc<Sidecar>> {
    let mut random = [0_u8; 12];
    getrandom::fill(&mut random).map_err(|_| Error::Task)?;
    let suffix = random
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let root = root.join(format!(".model-loader-{suffix}"));
    fs::create_dir(&root)?;
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
    let owner = Arc::new(Sidecar { root });
    fs::create_dir(owner.root.join("metadata"))?;
    write_vllm_bundle(&owner.root.join("plugin"))?;
    Ok(owner)
}

async fn metadata(h: &ExtentHydration, owner: Arc<Sidecar>) -> Result<()> {
    crate::model::metadata_bytes(h.manifest())?;
    for (index, entry) in h.manifest().entries.iter().enumerate() {
        match entry {
            Entry::Directory { path, .. } => {
                fs::create_dir_all(owner.root.join("metadata").join(path))?
            }
            Entry::File { path, size, .. } if !path.ends_with(".safetensors") => {
                let ready = h.wait_range(index, 0, *size).await?;
                let destination = owner.root.join("metadata").join(path);
                let owner = owner.clone();
                tokio::task::spawn_blocking(move || -> Result<()> {
                    let _owner = owner;
                    let mut input = ready.open_file()?;
                    let mut output = fs::File::options()
                        .write(true)
                        .create_new(true)
                        .open(destination)?;
                    let mut buffer = [0_u8; 64 * 1024];
                    let mut remaining = ready.length;
                    while remaining != 0 {
                        let n = (remaining.min(buffer.len() as u64)) as usize;
                        input.read_exact(&mut buffer[..n])?;
                        output.write_all(&buffer[..n])?;
                        remaining -= n as u64;
                    }
                    Ok(())
                })
                .await
                .map_err(|_| Error::Task)??;
            }
            _ => {}
        }
    }
    Ok(())
}
pub(super) async fn prepare(
    ticket: &Ticket,
    roots: &RootPaths,
    state: Arc<RwLock<State>>,
    epoch: Instant,
) -> Result<engine::Process> {
    crate::hydrate::preflight(ticket, &roots.models, &state).await?;
    crate::runtime::preflight(ticket, roots, &state).await?;
    let (artifact, loader) = ticket
        .source
        .extent()
        .ok_or_else(|| Error::Manifest("missing extent model".into()))?;
    let runtime = ticket
        .runtime
        .as_ref()
        .ok_or_else(|| Error::Manifest("extent model needs a selected Python runtime".into()))?;
    loader.validate()?;
    let limit = Arc::new(Semaphore::new(ticket.concurrency));
    let owner = sidecar(&roots.models)?;
    let diagnostic = crate::transport::TransportDiagnostics::default();
    state.write().await.model_transport = Some(diagnostic.clone());
    let hydration = Arc::new(ExtentHydration::prepare(
        artifact.clone(),
        &roots.models.join("model"),
        ExtentOptions {
            mode: ticket.startup.mode,
            reuse_existing: true,
            observer: Some(diagnostic.observer()),
            priority_extents: switchyard_weights::priority_extents(&artifact.manifest),
            concurrency: ticket.concurrency,
            download_limit: Some(limit.clone()),
            request_timeout: Duration::from_secs(ticket.lifetime_s),
            ..ExtentOptions::default()
        },
    )?);
    let bridge = WeightBridge::bind(&owner.root.join("weights.sock"))?;
    let (tx, mut rx) = watch::channel(None);
    let network = async {
        milestone(&state, epoch, StartupEvent::ModelHydrationStarted).await;
        let report = hydration.run().await?;
        {
            let mut s = state.write().await;
            s.bytes_downloaded = report.downloaded_bytes;
            s.hydration_seconds = epoch.elapsed().as_secs_f64();
        }
        milestone(&state, epoch, StartupEvent::ModelHydrationFinished).await;
        Ok::<_, Error>(())
    };
    let loader_bridge = async {
        bridge
            .serve(
                &hydration,
                epoch,
                || {},
                move |p| {
                    tx.send_replace(Some(p));
                },
            )
            .await?;
        milestone(&state, epoch, StartupEvent::TensorLoadingFinished).await;
        Ok::<_, Error>(())
    };
    let (metadata_ready, ready_rx) = tokio::sync::oneshot::channel();
    let metadata_gate = async {
        metadata(&hydration, owner.clone()).await?;
        milestone(&state, epoch, StartupEvent::ModelMetadataReady).await;
        let _ = metadata_ready.send(());
        Ok::<_, Error>(())
    };
    let environment = async {
        milestone(&state, epoch, StartupEvent::RuntimeHydrationStarted).await;
        crate::runtime::command_preflighted_with_limit(ticket, roots, state.clone(), limit.clone())
            .await?;
        milestone(&state, epoch, StartupEvent::RuntimeHydrationFinished).await;
        let program = roots.runtime.join(&loader.python_program);
        let resolved = fs::canonicalize(&program)?;
        if !resolved.starts_with(fs::canonicalize(&roots.runtime)?) || !resolved.is_file() {
            return Err(Error::Manifest(
                "loader Python escapes selected runtime".into(),
            ));
        }
        let mut command = Command::new(program);
        crate::runtime::child_environment(&mut command);
        let plugin = owner.root.join("plugin");
        command.arg(plugin.join("serve.py"));
        let mut args = ticket.engine_args.clone();
        if args.first().map(String::as_str) != Some("serve") || args.len() < 2 {
            return Err(Error::Manifest(
                "vLLM extent source requires serve model arguments".into(),
            ));
        }
        args[1] = owner.root.join("metadata").to_string_lossy().into_owned();
        let bridge_config = serde_json::json!({
            "socket_path": bridge.path(),
            "timeout_seconds": ticket.lifetime_s,
            "cuda_stage": loader.cuda_stage,
            "pinned_bytes": loader.pinned_bytes,
        });
        args.extend([
            "--load-format".into(),
            "switchyard_extents".into(),
            "--model-loader-extra-config".into(),
            bridge_config.to_string(),
        ]);
        command
            .args(args)
            .env("SWITCHYARD_RUNTIME_ROOT", &roots.runtime);
        let cache_root = roots.caches.join(&runtime.artifact.manifest.id);
        command.env("SWITCHYARD_CACHE_ROOT", &cache_root);
        for (key, value) in &runtime.environment {
            command.env(
                key,
                value
                    .replace("{runtime_root}", &roots.runtime.to_string_lossy())
                    .replace("{cache_root}", &cache_root.to_string_lossy()),
            );
        }
        // Apply the owned loader environment after caller-supplied runtime flags.
        let inherited = runtime
            .environment
            .get("PYTHONPATH")
            .map(|s| s.replace("{runtime_root}", &roots.runtime.to_string_lossy()));
        let python_path = match inherited {
            Some(p) => format!("{}:{p}", plugin.display()),
            None => plugin.to_string_lossy().into_owned(),
        };
        command
            .env("PYTHONPATH", python_path)
            .env("VLLM_PLUGINS", "switchyard_extents");
        command.stdin(Stdio::piped());
        let mut process = engine::Process::spawn(&mut command)?;
        process.retain(owner.clone());
        milestone(&state, epoch, StartupEvent::EngineSpawned).await;
        tokio::select! {
            ready = ready_rx => ready.map_err(|_| Error::Transfer("metadata readiness gate cancelled".into()))?,
            status = process.child.wait() => {
                status?;
                return Err(Error::Transfer("engine exited before metadata readiness".into()));
            }
        }
        let mut stdin = process
            .child
            .stdin
            .take()
            .ok_or_else(|| Error::Transfer("metadata gate stdin unavailable".into()))?;
        stdin.write_all(b"switchyard-metadata-ready-v1\n").await?;
        stdin.shutdown().await?;
        Ok::<_, Error>(process)
    };
    let work = async { tokio::try_join!(network, loader_bridge, metadata_gate) };
    tokio::pin!(work);
    tokio::pin!(environment);
    let mut process = None;
    let mut completed = false;
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    loop {
        if completed && process.is_some() {
            break;
        }
        tokio::select! {
            result = &mut work, if !completed => {
                result?;
                completed = true;
            },
            result = &mut environment, if process.is_none() => {
                process = Some(result?);
            },
            _ = tick.tick() => {
                if let Some(process) = process.as_mut()
                    && process.child.try_wait()?.is_some()
                {
                    return Err(Error::Transfer("engine exited while receiving weights".into()));
                }
                let progress = hydration.progress();
                let mut snapshot = state.write().await;
                snapshot.bytes_downloaded = progress.received_bytes;
                snapshot.volume_bytes_per_s = Some(
                    progress.received_bytes as f64 / epoch.elapsed().as_secs_f64().max(0.001),
                );
                snapshot.weight_loading = rx.borrow_and_update().clone();
            }
        }
    }
    let process =
        process.ok_or_else(|| Error::Transfer("engine initialization cancelled".into()))?;
    {
        let mut s = state.write().await;
        s.weight_loading = rx.borrow_and_update().clone();
        s.phase = Phase::Initializing;
    }
    Ok(process)
}

#[cfg(test)]
mod tests;
