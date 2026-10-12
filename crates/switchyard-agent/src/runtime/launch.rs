//! Boot publication and dependent artifact gates for cooperative serving processes.
use super::*;

/// Publish the complete boot tree and expose its command while dependencies hydrate.
pub(crate) async fn command_staged_with_limit(
    ticket: &Ticket,
    roots: &RootPaths,
    state: Arc<RwLock<State>>,
    download_limit: Arc<tokio::sync::Semaphore>,
    boot_ready: Option<tokio::sync::oneshot::Sender<Command>>,
) -> Result<Option<Command>> {
    let Some(runtime) = &ticket.runtime else {
        if let Some(sender) = boot_ready {
            let mut command = Command::new("vllm");
            command.args(&ticket.engine_args);
            child_environment(&mut command);
            command.env(READY_FILE, &roots.ready);
            sender
                .send(command)
                .map_err(|_| Error::Transfer("boot readiness receiver cancelled".into()))?;
        }
        return Ok(None);
    };
    state.write().await.phase = Phase::Hydrating;
    let started = Instant::now();
    let context = HydrationContext {
        roots,
        ticket,
        state: state.clone(),
        started,
        download_limit,
    };
    let cache_root = roots.caches.join(&runtime.artifact.manifest.id);
    cache::prepare_root(&cache_root)?;
    let (early_sender, final_sender) = if ticket.startup.early_launch {
        (boot_ready, None)
    } else {
        (None, boot_ready)
    };
    let boot = async {
        hydrate_one(runtime.artifact.clone(), &roots.runtime, 0, &context).await?;
        let command = prepare_command(ticket, roots)?;
        let command = if let Some(sender) = early_sender {
            sender
                .send(command)
                .map_err(|_| Error::Transfer("boot readiness receiver cancelled".into()))?;
            None
        } else {
            Some(command)
        };
        Ok::<_, Error>(command)
    };
    let components = async {
        for (index, part) in runtime.components.iter().enumerate() {
            hydrate_one(
                part.artifact.clone(),
                &roots.components().join(&part.name),
                index + 1,
                &context,
            )
            .await?;
        }
        Ok::<_, Error>(())
    };
    let caches = async {
        for (i, cache) in runtime.caches.iter().enumerate() {
            let index = runtime.components.len() + i + 1;
            let seed = roots.cache_seeds.as_ref().map_or_else(
                || cache_root.join(".seeds").join(&cache.name),
                |shared| shared.join(&cache.artifact.manifest.id),
            );
            hydrate_one(cache.artifact.clone(), &seed, index, &context).await?;
            let destination = cache_root.join(&cache.name);
            let manifest = cache.artifact.manifest.clone();
            if let Err(error) =
                cache::writable_copy_mode(seed, destination, manifest, ticket.startup.mode).await
            {
                mark_failure(&state, index, &error.to_string()).await;
                return Err(error);
            }
        }
        Ok::<_, Error>(())
    };
    let (command, (), ()) = tokio::try_join!(boot, components, caches)?;
    if let Some(sender) = final_sender {
        sender
            .send(command.ok_or_else(|| Error::Manifest("missing staged runtime command".into()))?)
            .map_err(|_| Error::Transfer("runtime readiness receiver cancelled".into()))?;
        Ok(None)
    } else {
        Ok(command)
    }
}

fn prepare_command(ticket: &Ticket, roots: &RootPaths) -> Result<Command> {
    let runtime = ticket
        .runtime
        .as_ref()
        .ok_or_else(|| Error::Manifest("missing boot runtime".into()))?;
    let cache_root = roots.caches.join(&runtime.artifact.manifest.id);
    let program = roots.runtime.join(&runtime.program);
    let resolved_program = std::fs::canonicalize(&program)?;
    if !resolved_program.starts_with(std::fs::canonicalize(&roots.runtime)?)
        || !resolved_program.is_file()
    {
        return Err(Error::Manifest(
            "runtime executable escapes the runtime directory".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if resolved_program.metadata()?.permissions().mode() & 0o111 == 0 {
            return Err(Error::Manifest(
                "runtime executable has no execute permission".into(),
            ));
        }
    }
    // Python discovers a virtual environment from the invoked symlink's directory.
    // Resolve it for containment checks, but preserve that path when executing it.
    let mut command = Command::new(program);
    command.args(&runtime.args).args(&ticket.engine_args);
    child_environment(&mut command);
    for (key, value) in &runtime.environment {
        command.env(key, expand(value, roots, &cache_root));
    }
    command
        .env("SWITCHYARD_RUNTIME_ROOT", &roots.runtime)
        .env("SWITCHYARD_COMPONENTS_ROOT", roots.components())
        .env(
            "SWITCHYARD_RUNTIME_ARTIFACT_ID",
            &runtime.artifact.manifest.id,
        )
        .env("SWITCHYARD_CACHE_ROOT", cache_root)
        .env(READY_FILE, &roots.ready);
    Ok(command)
}
