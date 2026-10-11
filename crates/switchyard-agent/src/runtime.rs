//! Separately verified serving runtimes and optional compatible compiler-cache artifacts.
use crate::{Error, Phase, Result, State, Ticket};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use switchyard_artifacts::{ArtifactTicket, HydrationOptions, Phase as TransferPhase, Progress};
use tokio::{
    process::Command,
    sync::{RwLock, watch},
};

mod budget;
mod cache;
mod compatibility;
mod launch;
pub use budget::DependencyBudget;
pub(crate) use launch::command_staged_with_limit;

/// Optional per-runtime transfer controls, copied from checked fleet configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeHydrationOptions {
    /// Additional HTTP attempts; zero preserves failures during diagnostic reproductions.
    pub retries: usize,
}
impl Default for RuntimeHydrationOptions {
    fn default() -> Self {
        Self { retries: 2 }
    }
}
impl RuntimeHydrationOptions {
    pub fn validate(&self) -> Result<()> {
        if self.retries > 10 {
            return Err(Error::Manifest("invalid runtime retry limit".into()));
        }
        Ok(())
    }
}

/// URLs are temporary access credentials, so runtime and cache tickets omit Debug.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeTicket {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hydration: Option<RuntimeHydrationOptions>,
    pub artifact: ArtifactTicket,
    pub compatibility: BTreeMap<String, String>,
    /// Immutable sibling trees needed after the boot runtime is available.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<RuntimeComponent>,
    /// Executable relative to /opt/switchyard/runtime; internal symlinks are supported.
    pub program: String,
    /// Prefix arguments; the model ticket's engine_args follow these arguments.
    pub args: Vec<String>,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    #[serde(default)]
    pub caches: Vec<CacheTicket>,
    /// Optional one-shot export of a writable cache subtree after stopping the engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_export: Option<crate::export::CacheExportConfig>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheTicket {
    pub name: String,
    pub artifact: ArtifactTicket,
    pub compatibility: BTreeMap<String, String>,
}

/// An immutable dependency tree published beside the boot runtime after it is ready.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeComponent {
    pub name: String,
    pub artifact: ArtifactTicket,
    pub compatibility: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactPhase {
    Pending,
    Downloading,
    Extracting,
    Verifying,
    Ready,
    Failed,
}

/// Runtime/cache diagnostics contain identities and counters, never download locations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactState {
    pub name: String,
    pub id: String,
    pub phase: ArtifactPhase,
    pub downloaded_bytes: u64,
    /// Actual response-body bytes, including repeated retry transfers.
    #[serde(default)]
    pub network_bytes_downloaded: u64,
    /// Newly network-downloaded verified bytes, excluding chunk-cache hits and retries.
    #[serde(default)]
    pub network_verified_bytes: u64,
    pub total_bytes: u64,
    pub unpacked_bytes: u64,
    pub total_unpacked_bytes: u64,
    pub verified_chunks: usize,
    pub total_chunks: usize,
    pub cached: bool,
    pub chunk_cache_hits: usize,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reception: Option<switchyard_artifacts::ReceptionProgress>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<crate::transport::TransportDiagnostics>,
}

/// Production destinations are fixed; tests inject roots without writing under /opt.
pub(crate) struct RootPaths {
    pub(crate) runtime: PathBuf,
    pub(crate) caches: PathBuf,
    pub(crate) chunks: PathBuf,
    pub(crate) models: PathBuf,
}
impl RootPaths {
    pub(crate) fn components(&self) -> PathBuf {
        self.runtime
            .parent()
            .expect("runtime root has a parent")
            .join("components")
    }
    pub(crate) fn production() -> Self {
        Self {
            runtime: "/opt/switchyard/runtime".into(),
            caches: "/opt/switchyard/cache".into(),
            chunks: "/opt/switchyard/artifacts".into(),
            models: "/work/switchyard".into(),
        }
    }
    pub(crate) fn for_ticket(ticket: &Ticket) -> Result<Self> {
        ticket.startup.validate()?;
        let mut roots = Self::production();
        roots.models = ticket
            .startup
            .model_directory()
            .parent()
            .unwrap()
            .to_owned();
        let components = roots.components();
        for reserved in [&roots.runtime, &roots.caches, &roots.chunks, &components] {
            if roots.models.starts_with(reserved) || reserved.starts_with(&roots.models) {
                return Err(Error::Manifest(
                    "model root overlaps runtime storage".into(),
                ));
            }
        }
        Ok(roots)
    }
    #[cfg(test)]
    pub(crate) fn under(root: &Path) -> Self {
        Self {
            runtime: root.join("runtime"),
            caches: root.join("cache"),
            chunks: root.join("chunks"),
            models: root.join("models"),
        }
    }
}

fn component(value: &str) -> bool {
    !matches!(value, "" | "." | "..")
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
}
fn relative_program(value: &str) -> bool {
    !value.is_empty()
        && !value.contains(['\\', '\0'])
        && Path::new(value)
            .components()
            .all(|p| matches!(p, Component::Normal(_)))
}
fn environment_name(value: &str) -> bool {
    let mut chars = value.bytes();
    chars
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && chars.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}
fn sensitive_environment(value: &str) -> bool {
    let value = value.to_ascii_uppercase();
    value.starts_with("SWITCHYARD_")
        || value.ends_with("_TOKEN")
        || value.ends_with("_KEY")
        || value.contains("SECRET")
        || value.contains("PASSWORD")
        || value.ends_with("_CREDENTIAL")
}
fn total(ticket: &ArtifactTicket) -> u64 {
    ticket
        .manifest
        .chunks
        .iter()
        .fold(0, |sum, c| sum.saturating_add(c.size))
}

impl RuntimeTicket {
    pub(crate) fn total_bytes(&self) -> u64 {
        self.components
            .iter()
            .map(|c| &c.artifact)
            .chain(self.caches.iter().map(|c| &c.artifact))
            .fold(total(&self.artifact), |sum, artifact| {
                sum.saturating_add(total(artifact))
            })
    }
    pub(crate) fn initial_states(&self) -> Vec<ArtifactState> {
        std::iter::once(("runtime", &self.artifact))
            .chain(
                self.components
                    .iter()
                    .map(|c| (c.name.as_str(), &c.artifact)),
            )
            .chain(self.caches.iter().map(|c| (c.name.as_str(), &c.artifact)))
            .map(|(name, ticket)| ArtifactState {
                name: if component(name) {
                    name.into()
                } else {
                    "invalid-cache".into()
                },
                id: if ticket.manifest.id.len() == 64
                    && ticket.manifest.id.bytes().all(|b| b.is_ascii_hexdigit())
                {
                    ticket.manifest.id.clone()
                } else {
                    "invalid-artifact".into()
                },
                phase: ArtifactPhase::Pending,
                downloaded_bytes: 0,
                network_bytes_downloaded: 0,
                network_verified_bytes: 0,
                total_bytes: total(ticket),
                unpacked_bytes: 0,
                total_unpacked_bytes: ticket
                    .manifest
                    .chunks
                    .iter()
                    .fold(0, |sum, c| sum.saturating_add(c.unpacked_size)),
                verified_chunks: 0,
                total_chunks: ticket.manifest.chunks.len(),
                cached: false,
                chunk_cache_hits: 0,
                error: None,
                reception: None,
                transport: None,
            })
            .collect()
    }
    fn validate_launch(&self, mode: switchyard_artifacts::HydrationMode) -> Result<()> {
        if let Some(options) = self.hydration {
            options.validate()?;
        }
        if let Some(export) = &self.cache_export {
            export.validate(&self.artifact.manifest.id)?;
        }
        if !relative_program(&self.program)
            || self.args.iter().any(|a| a.contains('\0'))
            || self
                .environment
                .iter()
                .any(|(k, v)| !environment_name(k) || sensitive_environment(k) || v.contains('\0'))
        {
            return Err(Error::Manifest(
                "invalid runtime command or environment".into(),
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        if self.caches.len() + self.components.len() > 64 {
            return Err(Error::Manifest("too many runtime cache artifacts".into()));
        }
        for part in &self.components {
            if !component(&part.name)
                || matches!(part.name.as_str(), "runtime" | ".seeds")
                || !names.insert(&part.name)
            {
                return Err(Error::Manifest(
                    "invalid or duplicate runtime component name".into(),
                ));
            }
        }
        for cache in &self.caches {
            if !component(&cache.name)
                || matches!(cache.name.as_str(), "runtime" | ".seeds")
                || !names.insert(&cache.name)
            {
                return Err(Error::Manifest(
                    "invalid or duplicate runtime cache name".into(),
                ));
            }
        }
        let declared_executable = if mode == switchyard_artifacts::HydrationMode::TrustedStreaming {
            self.artifact
                .manifest
                .entries
                .iter()
                .find(|entry| entry.path() == self.program)
                .is_some_and(|entry| {
                    matches!(entry, switchyard_artifacts::Entry::Symlink { .. })
                        || matches!(entry, switchyard_artifacts::Entry::File { .. })
                            && entry.mode() & 0o100 != 0
                })
        } else {
            self.artifact
                .manifest
                .resolve_entry(&self.program)?
                .is_some_and(|entry| {
                    matches!(entry, switchyard_artifacts::Entry::File { .. })
                        && entry.mode() & 0o100 != 0
                })
        };
        if !declared_executable {
            return Err(Error::Manifest(
                "runtime executable is absent or not executable".into(),
            ));
        }
        Ok(())
    }
}

/// Check every compatibility label before transfer, then hydrate immutable runtime/cache trees.
#[cfg(test)]
pub(crate) async fn command(
    ticket: &Ticket,
    roots: &RootPaths,
    state: Arc<RwLock<State>>,
) -> Result<Option<Command>> {
    command_with_limit(
        ticket,
        roots,
        state,
        Arc::new(tokio::sync::Semaphore::new(ticket.concurrency)),
    )
    .await
}

#[cfg(test)]
pub(crate) async fn command_with_limit(
    ticket: &Ticket,
    roots: &RootPaths,
    state: Arc<RwLock<State>>,
    download_limit: Arc<tokio::sync::Semaphore>,
) -> Result<Option<Command>> {
    preflight(ticket, roots, &state).await?;
    command_preflighted_with_limit(ticket, roots, state, download_limit).await
}
pub(crate) async fn preflight(
    ticket: &Ticket,
    roots: &RootPaths,
    state: &RwLock<State>,
) -> Result<()> {
    let Some(runtime) = &ticket.runtime else {
        let required = state
            .read()
            .await
            .model_storage
            .as_ref()
            .map(|s| s.required_bytes);
        state.write().await.dependency_budget = Some(budget::check(ticket, roots, required)?);
        return Ok(());
    };
    runtime.validate_launch(ticket.startup.mode)?;
    if ticket.startup.mode == switchyard_artifacts::HydrationMode::TrustedStreaming
        && runtime.hydration.is_some_and(|h| h.retries != 0)
    {
        return Err(Error::Manifest(
            "trusted runtime streaming requires zero retries".into(),
        ));
    }
    if !(1..=64).contains(&ticket.concurrency) || ticket.stall_s == 0 || ticket.stall_s > 86400 {
        return Err(Error::Manifest("invalid artifact transfer limits".into()));
    }
    if let Err((index, error)) = validate_artifacts(runtime, &ticket.revision, ticket.startup.mode)
    {
        mark_failure(state, index, &error.to_string()).await;
        return Err(error);
    }
    compatibility::observe_and_check(runtime).await?;
    let required = runtime
        .artifact
        .manifest
        .entries
        .iter()
        .chain(
            runtime
                .components
                .iter()
                .flat_map(|part| part.artifact.manifest.entries.iter()),
        )
        .try_fold(0_u64, |sum, entry| {
            sum.checked_add(match entry {
                switchyard_artifacts::Entry::File { size, .. } => *size,
                _ => 0,
            })
        })
        .ok_or_else(|| Error::Manifest("runtime capacity overflow".into()))?;
    let observation = crate::startup::observe(roots.runtime.parent().unwrap(), required)?;
    state.write().await.runtime_storage = Some(observation.clone());
    crate::startup::check_capacity(&observation, false, 0)?;
    let required = state
        .read()
        .await
        .model_storage
        .as_ref()
        .map(|s| s.required_bytes);
    state.write().await.dependency_budget = Some(budget::check(ticket, roots, required)?);
    Ok(())
}
pub(crate) async fn command_preflighted_with_limit(
    ticket: &Ticket,
    roots: &RootPaths,
    state: Arc<RwLock<State>>,
    download_limit: Arc<tokio::sync::Semaphore>,
) -> Result<Option<Command>> {
    command_staged_with_limit(ticket, roots, state, download_limit, None).await
}

fn validate_artifacts(
    runtime: &RuntimeTicket,
    revision: &str,
    mode: switchyard_artifacts::HydrationMode,
) -> std::result::Result<(), (usize, Error)> {
    check_labels(&runtime.artifact.manifest, &runtime.compatibility, mode)
        .map_err(|error| (0, error))?;
    for (i, part) in runtime.components.iter().enumerate() {
        check_labels(&part.artifact.manifest, &part.compatibility, mode)
            .map_err(|error| (i + 1, error))?;
    }
    for (i, cache) in runtime.caches.iter().enumerate() {
        let i = i + runtime.components.len();
        check_labels(&cache.artifact.manifest, &cache.compatibility, mode)
            .map_err(|error| (i + 1, error))?;
        let metadata = &cache.artifact.manifest.metadata.compatibility;
        if metadata.get("runtime_id") != Some(&runtime.artifact.manifest.id) {
            return Err((
                i + 1,
                Error::Manifest("cache does not identify the selected runtime".into()),
            ));
        }
        if metadata
            .get("model_revision")
            .is_some_and(|value| value != revision)
        {
            return Err((
                i + 1,
                Error::Manifest("cache model revision differs from the selected model".into()),
            ));
        }
    }
    Ok(())
}

fn check_labels(
    manifest: &switchyard_artifacts::Manifest,
    required: &BTreeMap<String, String>,
    mode: switchyard_artifacts::HydrationMode,
) -> Result<()> {
    if mode == switchyard_artifacts::HydrationMode::Verified {
        return manifest.check_compatibility(required).map_err(Error::from);
    }
    // The trusted hydrator checks archive structure once. Compatibility needs only labels.
    for (key, value) in required {
        if manifest.metadata.compatibility.get(key) != Some(value) {
            return Err(Error::Manifest(format!(
                "artifact compatibility mismatch: {key}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn expand(value: &str, roots: &RootPaths, cache_root: &Path) -> String {
    value
        .replace("{runtime_root}", &roots.runtime.to_string_lossy())
        .replace("{cache_root}", &cache_root.to_string_lossy())
        .replace("{components_root}", &roots.components().to_string_lossy())
}

/// Preserve only host execution settings; all inherited cloud/storage credentials are removed.
pub(crate) fn child_environment(command: &mut Command) {
    const HOST: &[&str] = &[
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "OMP_NUM_THREADS",
    ];
    command.env_clear();
    for name in HOST {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("VLLM_NO_USAGE_STATS", "1");
}

async fn mark_failure(state: &RwLock<State>, index: usize, error: &str) {
    if let Some(artifact) = state.write().await.artifacts.get_mut(index) {
        artifact.phase = ArtifactPhase::Failed;
        artifact.error = Some(error.into());
    }
}

struct HydrationContext<'a> {
    roots: &'a RootPaths,
    ticket: &'a Ticket,
    state: Arc<RwLock<State>>,
    started: Instant,
    download_limit: Arc<tokio::sync::Semaphore>,
}

fn extraction_concurrency(ticket: &Ticket) -> usize {
    ticket.concurrency.min(
        if ticket.startup.mode == switchyard_artifacts::HydrationMode::TrustedStreaming {
            20
        } else {
            8
        },
    )
}

async fn hydrate_one(
    artifact: ArtifactTicket,
    destination: &Path,
    index: usize,
    context: &HydrationContext<'_>,
) -> Result<()> {
    let diagnostic = crate::transport::TransportDiagnostics::default();
    context.state.write().await.artifacts[index].transport = Some(diagnostic.clone());
    let options = HydrationOptions {
        // A bounded HTTP body feeds each decoder directly. No compressed reception spool.
        transport_observer: Some(diagnostic.observer()),
        mode: context.ticket.startup.mode,
        download_limit: Some(context.download_limit.clone()),
        cache_dir: context.roots.chunks.clone(),
        download_concurrency: context.ticket.concurrency,
        extraction_concurrency: extraction_concurrency(context.ticket),
        retries: context
            .ticket
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.hydration)
            .unwrap_or(RuntimeHydrationOptions {
                retries: if context.ticket.startup.mode
                    == switchyard_artifacts::HydrationMode::TrustedStreaming
                {
                    0
                } else {
                    2
                },
            })
            .retries,
        stall_timeout_s: context.ticket.stall_s,
        request_timeout_s: context.ticket.lifetime_s.clamp(1, 600),
        ..HydrationOptions::default()
    };
    let (tx, mut rx) = watch::channel::<Option<Progress>>(None);
    let hydration =
        switchyard_artifacts::hydrate(artifact, destination, options, move |progress| {
            tx.send_replace(Some(progress));
        });
    tokio::pin!(hydration);
    let mut observing = true;
    let result = loop {
        tokio::select! {
            biased;
            result = &mut hydration => break result,
            progress = rx.changed(), if observing => {
                observing = progress.is_ok();
                let progress = rx.borrow_and_update().clone();
                observe_progress(progress, index, context).await;
            }
        }
    };
    // Completion can win the select before its last watch update is consumed. In particular,
    // failed transfers must retain their final network/retry observations.
    let progress = rx.borrow_and_update().clone();
    observe_progress(progress, index, context).await;
    match result {
        Ok(report) => {
            let mut st = context.state.write().await;
            let item = &mut st.artifacts[index];
            item.phase = ArtifactPhase::Ready;
            item.downloaded_bytes = report.verified_bytes;
            item.network_bytes_downloaded = report.downloaded_bytes;
            item.network_verified_bytes = report.network_verified_bytes;
            item.unpacked_bytes = report.unpacked_bytes;
            item.verified_chunks =
                if context.ticket.startup.mode == switchyard_artifacts::HydrationMode::Verified {
                    item.total_chunks
                } else {
                    0
                };
            item.cached = report.already_present;
            item.chunk_cache_hits = report.cache_hits;
            item.reception = report.reception;
            totals(&mut st, context.started);
            Ok(())
        }
        Err(error) => {
            mark_failure(&context.state, index, &error.to_string()).await;
            Err(error.into())
        }
    }
}

async fn observe_progress(
    progress: Option<Progress>,
    index: usize,
    context: &HydrationContext<'_>,
) {
    let Some(progress) = progress else {
        return;
    };
    let mut state = context.state.write().await;
    let item = &mut state.artifacts[index];
    item.phase = match progress.phase {
        TransferPhase::Downloading => ArtifactPhase::Downloading,
        TransferPhase::Extracting => ArtifactPhase::Extracting,
        TransferPhase::Verifying => ArtifactPhase::Verifying,
        TransferPhase::Ready => ArtifactPhase::Ready,
    };
    item.downloaded_bytes = progress.verified_bytes;
    item.network_bytes_downloaded = progress.downloaded_bytes;
    item.network_verified_bytes = progress.network_verified_bytes;
    item.reception = progress.reception;
    item.total_bytes = progress.total_bytes;
    item.unpacked_bytes = progress.unpacked_bytes;
    item.total_unpacked_bytes = progress.total_unpacked_bytes;
    item.verified_chunks = progress.verified_chunks;
    item.total_chunks = progress.total_chunks;
    totals(&mut state, context.started);
}
fn totals(state: &mut State, started: Instant) {
    state.artifact_bytes_downloaded = state
        .artifacts
        .iter()
        .fold(0, |n, a| n.saturating_add(a.downloaded_bytes));
    state.artifact_network_bytes_downloaded = state
        .artifacts
        .iter()
        .fold(0, |n, a| n.saturating_add(a.network_bytes_downloaded));
    if state.artifact_network_bytes_downloaded > 0 {
        state.artifact_network_bytes_per_s = Some(
            state.artifact_network_bytes_downloaded as f64
                / started.elapsed().as_secs_f64().max(0.001),
        );
    }
    let network_verified = state
        .artifacts
        .iter()
        .fold(0u64, |n, a| n.saturating_add(a.network_verified_bytes));
    if network_verified > 0 {
        state.artifact_volume_bytes_per_s =
            Some(network_verified as f64 / started.elapsed().as_secs_f64().max(0.001));
    }
}

#[cfg(test)]
mod tests;
