//! Pod-side hydration tickets and observable lifecycle state. Tickets contain expiring URLs.
pub mod export;
pub mod hydrate;
pub mod runtime;
pub mod server;
pub mod startup;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct Ticket {
    pub model: String,
    pub revision: String,
    pub manifest: Manifest,
    pub urls: BTreeMap<String, String>,
    pub engine_args: Vec<String>,
    pub concurrency: usize,
    pub stall_s: u64,
    pub lifetime_s: u64,
    /// Optional separately packed serving runtime. Legacy tickets use the image's vLLM.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<runtime::RuntimeTicket>,
    #[serde(default)]
    pub startup: startup::StartupPolicy,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub compress: String,
    pub chunks: Vec<Chunk>,
    pub total_size: u64,
    pub raw_size: u64,
    pub sha256: String,
    #[serde(default)]
    pub files: BTreeMap<String, FileInfo>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub name: String,
    pub size: u64,
    pub raw: u64,
    pub shard: usize,
    pub seq: usize,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub size: u64,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Booting,
    Hydrating,
    Initializing,
    Ready,
    /// Inference admission is closed while a stopped engine's caches are exported.
    Exporting,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub model: String,
    pub revision: String,
    pub phase: Phase,
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    pub hydration_seconds: f64,
    pub volume_bytes_per_s: Option<f64>,
    pub model_cached: bool,
    /// Image cache residency is not discoverable from inside a newly running container.
    pub image_cached: Option<bool>,
    pub error: Option<String>,
    /// Additional runtime/cache transfer bytes; legacy model counters remain separate.
    #[serde(default)]
    pub artifact_bytes_downloaded: u64,
    /// Actual wire bytes, including retries; verified transfer progress remains separate.
    #[serde(default)]
    pub artifact_network_bytes_downloaded: u64,
    #[serde(default)]
    pub artifact_bytes_total: u64,
    #[serde(default)]
    pub artifact_volume_bytes_per_s: Option<f64>,
    /// Received runtime/cache wire bytes per elapsed hydration second, including retries.
    #[serde(default)]
    pub artifact_network_bytes_per_s: Option<f64>,
    #[serde(default)]
    pub artifacts: Vec<runtime::ArtifactState>,
    /// True after health, tokenization, and a minimal nonstream inference succeed.
    #[serde(default)]
    pub inference_verified: bool,
    /// Trusted mode reports transferred data without claiming content verification.
    #[serde(default)]
    pub hydration_mode: switchyard_artifacts::HydrationMode,
    #[serde(default)]
    pub model_storage: Option<startup::ModelStorageObservation>,
    #[serde(default)]
    pub runtime_storage: Option<startup::ModelStorageObservation>,
    #[serde(default)]
    pub startup_milestones: Vec<startup::StartupMilestone>,
    /// Bounded legacy streaming diagnostics. Old observations omit this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_diagnostics: Option<hydrate::ModelDiagnostics>,
}
impl State {
    pub fn new(ticket: &Ticket) -> Self {
        Self {
            model: ticket.model.clone(),
            revision: ticket.revision.clone(),
            phase: Phase::Booting,
            bytes_downloaded: 0,
            bytes_total: ticket.manifest.total_size,
            hydration_seconds: 0.0,
            volume_bytes_per_s: None,
            model_cached: false,
            image_cached: None,
            error: None,
            artifact_bytes_downloaded: 0,
            artifact_network_bytes_downloaded: 0,
            artifact_bytes_total: ticket
                .runtime
                .as_ref()
                .map_or(0, runtime::RuntimeTicket::total_bytes),
            artifact_volume_bytes_per_s: None,
            artifact_network_bytes_per_s: None,
            artifacts: ticket
                .runtime
                .as_ref()
                .map_or_else(Vec::new, runtime::RuntimeTicket::initial_states),
            inference_verified: false,
            hydration_mode: ticket.startup.mode,
            model_storage: None,
            runtime_storage: None,
            startup_milestones: vec![],
            model_diagnostics: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Artifact(#[from] switchyard_artifacts::Error),
    #[error("invalid hydration manifest: {0}")]
    Manifest(String),
    #[error("hydration transfer failed: {0}")]
    Transfer(String),
    #[error("filesystem operation failed")]
    Io(#[from] std::io::Error),
    #[error("hydration task failed")]
    Task,
}
pub type Result<T> = std::result::Result<T, Error>;
