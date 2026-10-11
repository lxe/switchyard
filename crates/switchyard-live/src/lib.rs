//! Live scheduler runtime and provider experiments.
pub mod artifacts;
pub mod bake;
mod bootstrap;
pub mod cli;
pub mod config;
pub mod model;
mod ownership;
pub mod planner;
pub mod runtime;
pub mod serving;
mod vllm;
pub mod workloads;
pub use runtime::{LiveRuntime, WorkCall};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Provider(#[from] switchyard_provider::Error),
    #[error("live configuration: {0}")]
    Config(String),
    #[error("live request failed: {0}")]
    Request(String),
    #[error("experiment limit: {0}")]
    Limit(String),
    #[error("local file operation failed")]
    Io(#[from] std::io::Error),
    #[error("JSON configuration or receipt is invalid")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Artifact(#[from] switchyard_artifacts::Error),
    #[error(transparent)]
    Publication(#[from] switchyard_artifact_tools::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
