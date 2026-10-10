//! Live scheduler runtime and provider experiments.
mod bootstrap;
pub mod cli;
pub mod config;
mod ownership;
pub mod planner;
pub mod runtime;
pub mod serving;
mod vllm;
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
}
pub type Result<T> = std::result::Result<T, Error>;
