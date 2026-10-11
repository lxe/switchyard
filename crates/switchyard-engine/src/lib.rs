//! A bounded inference engine independent of GPU-cloud provisioning and hydration.
pub mod config;
pub mod engine;
pub mod models;
pub mod server;
pub mod telemetry;
pub mod tokenizer;
pub mod weights;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Config(String),
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
