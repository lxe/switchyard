//! Local artifact tooling combines a cloud-independent archive library with object storage.
pub mod cli;
mod publication;
mod runtime_build;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Artifact(#[from] switchyard_artifacts::Error),
    #[error(transparent)]
    Storage(#[from] switchyard_provider::Error),
    #[error("local artifact operation failed")]
    Io(#[from] std::io::Error),
    #[error("artifact configuration is invalid")]
    Json(#[from] serde_json::Error),
    #[error("artifact tooling: {0}")]
    Config(String),
    #[error("artifact worker failed")]
    Task(#[from] tokio::task::JoinError),
}
pub type Result<T> = std::result::Result<T, Error>;
