//! Format admission and projection operations, without a serving loop or model arithmetic.
pub mod decode_plan;
pub mod gdn;
pub mod gguf;
pub mod projection;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("GGUF compatibility: {0}")]
    Invalid(String),
    #[error("Explicit resource limit: {0}")]
    Limit(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
