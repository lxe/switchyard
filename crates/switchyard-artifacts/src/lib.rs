//! Deterministic archives and verified hydration, independent of storage or serving providers.
//!
//! [`pack`] creates independently compressed tar chunks and a content-addressed
//! [`Manifest`]. [`hydrate`] downloads verified chunks into a reusable cache,
//! extracts into a private staging tree, verifies its entries, then publishes it
//! by renaming the complete directory. Existing destinations are never overlaid.
mod extract;
mod hydrate;
mod manifest;
mod pack;
mod paths;

pub use extract::verify;
pub use hydrate::{HydrationOptions, HydrationReport, Phase, Progress, hydrate};
pub use manifest::{ArtifactTicket, Chunk, Entry, Manifest, Metadata};
pub use pack::{PackOptions, pack};

/// Errors intentionally exclude signed URLs, request bodies, and metadata values.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid artifact: {0}")]
    Invalid(String),
    #[error("artifact content verification failed: {0}")]
    Integrity(String),
    #[error("artifact filesystem operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("artifact serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("artifact download failed (HTTP status {0})")]
    Http(u16),
    #[error("artifact transfer failed")]
    Transfer,
    #[error("artifact worker failed")]
    Worker,
    #[error("artifact hydration was cancelled")]
    Cancelled,
    #[error("artifact destination already exists with different contents")]
    DestinationExists,
    #[error("artifact compatibility requirement does not match: {0}")]
    Compatibility(String),
}

pub type Result<T> = std::result::Result<T, Error>;
