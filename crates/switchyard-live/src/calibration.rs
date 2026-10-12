//! Serving deployment fingerprints, independent of provider prices and scheduler policy.
use crate::{
    Result,
    config::{ExecutionConfig, ModelConfig},
};
use sha2::{Digest, Sha256};

/// Changing weights, model revision, engine options (including precision), runtime
/// or static/learned cache compatibility invalidates old measurements.
pub fn binding(config: &ExecutionConfig, model: &ModelConfig, gpu: &str) -> Result<String> {
    let static_caches = model
        .caches
        .iter()
        .filter(|c| c.gpu == gpu)
        .collect::<Vec<_>>();
    let identity = serde_json::json!({
        "version":1, "model":model.id, "repository":model.repository,
        "revision":model.revision, "weights":model.artifact,
        "engine":model.engine, "gpu":gpu, "runtime":config.bootstrap.runtime,
        "static_caches":static_caches,
        "learned_cache_compatibility":config.bootstrap.native_cache_registry.as_ref().map(|r| &r.compatibility),
        "startup":config.bootstrap.startup,
    });
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&identity)?)
    ))
}
