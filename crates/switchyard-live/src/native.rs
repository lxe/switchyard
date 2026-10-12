//! Native process arguments are independent of GPU provisioning and simulation.
use crate::config::{EngineConfig, ModelConfig};

pub(crate) fn args(engine: &EngineConfig, model: &ModelConfig, path: &str) -> Vec<String> {
    let bounds = engine
        .native
        .as_ref()
        .expect("validated native engine bounds");
    let mut config = serde_json::json!({
        "model":model.id, "model_root":path,
        "device":{"kind":"cuda", "ordinal":bounds.cuda_ordinal},
        "dtype":engine.dtype, "max_context":engine.max_model_len,
        "max_weight_bytes":bounds.max_weight_bytes,
        "max_batch_size":engine.max_num_seqs, "queue_capacity":bounds.queue_capacity,
        "batch_wait_ms":bounds.batch_wait_ms
    });
    if let Some(cache) = &bounds.cache {
        config["cache"] = serde_json::to_value(cache).expect("Serializable native cache bounds");
    }
    if let Some(lora) = &bounds.lora {
        config["lora"] = serde_json::to_value(lora).expect("Serializable native adapter bounds");
    }
    vec![
        "--config-json".into(),
        config.to_string(),
        "serve".into(),
        "--bind".into(),
        "127.0.0.1:8000".into(),
    ]
}
