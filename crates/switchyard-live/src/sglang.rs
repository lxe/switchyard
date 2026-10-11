//! SGLang-specific launch arguments. The runtime supplies its Python entry point.
use crate::config::{EngineConfig, ModelConfig};

pub(crate) fn args(engine: &EngineConfig, model: &ModelConfig, path: &str) -> Vec<String> {
    let mut args = vec![
        "--model-path".into(),
        path.into(),
        "--served-model-name".into(),
        model.id.clone(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        "8000".into(),
        "--context-length".into(),
        engine.max_model_len.to_string(),
        "--max-running-requests".into(),
        engine.max_num_seqs.to_string(),
        "--mem-fraction-static".into(),
        engine.gpu_memory_utilization.to_string(),
        "--dtype".into(),
        engine.dtype.clone(),
    ];
    if engine.language_model_only {
        // Qwen's native model honors the config field, while SGLang's CLI flag
        // currently admits only the Muse/Cosmos architecture families.
        if model.repository.starts_with("Qwen/Qwen3.5-") {
            args.extend([
                "--json-model-override-args".into(),
                "{\"language_model_only\":true}".into(),
            ]);
        } else {
            args.push("--language-model-only".into());
        }
    }
    args.extend(engine.extra_args.clone());
    args
}
