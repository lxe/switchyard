//! vLLM-specific process arguments stay outside the cloud and simulator contracts.
use crate::config::{EngineConfig, EngineKind, ModelConfig};
impl EngineConfig {
    pub fn args(&self, model: &ModelConfig, path: &str) -> Vec<String> {
        if self.kind == EngineKind::Sglang {
            return crate::sglang::args(self, model, path);
        }
        let mut args = vec![
            "serve".into(),
            path.into(),
            "--served-model-name".into(),
            model.id.clone(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            "8000".into(),
            "--max-model-len".into(),
            self.max_model_len.to_string(),
            "--max-num-seqs".into(),
            self.max_num_seqs.to_string(),
            "--gpu-memory-utilization".into(),
            self.gpu_memory_utilization.to_string(),
            "--dtype".into(),
            self.dtype.clone(),
        ];
        if self.language_model_only {
            args.push("--language-model-only".into());
        }
        args.extend(self.extra_args.clone());
        args
    }
}
