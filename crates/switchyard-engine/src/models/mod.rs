//! Model adapters share immutable weights; each batch owns its sequence state.
pub mod qwen35;
use crate::weights::Weights;
use crate::{
    Error, Result,
    config::{Config, DeviceConfig, Precision},
    telemetry::Telemetry,
};
use candle_core::{DType, Device, Tensor};
pub type LayerTrace = Vec<(String, Tensor)>;

/// Execution contract for one cohort of equal-length token sequences.
/// A new session must have empty state and may share immutable weight storage.
pub trait ModelSession: Send {
    fn forward(&mut self, tokens: &Tensor, position: usize) -> Result<Tensor>;
}
pub trait ModelFactory: Send {
    fn session(&self) -> Box<dyn ModelSession>;
    fn device(&self) -> &Device;
    fn vocab_size(&self) -> usize;
    fn eos_token(&self) -> u32;
    /// Offline layer inspection; unsupported adapters need not implement it.
    fn inspect_layers(&self, _: &Tensor) -> Result<LayerTrace> {
        Err(Error::Config(
            "Layer inspection unsupported by this adapter".into(),
        ))
    }
}

struct QwenSession {
    model: qwen35::ModelForCausalLM,
    position: usize,
    rows: Option<usize>,
}
impl ModelSession for QwenSession {
    fn forward(&mut self, tokens: &Tensor, position: usize) -> Result<Tensor> {
        let (rows, length) = tokens.dims2()?;
        if length == 0
            || position != self.position
            || self.rows.is_some_and(|n| n != rows)
            || (position > 0 && length != 1)
        {
            return Err(Error::Config(
                "Session requires one prefill then contiguous single-token decode with fixed rows"
                    .into(),
            ));
        }
        let logits = self.model.forward(tokens, position)?.squeeze(1)?;
        self.position += length;
        self.rows = Some(rows);
        Ok(logits)
    }
}
struct QwenFactory {
    weights: qwen35::ModelForCausalLM,
    device: Device,
    vocab: usize,
    eos: u32,
}
impl ModelFactory for QwenFactory {
    fn session(&self) -> Box<dyn ModelSession> {
        Box::new(QwenSession {
            model: self.weights.clone(),
            position: 0,
            rows: None,
        })
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn eos_token(&self) -> u32 {
        self.eos
    }
    fn inspect_layers(&self, tokens: &Tensor) -> Result<LayerTrace> {
        let mut model = self.weights.clone();
        let mut layers = Vec::new();
        model
            .base_model
            .forward_internal(tokens, 0, Some(&mut layers))?;
        Ok(layers)
    }
}

pub fn load(config: &Config, telemetry: &Telemetry) -> Result<Box<dyn ModelFactory>> {
    config.validate()?;
    let raw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(config.model_root.join("config.json"))?)?;
    if raw["model_type"] != "qwen3_5" {
        return Err(Error::Config(
            "Unsupported model architecture; expected qwen3_5".into(),
        ));
    }
    let mut model_config: qwen35::Config = serde_json::from_value(raw.clone())?;
    model_config.validate()?;
    if config.max_context > model_config.text_config.max_position_embeddings {
        return Err(Error::Config(
            "Context exceeds model's position limit".into(),
        ));
    }
    model_config.text_config.max_position_embeddings = config.max_context;
    let eos = raw["text_config"]["eos_token_id"]
        .as_u64()
        .and_then(|x| u32::try_from(x).ok())
        .ok_or_else(|| Error::Config("Missing EOS token".into()))?;
    if eos as usize >= model_config.text_config.vocab_size {
        return Err(Error::Config("EOS token exceeds vocabulary".into()));
    }
    telemetry.event("device_initialization_started", &config.device);
    let device = match config.device {
        DeviceConfig::Cpu => Device::Cpu,
        DeviceConfig::Cuda { ordinal } => Device::new_cuda(ordinal)?,
    };
    telemetry.event("device_initialized", &config.device);
    let dtype = match config.dtype {
        Precision::Bf16 => DType::BF16,
        Precision::F32 => DType::F32,
    };
    let vb = Weights::read(&config.model_root, config.max_weight_bytes, telemetry)?
        .builder(dtype, &device);
    let weights = qwen35::ModelForCausalLM::new(&model_config, vb)?;
    device.synchronize()?;
    telemetry.event("model_loaded", serde_json::json!({"vocab":model_config.text_config.vocab_size,"layers":model_config.text_config.num_hidden_layers}));
    Ok(Box::new(QwenFactory {
        weights,
        device,
        vocab: model_config.text_config.vocab_size,
        eos,
    }))
}
