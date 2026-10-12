//! Model adapters share immutable weights; each batch owns its sequence state.
mod gguf;
pub mod lora;
mod mixed;
mod prefix;
mod projection;
pub mod qwen35;
pub mod vision;
mod vision_session;
use crate::weights::Weights;
use crate::{
    Error, Result,
    config::{Config, DeviceConfig, Precision, WeightFormat},
    telemetry::Telemetry,
};
use candle_core::{DType, Device, Tensor};
pub use mixed::QwenMixedSession;
pub use prefix::{PrefixCache, PrefixCacheStats};
pub type LayerTrace = Vec<(String, Tensor)>;

/// Execution contract for one cohort of equal-length token sequences.
/// A new session must have empty state and may share immutable weight storage.
pub trait ModelSession: Send {
    fn forward(&mut self, tokens: &Tensor, position: usize) -> Result<Tensor>;
    fn prefill_images(
        &mut self,
        _: &Tensor,
        _: &[crate::multimodal::ImageInput],
    ) -> Result<Tensor> {
        Err(Error::Config(
            "Image prefill unsupported by this adapter".into(),
        ))
    }
}
/// Independently prefilled rows join one decode cohort. No new prefill is allowed
/// after decoding starts. Row IDs stay stable through completion/cancellation.
pub trait MixedModelSession: Send {
    fn prefill(&mut self, rows: &[usize], tokens: &Tensor) -> Result<Tensor>;
    fn decode(&mut self, rows: &[usize], tokens: &Tensor, positions: &[usize]) -> Result<Tensor>;
    /// Called after token IDs have reached the host, completing this step's CUDA
    /// work. Implementations release only the named rows' private model state.
    fn release(&mut self, rows: &[usize]) -> Result<()>;
}
pub trait ModelFactory: Send {
    fn adapter_dtype(&self) -> DType {
        DType::F32
    }
    fn with_lora(&self, _: &lora::Adapter) -> Result<Box<dyn ModelFactory>> {
        Err(Error::Config(
            "Runtime LoRA unsupported by this model adapter".into(),
        ))
    }
    fn session(&self) -> Box<dyn ModelSession>;
    fn vision_session(&self) -> Result<Option<Box<dyn ModelSession>>> {
        Ok(None)
    }
    fn supports_vision(&self) -> bool {
        false
    }
    /// Offline reference path bypassing prefix lookup and checkpoint retention.
    fn uncached_session(&self) -> Box<dyn ModelSession> {
        self.session()
    }
    fn clear_prefix_cache(&self) -> Result<()> {
        Ok(())
    }
    fn prefix_cache_stats(&self) -> Result<Option<PrefixCacheStats>> {
        Ok(None)
    }
    fn device(&self) -> &Device;
    fn vocab_size(&self) -> usize;
    fn eos_token(&self) -> u32;
    fn mixed_session(&self, _max_rows: usize) -> Result<Option<Box<dyn MixedModelSession>>> {
        Ok(None)
    }
    /// Offline layer inspection; unsupported adapters need not implement it.
    fn inspect_layers(&self, _: &Tensor) -> Result<LayerTrace> {
        Err(Error::Config(
            "Layer inspection unsupported by this adapter".into(),
        ))
    }
}

pub struct QwenSession {
    model: qwen35::ModelForCausalLM,
    position: usize,
    rows: Option<usize>,
    cache: Option<PrefixCache>,
}
impl QwenSession {
    pub fn new(mut model: qwen35::ModelForCausalLM) -> Self {
        model.clear_kv_cache();
        Self {
            model,
            position: 0,
            rows: None,
            cache: None,
        }
    }
    pub fn with_cache(model: qwen35::ModelForCausalLM, cache: PrefixCache) -> Self {
        let mut session = Self::new(model);
        session.cache = cache.enabled().then_some(cache);
        session
    }
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
        let logits = if position == 0
            && let Some(cache) = &self.cache
        {
            let (logits, states) = cache.prefill(&self.model, tokens)?;
            self.model
                .base_model
                .restore_sequences(&states.iter().collect::<Vec<_>>())?;
            logits
        } else {
            self.model.forward(tokens, position)?.squeeze(1)?
        };
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
    cache: PrefixCache,
    vision: Option<(std::sync::Arc<vision::VisionModel>, usize)>,
    cache_config: crate::config::CacheConfig,
    telemetry: Telemetry,
}
impl ModelFactory for QwenFactory {
    fn adapter_dtype(&self) -> DType {
        DType::F32
    }
    fn with_lora(&self, adapter: &lora::Adapter) -> Result<Box<dyn ModelFactory>> {
        Ok(Box::new(Self {
            weights: self.weights.with_lora(adapter)?,
            device: self.device.clone(),
            vocab: self.vocab,
            eos: self.eos,
            cache: PrefixCache::new(self.cache_config.clone(), self.telemetry.clone())?,
            vision: self.vision.clone(),
            cache_config: self.cache_config.clone(),
            telemetry: self.telemetry.clone(),
        }))
    }
    fn supports_vision(&self) -> bool {
        self.vision.is_some()
    }
    fn vision_session(&self) -> Result<Option<Box<dyn ModelSession>>> {
        Ok(self.vision.as_ref().map(|(vision, max_pixels)| {
            Box::new(vision_session::QwenVisionSession::new(
                self.weights.clone(),
                vision.clone(),
                *max_pixels,
            )) as Box<dyn ModelSession>
        }))
    }
    fn session(&self) -> Box<dyn ModelSession> {
        Box::new(QwenSession::with_cache(
            self.weights.clone(),
            self.cache.clone(),
        ))
    }
    fn uncached_session(&self) -> Box<dyn ModelSession> {
        Box::new(QwenSession::new(self.weights.clone()))
    }
    fn clear_prefix_cache(&self) -> Result<()> {
        self.cache.clear()
    }
    fn prefix_cache_stats(&self) -> Result<Option<PrefixCacheStats>> {
        Ok(Some(self.cache.stats()?))
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
    fn mixed_session(&self, max_rows: usize) -> Result<Option<Box<dyn MixedModelSession>>> {
        Ok(Some(Box::new(QwenMixedSession::with_cache(
            self.weights.clone(),
            max_rows,
            self.cache.clone(),
        )?)))
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
    let mut eos = raw["text_config"]["eos_token_id"]
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
    let mut dense_vision_builder = None;
    let mut weights = match &config.weights {
        WeightFormat::Safetensors => {
            let vb = Weights::read(&config.model_root, config.max_weight_bytes, telemetry)?
                .builder(dtype, &device);
            if config.vision.as_ref().is_some_and(|v| v.mmproj.is_none()) {
                dense_vision_builder = Some(vb.clone());
            }
            qwen35::ModelForCausalLM::new(&model_config, vb)?
        }
        WeightFormat::Gguf { file } => {
            let source = gguf::Source::open(
                &config.model_root.join(file),
                config.max_weight_bytes,
                &model_config,
                telemetry,
            )?;
            eos = source.eos_token()?;
            let weights =
                qwen35::ModelForCausalLM::new_with_weights(&model_config, source.builder(&device))?;
            telemetry.event(
                "weights_read_finished",
                serde_json::json!({"format":"gguf","projection_storage":"packed"}),
            );
            weights
        }
    };
    let vision = if let Some(bounds) = &config.vision {
        let vision_config: vision::VisionConfig =
            serde_json::from_value(raw["vision_config"].clone())?;
        vision_config.validate()?;
        telemetry.event(
            "vision_loading_started",
            serde_json::json!({"mmproj":bounds.mmproj.is_some()}),
        );
        let model = match &bounds.mmproj {
            Some(path) => vision::VisionModel::from_gguf(
                &vision_config,
                &config.model_root.join(path),
                &device,
                dtype,
            )?,
            None => vision::VisionModel::new(
                &vision_config,
                dense_vision_builder
                    .ok_or_else(|| Error::Config("Original HF vision tensors unavailable".into()))?
                    .pp("model.visual"),
            )?,
        };
        telemetry.event("vision_loaded", serde_json::json!({"layers":vision_config.depth,"hidden":vision_config.out_hidden_size}));
        Some((std::sync::Arc::new(model), bounds.max_pixels))
    } else {
        None
    };
    weights
        .base_model
        .set_kv_page_tokens(config.cache.page_tokens)?;
    let cache = PrefixCache::new(config.cache.clone(), telemetry.clone())?;
    device.synchronize()?;
    telemetry.event("model_loaded", serde_json::json!({"vocab":model_config.text_config.vocab_size,"layers":model_config.text_config.num_hidden_layers}));
    Ok(Box::new(QwenFactory {
        weights,
        device,
        vocab: model_config.text_config.vocab_size,
        eos,
        cache,
        vision,
        cache_config: config.cache.clone(),
        telemetry: telemetry.clone(),
    }))
}
