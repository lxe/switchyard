//! Request-owned multimodal state; encoded images enter CUDA only on the inference owner.
use super::{ModelSession, qwen35, vision};
use crate::{Error, Result, multimodal::ImageInput};
use candle_core::Tensor;
use std::sync::Arc;

pub(super) struct QwenVisionSession {
    model: qwen35::ModelForCausalLM,
    vision: Arc<vision::VisionModel>,
    max_pixels: usize,
    delta: Option<i64>,
}
impl QwenVisionSession {
    pub(super) fn new(
        mut model: qwen35::ModelForCausalLM,
        vision: Arc<vision::VisionModel>,
        max_pixels: usize,
    ) -> Self {
        model.clear_kv_cache();
        Self {
            model,
            vision,
            max_pixels,
            delta: None,
        }
    }
}
impl ModelSession for QwenVisionSession {
    fn prefill_images(&mut self, tokens: &Tensor, inputs: &[ImageInput]) -> Result<Tensor> {
        if self.delta.is_some() || inputs.is_empty() {
            return Err(Error::Config(
                "Image session requires exactly one nonempty prefill".into(),
            ));
        }
        let mut spans = Vec::with_capacity(inputs.len());
        for input in inputs {
            let image = vision::prepare_image(&input.encoded, self.max_pixels, tokens.device())?;
            if image.grid != input.grid {
                return Err(Error::Config(
                    "Image grid disagrees with request planning".into(),
                ));
            }
            spans.push(qwen35::ImageSpan {
                offset: input.offset,
                grid: image.grid,
                embeddings: self.vision.forward(&image)?,
            });
        }
        let (logits, delta) = self.model.prefill_images(tokens, &spans)?;
        self.delta = Some(delta);
        Ok(logits.squeeze(1)?)
    }
    fn forward(&mut self, tokens: &Tensor, position: usize) -> Result<Tensor> {
        let delta = self
            .delta
            .ok_or_else(|| Error::Config("Image session has no prefill".into()))?;
        Ok(self
            .model
            .forward_with_rope_offset(tokens, position, delta)?
            .squeeze(1)?)
    }
}
