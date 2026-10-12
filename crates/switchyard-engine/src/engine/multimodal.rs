//! Vision sequences own independent state; ordinary text retains tensor batching.
use super::{Engine, Generation};
use crate::{Error, Result, multimodal::InferenceRequest};
use candle_core::{DType, Tensor};
use std::time::Instant;

impl Engine {
    pub fn generate_requests_observed(
        &self,
        requests: &[InferenceRequest],
        mut observer: impl FnMut(usize, u32, f64, bool) -> bool,
    ) -> Result<Vec<Generation>> {
        if requests.iter().any(|r| !r.images.is_empty()) && !self.model.supports_vision() {
            return Err(Error::Config(
                "Model has no vision encoder/projector".into(),
            ));
        }
        for request in requests {
            self.validate(&request.generation)?;
            if !request.images.is_empty() {
                let bounds = self
                    .config
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::Config("Vision serving is disabled".into()))?;
                bounds.validate()?;
                if request.images.len() > bounds.max_images {
                    return Err(Error::Config("Too many request images".into()));
                }
                let mut end = 0;
                for image in &request.images {
                    // Qwen's static images have a temporal grid of one and 2x2 merging.
                    if image.grid[0] != 1
                        || image.grid[1] == 0
                        || image.grid[2] == 0
                        || !image.grid[1].is_multiple_of(2)
                        || !image.grid[2].is_multiple_of(2)
                        || image.encoded.is_empty()
                        || image.encoded.len() > bounds.max_encoded_bytes
                        || image.offset < end
                    {
                        return Err(Error::Config("Invalid bounded image input".into()));
                    }
                    let patches = image.grid[1]
                        .checked_mul(image.grid[2])
                        .ok_or_else(|| Error::Config("Image grid overflow".into()))?;
                    if patches
                        .checked_mul(256)
                        .is_none_or(|n| n > bounds.max_pixels)
                    {
                        return Err(Error::Config("Image grid exceeds pixel budget".into()));
                    }
                    end = image
                        .offset
                        .checked_add(patches / 4)
                        .filter(|&n| n <= request.generation.prompt.len())
                        .ok_or_else(|| Error::Config("Image span exceeds prompt".into()))?;
                }
            }
        }
        let text_rows: Vec<_> = requests
            .iter()
            .enumerate()
            .filter_map(|(i, request)| request.images.is_empty().then_some(i))
            .collect();
        let text_requests: Vec<_> = text_rows
            .iter()
            .map(|&i| requests[i].generation.clone())
            .collect();
        let mut output: Vec<Option<Generation>> = vec![None; requests.len()];
        if !text_requests.is_empty() {
            let generated = self
                .generate_observed(&text_requests, |row, token, elapsed, finished| {
                    observer(text_rows[row], token, elapsed, finished)
                })?;
            for (row, generated) in text_rows.into_iter().zip(generated) {
                output[row] = Some(generated);
            }
        }
        for (row, request) in requests
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.images.is_empty())
        {
            let mut session = self
                .model
                .vision_session()?
                .ok_or_else(|| Error::Config("Model has no vision encoder/projector".into()))?;
            let began = Instant::now();
            let request_tokens = &request.generation;
            let mut input = Tensor::from_vec(
                request_tokens.prompt.clone(),
                (1, request_tokens.prompt.len()),
                self.model.device(),
            )?;
            let mut position = 0;
            let mut generation = Generation {
                token_ids: Vec::new(),
                token_latency_s: Vec::new(),
                elapsed_s: 0.,
                prompt_tokens: request_tokens.prompt.len(),
                finish_reason: "length".into(),
            };
            self.telemetry.event("vision_sequence_started", serde_json::json!({"row":row,"images":request.images.len(),"prompt_tokens":request_tokens.prompt.len(),"execution":"independent_sequence"}));
            loop {
                let length = input.dim(1)?;
                let logits = if position == 0 {
                    session.prefill_images(&input, &request.images)?
                } else {
                    session.forward(&input, position)?
                };
                let next = self
                    .sampler
                    .select(&logits)?
                    .to_dtype(DType::U32)?
                    .to_vec1::<u32>()?;
                if next.len() != 1 {
                    return Err(Error::Config("Vision logits row count".into()));
                }
                let token = next[0];
                let elapsed = began.elapsed().as_secs_f64();
                generation.token_ids.push(token);
                generation.token_latency_s.push(elapsed);
                generation.elapsed_s = elapsed;
                let stop = token == self.model.eos_token() && !request_tokens.ignore_eos;
                let finished = stop || generation.token_ids.len() == request_tokens.max_tokens;
                if stop {
                    generation.finish_reason = "stop".into();
                }
                if !observer(row, token, elapsed, finished) {
                    generation.finish_reason = "cancelled".into();
                    break;
                }
                if finished {
                    break;
                }
                position += length;
                input = Tensor::from_vec(next, (1, 1), self.model.device())?;
            }
            self.telemetry.event("vision_sequence_finished", serde_json::json!({"row":row,"elapsed_s":began.elapsed().as_secs_f64(),"output_tokens":generation.token_ids.len()}));
            output[row] = Some(generation);
        }
        output
            .into_iter()
            .map(|r| r.ok_or_else(|| Error::Config("Missing image/text row".into())))
            .collect()
    }
}
