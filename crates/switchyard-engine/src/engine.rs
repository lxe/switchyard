use crate::{Error, Result, config::Config, models::ModelFactory, telemetry::Telemetry};
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, time::Instant};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Generate {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    #[serde(default)]
    pub ignore_eos: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct Generation {
    pub token_ids: Vec<u32>,
    pub token_latency_s: Vec<f64>,
    pub elapsed_s: f64,
    pub prompt_tokens: usize,
    pub finish_reason: String,
}
pub struct Engine {
    config: Config,
    model: Box<dyn ModelFactory>,
    telemetry: Telemetry,
    sampler: crate::sampling::Greedy,
}
impl Engine {
    pub fn new(config: Config, model: Box<dyn ModelFactory>, telemetry: Telemetry) -> Result<Self> {
        let sampler = crate::sampling::Greedy::new(model.vocab_size(), model.device())?;
        Ok(Self {
            config,
            model,
            telemetry,
            sampler,
        })
    }
    pub fn eos_token(&self) -> u32 {
        self.model.eos_token()
    }
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }
    pub fn validate(&self, request: &Generate) -> Result<()> {
        if request.prompt.is_empty()
            || request.max_tokens == 0
            || request
                .prompt
                .len()
                .checked_add(request.max_tokens)
                .is_none_or(|n| n > self.config.max_context)
            || request
                .prompt
                .iter()
                .any(|&x| x as usize >= self.model.vocab_size())
        {
            return Err(Error::Config(
                "Invalid token IDs or context/output bounds".into(),
            ));
        }
        Ok(())
    }
    /// Groups by prompt length so no padding can contaminate recurrent state.
    /// Rows execute as a genuine tensor batch; returned results retain caller order.
    pub fn generate(&self, requests: &[Generate]) -> Result<Vec<Generation>> {
        self.generate_observed(requests, |_, _, _, _| true)
    }
    /// Called after each device-synchronized token. Returning false cancels that
    /// sequence; other rows continue. The observer must not block GPU progress.
    pub fn generate_observed(
        &self,
        requests: &[Generate],
        mut observer: impl FnMut(usize, u32, f64, bool) -> bool,
    ) -> Result<Vec<Generation>> {
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, request) in requests.iter().enumerate() {
            self.validate(request)?;
            groups.entry(request.prompt.len()).or_default().push(i);
        }
        let mut results: Vec<Option<Generation>> = vec![None; requests.len()];
        for (length, indices) in groups {
            for cohort in indices.chunks(self.config.max_batch_size) {
                let began = Instant::now();
                let mut session = self.model.session();
                let mut generated: Vec<Generation> = cohort
                    .iter()
                    .map(|_| Generation {
                        token_ids: Vec::new(),
                        token_latency_s: Vec::new(),
                        elapsed_s: 0.,
                        prompt_tokens: length,
                        finish_reason: "length".into(),
                    })
                    .collect();
                let mut finished = vec![false; cohort.len()];
                let tokens: Vec<u32> = cohort
                    .iter()
                    .flat_map(|&i| requests[i].prompt.clone())
                    .collect();
                let mut input =
                    Tensor::from_vec(tokens, (cohort.len(), length), self.model.device())?;
                let mut position = 0;
                self.telemetry.event(
                    "batch_started",
                    serde_json::json!({"batch_size":cohort.len(),"prompt_tokens":length}),
                );
                loop {
                    let input_len = input.dim(1)?;
                    let logits = session.forward(&input, position)?;
                    // Greedy argmax on device; transferring token IDs also synchronizes work.
                    let next = self
                        .sampler
                        .select(&logits)?
                        .to_dtype(DType::U32)?
                        .to_vec1::<u32>()?;
                    let elapsed = began.elapsed().as_secs_f64();
                    for (row, &i) in cohort.iter().enumerate() {
                        if finished[row] {
                            continue;
                        }
                        let out = &mut generated[row];
                        out.token_ids.push(next[row]);
                        out.token_latency_s.push(elapsed);
                        out.elapsed_s = elapsed;
                        if next[row] == self.model.eos_token() && !requests[i].ignore_eos {
                            out.finish_reason = "stop".into();
                            finished[row] = true;
                        }
                        if out.token_ids.len() == requests[i].max_tokens {
                            finished[row] = true;
                        }
                        if !observer(i, next[row], elapsed, finished[row]) {
                            out.finish_reason = "cancelled".into();
                            finished[row] = true;
                        }
                    }
                    if finished.iter().all(|x| *x) {
                        break;
                    }
                    position += input_len;
                    input = Tensor::from_vec(next, (cohort.len(), 1), self.model.device())?;
                }
                self.telemetry.event("batch_finished",serde_json::json!({"batch_size":cohort.len(),"elapsed_s":began.elapsed().as_secs_f64()}));
                for (&i, result) in cohort.iter().zip(generated) {
                    results[i] = Some(result);
                }
            }
        }
        results
            .into_iter()
            .map(|x| x.ok_or_else(|| Error::Config("Missing generated row".into())))
            .collect()
    }
}
