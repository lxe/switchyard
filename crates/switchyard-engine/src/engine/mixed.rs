//! Existing observer semantics with separate prefill cohorts and shared decode.
use super::{Engine, Generate, Generation};
use crate::{Error, Result, models::MixedModelSession};
use candle_core::{DType, Tensor};
use std::{collections::BTreeMap, time::Instant};

impl Engine {
    pub(super) fn generate_mixed(
        &self,
        requests: &[Generate],
        session: Box<dyn MixedModelSession>,
        observer: &mut impl FnMut(usize, u32, f64, bool) -> bool,
    ) -> Result<Vec<Generation>> {
        let mut first = Some(session);
        let mut results = Vec::new();
        for (chunk, cohort) in requests.chunks(self.config.max_batch_size).enumerate() {
            let base = chunk * self.config.max_batch_size;
            let mut session = match first.take() {
                Some(s) => s,
                None => self
                    .model
                    .mixed_session(self.config.max_batch_size)?
                    .ok_or_else(|| Error::Config("Adapter lost mixed session support".into()))?,
            };
            let began = Instant::now();
            let mut output: Vec<_> = cohort
                .iter()
                .map(|r| Generation {
                    token_ids: Vec::new(),
                    token_latency_s: Vec::new(),
                    elapsed_s: 0.,
                    prompt_tokens: r.prompt.len(),
                    finish_reason: "length".into(),
                })
                .collect();
            let mut finished = vec![false; cohort.len()];
            let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (row, r) in cohort.iter().enumerate() {
                groups.entry(r.prompt.len()).or_default().push(row);
            }
            self.telemetry.event("mixed_batch_started",serde_json::json!({"batch_size":cohort.len(),"prefill_cohorts":groups.len(),"prompt_lengths":cohort.iter().map(|r|r.prompt.len()).collect::<Vec<_>>()}));
            for (length, rows) in groups {
                let ids = rows
                    .iter()
                    .flat_map(|&r| cohort[r].prompt.iter().copied())
                    .collect::<Vec<_>>();
                let input = Tensor::from_vec(ids, (rows.len(), length), self.model.device())?;
                let logits = session.prefill(&rows, &input)?;
                self.observe_mixed(
                    cohort,
                    base,
                    &rows,
                    &logits,
                    began,
                    &mut output,
                    &mut finished,
                    observer,
                    session.as_mut(),
                )?;
            }
            loop {
                let rows: Vec<_> = (0..cohort.len()).filter(|&row| !finished[row]).collect();
                if rows.is_empty() {
                    break;
                }
                let ids = rows
                    .iter()
                    .map(|&row| *output[row].token_ids.last().unwrap())
                    .collect::<Vec<_>>();
                let positions = rows
                    .iter()
                    .map(|&row| cohort[row].prompt.len() + output[row].token_ids.len() - 1)
                    .collect::<Vec<_>>();
                let input = Tensor::from_vec(ids, (rows.len(), 1), self.model.device())?;
                self.telemetry.event("mixed_decode_step",serde_json::json!({"active_rows":rows.len(),"request_rows":rows,"positions":positions}));
                let logits = session.decode(&rows, &input, &positions)?;
                self.observe_mixed(
                    cohort,
                    base,
                    &rows,
                    &logits,
                    began,
                    &mut output,
                    &mut finished,
                    observer,
                    session.as_mut(),
                )?;
            }
            self.telemetry.event("mixed_batch_finished",serde_json::json!({"batch_size":cohort.len(),"elapsed_s":began.elapsed().as_secs_f64()}));
            results.extend(output);
        }
        Ok(results)
    }

    #[allow(clippy::too_many_arguments)]
    fn observe_mixed(
        &self,
        requests: &[Generate],
        base: usize,
        rows: &[usize],
        logits: &Tensor,
        began: Instant,
        output: &mut [Generation],
        finished: &mut [bool],
        observer: &mut impl FnMut(usize, u32, f64, bool) -> bool,
        session: &mut dyn MixedModelSession,
    ) -> Result<()> {
        let next = self
            .sampler
            .select(logits)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        if next.len() != rows.len() {
            return Err(Error::Config("Mixed logits row count".into()));
        }
        let elapsed = began.elapsed().as_secs_f64();
        let mut released = Vec::new();
        for (&row, &token) in rows.iter().zip(&next) {
            let out = &mut output[row];
            out.token_ids.push(token);
            out.token_latency_s.push(elapsed);
            out.elapsed_s = elapsed;
            if token == self.model.eos_token() && !requests[row].ignore_eos {
                out.finish_reason = "stop".into();
                finished[row] = true;
            }
            if out.token_ids.len() == requests[row].max_tokens {
                finished[row] = true;
            }
            if !observer(base + row, token, elapsed, finished[row]) {
                out.finish_reason = "cancelled".into();
                finished[row] = true;
            }
            if finished[row] {
                released.push(row);
            }
        }
        // to_vec1 above completes the GPU step before state can be reclaimed.
        session.release(&released)
    }
}
