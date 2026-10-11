//! Offline teacher-forced diagnostics; never used by the serving hot path.
use crate::{Error, Result, config::Config, models::ModelFactory};
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub prompt: Vec<u32>,
    pub continuation: Vec<u32>,
}
#[derive(Serialize)]
pub struct Candidate {
    pub token: u32,
    pub logit: f32,
    pub logprob: f64,
}
#[derive(Serialize)]
pub struct Step {
    pub position: usize,
    pub expected: u32,
    pub expected_logprob: f64,
    pub top: Vec<Candidate>,
}
pub fn probe(
    model: &dyn ModelFactory,
    config: &Config,
    traces: &[Trace],
) -> Result<Vec<Vec<Step>>> {
    let mut output = Vec::new();
    for trace in traces {
        if trace.prompt.is_empty()
            || trace.continuation.is_empty()
            || trace
                .prompt
                .len()
                .checked_add(trace.continuation.len())
                .is_none_or(|n| n > config.max_context)
            || trace
                .prompt
                .iter()
                .chain(&trace.continuation)
                .any(|&t| t as usize >= model.vocab_size())
        {
            return Err(Error::Config("Invalid diagnostic trace".into()));
        }
        let mut session = model.session();
        let mut position = 0;
        let mut input = Tensor::from_vec(
            trace.prompt.clone(),
            (1, trace.prompt.len()),
            model.device(),
        )?;
        let mut steps = Vec::new();
        for &expected in &trace.continuation {
            let logits = session
                .forward(&input, position)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?
                .remove(0);
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() || logits.iter().any(|x| !x.is_finite()) {
                return Err(Error::Config("Nonfinite diagnostic logits".into()));
            }
            let normalizer = max as f64
                + logits
                    .iter()
                    .map(|&x| (x as f64 - max as f64).exp())
                    .sum::<f64>()
                    .ln();
            let mut indices: Vec<usize> = (0..logits.len()).collect();
            indices.select_nth_unstable_by(20.min(logits.len() - 1), |&a, &b| {
                logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
            });
            indices.truncate(20.min(logits.len()));
            indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            steps.push(Step {
                position,
                expected,
                expected_logprob: logits[expected as usize] as f64 - normalizer,
                top: indices
                    .into_iter()
                    .map(|i| Candidate {
                        token: i as u32,
                        logit: logits[i],
                        logprob: logits[i] as f64 - normalizer,
                    })
                    .collect(),
            });
            position += input.dim(1)?;
            input = Tensor::from_vec(vec![expected], (1, 1), model.device())?;
        }
        output.push(steps);
    }
    Ok(output)
}
