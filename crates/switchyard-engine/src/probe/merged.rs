//! Offline teacher forcing through the same mixed session used by generation.
use super::{Candidate, Step, Trace};
use crate::{Error, Result, config::Config, models::ModelFactory};
use candle_core::{DType, Tensor};
use std::collections::BTreeMap;

pub fn probe_merged(
    model: &dyn ModelFactory,
    config: &Config,
    traces: &[Trace],
) -> Result<Vec<Vec<Step>>> {
    if traces.is_empty() || config.max_batch_size == 0 {
        return Err(Error::Config(
            "Empty merged diagnostic or batch capacity".into(),
        ));
    }
    for trace in traces {
        if trace.prompt.is_empty()
            || trace.continuation.is_empty()
            || trace
                .prompt
                .len()
                .checked_add(trace.continuation.len())
                .is_none_or(|end| end > config.max_context)
            || trace
                .prompt
                .iter()
                .chain(&trace.continuation)
                .any(|&id| id as usize >= model.vocab_size())
        {
            return Err(Error::Config("Invalid merged diagnostic trace".into()));
        }
    }
    let mut output: Vec<Vec<Step>> = traces.iter().map(|_| Vec::new()).collect();
    for (chunk, cohort) in traces.chunks(config.max_batch_size).enumerate() {
        let base = chunk * config.max_batch_size;
        let mut session = model
            .mixed_session(config.max_batch_size)?
            .ok_or_else(|| Error::Config("Model has no mixed decode adapter".into()))?;
        let mut consumed = vec![0usize; cohort.len()];
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (row, trace) in cohort.iter().enumerate() {
            groups.entry(trace.prompt.len()).or_default().push(row);
        }
        for (length, rows) in groups {
            let ids = rows
                .iter()
                .flat_map(|&row| cohort[row].prompt.iter().copied())
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(ids, (rows.len(), length), model.device())?;
            let logits = session
                .prefill(&rows, &input)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            if logits.len() != rows.len() {
                return Err(Error::Config("Merged prefill logit rows".into()));
            }
            let mut released = Vec::new();
            for (&row, values) in rows.iter().zip(logits) {
                output[base + row].push(describe(values, 0, cohort[row].continuation[0])?);
                consumed[row] = 1;
                if consumed[row] == cohort[row].continuation.len() {
                    released.push(row);
                }
            }
            session.release(&released)?;
        }
        loop {
            let rows: Vec<_> = (0..cohort.len())
                .filter(|&row| consumed[row] < cohort[row].continuation.len())
                .collect();
            if rows.is_empty() {
                break;
            }
            let ids = rows
                .iter()
                .map(|&row| cohort[row].continuation[consumed[row] - 1])
                .collect::<Vec<_>>();
            let positions = rows
                .iter()
                .map(|&row| cohort[row].prompt.len() + consumed[row] - 1)
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(ids, (rows.len(), 1), model.device())?;
            let logits = session
                .decode(&rows, &input, &positions)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            if logits.len() != rows.len() {
                return Err(Error::Config("Merged decode logit rows".into()));
            }
            let mut released = Vec::new();
            for ((&row, &position), values) in rows.iter().zip(&positions).zip(logits) {
                output[base + row].push(describe(
                    values,
                    position,
                    cohort[row].continuation[consumed[row]],
                )?);
                consumed[row] += 1;
                if consumed[row] == cohort[row].continuation.len() {
                    released.push(row);
                }
            }
            session.release(&released)?;
        }
    }
    Ok(output)
}

fn describe(logits: Vec<f32>, position: usize, expected: u32) -> Result<Step> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite()
        || logits.iter().any(|x| !x.is_finite())
        || expected as usize >= logits.len()
    {
        return Err(Error::Config(
            "Invalid/nonfinite merged diagnostic logits".into(),
        ));
    }
    let normalizer = max as f64
        + logits
            .iter()
            .map(|&x| (x as f64 - max as f64).exp())
            .sum::<f64>()
            .ln();
    let mut indices: Vec<_> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(20.min(logits.len() - 1), |&a, &b| {
        logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
    });
    indices.truncate(20.min(logits.len()));
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    Ok(Step {
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
    })
}
