//! Offline prefix-reuse timing with one loaded model and identical token inputs.
use candle_core::{DType, Tensor};
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};
use switchyard_engine::{
    Error, Result,
    config::{Config, Precision},
    engine::Generate,
    models::{self, ModelFactory},
    sampling::Greedy,
    telemetry::Telemetry,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    requests: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 3)]
    samples: usize,
}
#[derive(Serialize)]
struct Sample {
    mode: &'static str,
    elapsed_s: f64,
    first_token_s: f64,
    output_tokens_per_s: f64,
    token_ids: Vec<Vec<u32>>,
    cache: Option<models::PrefixCacheStats>,
}
fn run(
    model: &dyn ModelFactory,
    sampler: &Greedy,
    requests: &[Generate],
    mode: &'static str,
) -> Result<Sample> {
    let batch = requests.len();
    let length = requests[0].prompt.len();
    let mut session = if mode == "uncached" {
        model.uncached_session()
    } else {
        model.session()
    };
    let ids = requests
        .iter()
        .flat_map(|r| r.prompt.iter().copied())
        .collect::<Vec<_>>();
    let mut input = Tensor::from_vec(ids, (batch, length), model.device())?;
    model.device().synchronize()?;
    let began = Instant::now();
    let mut position = 0;
    let mut first_token_s = 0.;
    let mut output = vec![Vec::new(); batch];
    for step in 0..requests[0].max_tokens {
        let width = input.dim(1)?;
        let logits = session.forward(&input, position)?;
        let next = sampler
            .select(&logits)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        if step == 0 {
            first_token_s = began.elapsed().as_secs_f64();
        }
        for (row, &token) in next.iter().enumerate() {
            output[row].push(token);
        }
        position += width;
        input = Tensor::from_vec(next, (batch, 1), model.device())?;
    }
    let elapsed_s = began.elapsed().as_secs_f64();
    Ok(Sample {
        mode,
        elapsed_s,
        first_token_s,
        output_tokens_per_s: (batch * requests[0].max_tokens) as f64 / elapsed_s,
        token_ids: output,
        cache: model.prefix_cache_stats()?,
    })
}
fn main() -> Result<()> {
    let args = Args::parse();
    let config: Config = serde_json::from_slice(&std::fs::read(&args.config)?)?;
    config.validate()?;
    let requests: Vec<Generate> = serde_json::from_slice(&std::fs::read(&args.requests)?)?;
    if !config.cache.enabled()
        || requests.is_empty()
        || requests.len() > config.max_batch_size
        || args.samples == 0
        || args.samples > 31
        || requests.iter().any(|r| {
            !r.ignore_eos
                || r.prompt.is_empty()
                || r.max_tokens == 0
                || r.prompt.len() != requests[0].prompt.len()
                || r.max_tokens != requests[0].max_tokens
                || r.prompt
                    .len()
                    .checked_add(r.max_tokens)
                    .is_none_or(|n| n > config.max_context)
        })
    {
        return Err(Error::Config(
            "Cache timing requires enabled fixed bounds and one equal-length/equal-output cohort"
                .into(),
        ));
    }
    let telemetry = Telemetry::default();
    let model = models::load(&config, &telemetry)?;
    if requests
        .iter()
        .any(|r| r.prompt.iter().any(|&t| t as usize >= model.vocab_size()))
    {
        return Err(Error::Config("Invalid benchmark token".into()));
    }
    let sampler = Greedy::new(model.vocab_size(), model.device())?;
    let warmup = run(model.as_ref(), &sampler, &requests, "uncached")?;
    let mut samples = Vec::new();
    for index in 0..args.samples {
        // Alternate baseline placement; each cold measurement starts empty and
        // its immediate repeat observes exactly the checkpoints it produced.
        if index % 2 == 0 {
            samples.push(run(model.as_ref(), &sampler, &requests, "uncached")?);
        }
        model.clear_prefix_cache()?;
        samples.push(run(model.as_ref(), &sampler, &requests, "empty_cache")?);
        samples.push(run(model.as_ref(), &sampler, &requests, "repeat")?);
        if index % 2 == 1 {
            samples.push(run(model.as_ref(), &sampler, &requests, "uncached")?);
        }
    }
    let identical = samples
        .iter()
        .all(|sample| sample.token_ids == warmup.token_ids);
    let mut summaries = Vec::new();
    for mode in ["uncached", "empty_cache", "repeat"] {
        let mut elapsed = samples
            .iter()
            .filter(|s| s.mode == mode)
            .map(|s| s.elapsed_s)
            .collect::<Vec<_>>();
        let mut first = samples
            .iter()
            .filter(|s| s.mode == mode)
            .map(|s| s.first_token_s)
            .collect::<Vec<_>>();
        elapsed.sort_by(f64::total_cmp);
        first.sort_by(f64::total_cmp);
        let median = |values: &[f64]| {
            if values.len() % 2 == 1 {
                values[values.len() / 2]
            } else {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.
            }
        };
        let elapsed_s = median(&elapsed);
        summaries.push(serde_json::json!({"mode":mode, "median_elapsed_s":elapsed_s,
            "median_first_token_s":median(&first),
            "output_tokens_per_median_s":(requests.len() * requests[0].max_tokens) as f64 / elapsed_s}));
    }
    let receipt = serde_json::json!({"scope":"warm loaded-model offline generation; includes prefill, excludes loading, HTTP, network and queueing",
        "config":config, "requests":requests, "warmup":warmup, "samples":samples, "summary":summaries,
        "binary_sha256":format!("{:x}", Sha256::digest(std::fs::read(std::env::current_exe()?)?)),
        "full_sequence_equality":identical, "cache_budget_note":"retained tensor/token storage; active state and host metadata bounded separately"});
    std::fs::write(args.output, serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string(&summaries)?);
    if !identical && matches!(config.dtype, Precision::F32) {
        return Err(Error::Config(
            "F32 cached/uncached full sequences differ; receipt retained".into(),
        ));
    }
    Ok(())
}
