//! Offline dense F32 timing after a successful mixed teacher-forcing gate.
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};
use switchyard_engine::{
    Error, Result,
    config::{Config, Precision},
    engine::{Engine, Generate, Generation},
    models,
    telemetry::Telemetry,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    requests: PathBuf,
    #[arg(long)]
    gate: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 5)]
    samples: usize,
}
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Independent,
    Cohorts,
    Merged,
}
#[derive(Serialize)]
struct Sample {
    mode: Mode,
    elapsed_s: f64,
    output_tokens: usize,
    output_tokens_per_s: f64,
    // Outer clock includes time spent waiting for earlier rows/cohorts.
    first_token_s: Vec<f64>,
    token_elapsed_s: Vec<Vec<f64>>,
    token_ids: Vec<Vec<u32>>,
}
fn run(engine: &Engine, requests: &[Generate], mode: Mode) -> Result<Sample> {
    let began = Instant::now();
    let mut times = vec![Vec::new(); requests.len()];
    let output: Vec<Generation> = match mode {
        Mode::Independent => {
            let mut all = Vec::new();
            for (row, request) in requests.iter().enumerate() {
                all.extend(engine.generate_observed(
                    std::slice::from_ref(request),
                    |_, _, _, _| {
                        times[row].push(began.elapsed().as_secs_f64());
                        true
                    },
                )?);
            }
            all
        }
        Mode::Cohorts => engine.generate_cohorts_observed(requests, |row, _, _, _| {
            times[row].push(began.elapsed().as_secs_f64());
            true
        })?,
        Mode::Merged => engine.generate_observed(requests, |row, _, _, _| {
            times[row].push(began.elapsed().as_secs_f64());
            true
        })?,
    };
    let elapsed_s = began.elapsed().as_secs_f64();
    let ids = output.into_iter().map(|o| o.token_ids).collect::<Vec<_>>();
    if ids.len() != requests.len()
        || ids
            .iter()
            .zip(&times)
            .any(|(ids, times)| ids.is_empty() || ids.len() != times.len())
    {
        return Err(Error::Config("Incomplete offline timing rows".into()));
    }
    let output_tokens = ids.iter().map(Vec::len).sum();
    Ok(Sample {
        mode,
        elapsed_s,
        output_tokens,
        output_tokens_per_s: output_tokens as f64 / elapsed_s,
        first_token_s: times.iter().map(|v| v[0]).collect(),
        token_elapsed_s: times,
        token_ids: ids,
    })
}
fn digest(path: &std::path::Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}
fn main() -> Result<()> {
    let args = Args::parse();
    let config: Config = serde_json::from_slice(&std::fs::read(&args.config)?)?;
    config.validate()?;
    let requests: Vec<Generate> = serde_json::from_slice(&std::fs::read(&args.requests)?)?;
    let gate: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.gate)?)?;
    if !matches!(config.dtype, Precision::F32)
        || requests.len() < 3
        || requests.len() > config.max_batch_size
        || args.samples == 0
        || args.samples > 31
        || requests.iter().any(|r| !r.ignore_eos)
        || gate["passed"] != true
        || gate["comparisons"]["independent"]["passed"] != true
        || gate["comparisons"]["published"]["passed"] != true
        || gate["requests"] != serde_json::to_value(&requests)?
        || gate["config"] != serde_json::to_value(&config)?
        || requests
            .iter()
            .map(|r| r.prompt.len())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            < 3
    {
        return Err(Error::Config(
            "Offline timing needs a passing published/independent F32 gate, varied prompts, explicit outputs and one bounded cohort".into(),
        ));
    }
    let telemetry = Telemetry::default();
    let model = models::load(&config, &telemetry)?;
    let engine = Engine::new(config.clone(), model, telemetry)?;
    for request in &requests {
        engine.validate(request)?;
    }
    let modes = [Mode::Independent, Mode::Cohorts, Mode::Merged];
    let mut warmups = Vec::new();
    for mode in modes {
        warmups.push(run(&engine, &requests, mode)?);
    }
    let gold = &warmups[0].token_ids;
    if warmups.iter().any(|s| &s.token_ids != gold) {
        return Err(Error::Config(
            "Full generated sequences differ before timing".into(),
        ));
    }
    let mut samples = Vec::new();
    for sample in 0..args.samples {
        let order = if sample % 2 == 0 {
            [0, 1, 2]
        } else {
            [2, 1, 0]
        };
        for index in order {
            let measured = run(&engine, &requests, modes[index])?;
            if &measured.token_ids != gold {
                return Err(Error::Config(
                    "Full generated sequences differ during timing".into(),
                ));
            }
            samples.push(measured);
        }
    }
    let mut summaries = Vec::new();
    for (index, mode) in modes.into_iter().enumerate() {
        let mut elapsed = samples
            .iter()
            .filter(|s| {
                matches!(
                    (s.mode, index),
                    (Mode::Independent, 0) | (Mode::Cohorts, 1) | (Mode::Merged, 2)
                )
            })
            .map(|s| s.elapsed_s)
            .collect::<Vec<_>>();
        elapsed.sort_by(f64::total_cmp);
        let middle = elapsed.len() / 2;
        let median = if elapsed.len() % 2 == 1 {
            elapsed[middle]
        } else {
            (elapsed[middle - 1] + elapsed[middle]) / 2.
        };
        summaries.push(serde_json::json!({"mode":mode,"median_elapsed_s":median,
            "output_tokens":warmups[0].output_tokens,
            "output_tokens_per_median_s":warmups[0].output_tokens as f64 / median}));
    }
    let receipt = serde_json::json!({
        "scope":"offline warm loaded-model F32 generation; excludes model loading and HTTP",
        "config":config,"requests":requests,"gate":gate,
        "gate_sha256":digest(&args.gate)?,"requests_sha256":digest(&args.requests)?,
        "binary_sha256":digest(&std::env::current_exe()?)?,
        "full_sequence_parity":true,"samples":samples,"warmups":warmups,"summary":summaries,
        "memory_note":"record host RSS and driver VRAM separately; model weights are loaded once and shared by all modes",
    });
    std::fs::write(args.output, serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string(&summaries)?);
    Ok(())
}
