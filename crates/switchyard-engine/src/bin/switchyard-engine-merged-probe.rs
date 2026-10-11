//! Offline mixed-decode numerical gate. Never invoked by production readiness.
use clap::Parser;
use std::path::PathBuf;
use switchyard_engine::{
    Result,
    config::Config,
    models,
    probe::{Trace, probe, probe_merged},
    telemetry::Telemetry,
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    independent_output: Option<PathBuf>,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let config: Config = serde_json::from_slice(&std::fs::read(args.config)?)?;
    let traces: Vec<Trace> = serde_json::from_slice(&std::fs::read(args.input)?)?;
    let telemetry = Telemetry::default();
    telemetry.event(
        "merged_probe_started",
        serde_json::json!({"scope":"offline teacher forcing","traces":traces.len()}),
    );
    let model = models::load(&config, &telemetry)?;
    let output = probe_merged(model.as_ref(), &config, &traces)?;
    std::fs::write(args.output, serde_json::to_vec_pretty(&output)?)?;
    if let Some(path) = args.independent_output {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&probe(model.as_ref(), &config, &traces)?)?,
        )?;
    }
    Ok(())
}
