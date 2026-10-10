use clap::{Parser, Subcommand};
use inference_fleet_sim::{POLICIES, Policy, Profile, Scenario, report, scenarios, simulate};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    version,
    about = "Offline deterministic inference fleet scheduler simulator"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure wall-clock software overhead on analytical workloads.
    Bench {
        #[arg(long, value_delimiter = ',', default_value = "1,100,1000,10000")]
        sizes: Vec<usize>,
        #[arg(long, default_value_t = 3)]
        repetitions: usize,
        /// Optional host-specific regression budget per full run.
        #[arg(long)]
        max_seconds: Option<f64>,
        #[arg(long, default_value = "results/performance.json")]
        out: PathBuf,
    },
    /// Serve a generic GPU-cloud API backed by the incremental event simulator.
    Serve {
        #[arg(long)]
        profiles: Option<PathBuf>,
        #[arg(long)]
        environment: Option<PathBuf>,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: std::net::SocketAddr,
        /// Simulated seconds per real second; zero enables manual clock control.
        #[arg(long, default_value_t = 1.0)]
        clock_rate: f64,
    },
    /// Fuzz all built-in controllers; save any failing configuration for replay.
    Fuzz {
        #[arg(long, default_value_t = 1000)]
        cases: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "results/fuzz")]
        out: PathBuf,
    },
    /// Export editable generic profile and workload examples.
    Init {
        #[arg(long, default_value = "data")]
        out: PathBuf,
    },
    /// Run all built-in workloads against five policies and emit detailed results.
    Suite {
        #[arg(long)]
        profiles: Option<PathBuf>,
        #[arg(long, default_value = "results")]
        out: PathBuf,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value_t = 1)]
        seeds: usize,
        #[arg(long)]
        scenario: Option<String>,
    },
    /// Replay an arbitrary JSON trace using a selected policy.
    Run {
        #[arg(long)]
        profiles: PathBuf,
        #[arg(long)]
        scenario: PathBuf,
        #[arg(long, value_enum, default_value = "planned-hybrid")]
        policy: Policy,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value = "results/custom.json")]
        out: PathBuf,
    },
    /// Validate profiles and a trace without simulating.
    Validate {
        #[arg(long)]
        profiles: PathBuf,
        #[arg(long)]
        scenario: PathBuf,
    },
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn save<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::Bench {
            sizes,
            repetitions,
            max_seconds,
            out,
        } => {
            if max_seconds.is_some_and(|t| !t.is_finite() || t <= 0.0) {
                return Err("--max-seconds must be finite and positive".into());
            }
            let mut records = vec![];
            for count in sizes {
                for policy in POLICIES {
                    records.push(inference_fleet_sim::performance::measure(
                        count,
                        policy,
                        repetitions,
                    )?);
                }
            }
            save(&out, &records)?;
            for r in &records {
                println!(
                    "{} requests, {}: p50 {:.2} ms, p95 {:.2} ms, {:.0} simulated requests/wall second",
                    r.requests,
                    r.policy,
                    r.wall_ms_p50,
                    r.wall_ms_p95,
                    r.simulated_requests_per_wall_second
                );
            }
            if records
                .iter()
                .any(|r| max_seconds.is_some_and(|limit| r.wall_ms_p95 > limit * 1000.0))
            {
                return Err("performance budget exceeded; measurement receipt saved".into());
            }
        }
        Command::Serve {
            profiles,
            environment,
            seed,
            bind,
            clock_rate,
        } => {
            let profiles = profiles
                .as_deref()
                .map(read)
                .transpose()?
                .unwrap_or_else(scenarios::default_profiles);
            let environment = environment
                .as_deref()
                .map(read)
                .transpose()?
                .unwrap_or_default();
            println!("Fleet API listening on http://{bind}; clock rate {clock_rate}");
            inference_fleet_sim::http::serve_with_config(
                profiles,
                environment,
                seed,
                bind,
                clock_rate,
            )
            .await?;
        }
        Command::Fuzz { cases, seed, out } => {
            let mut failures = vec![];
            for index in 0..cases {
                let case_seed = seed.checked_add(index as u64).ok_or("seed overflow")?;
                let (profiles, scenario) = inference_fleet_sim::fuzz::case(case_seed);
                for policy in POLICIES {
                    let run = simulate(&profiles, &scenario, policy, case_seed)?;
                    if let Err(error) = inference_fleet_sim::fuzz::check(&run, &profiles, &scenario)
                    {
                        let folder = out.join(format!("failure-{case_seed}-{policy}"));
                        save(&folder.join("profiles.json"), &profiles)?;
                        save(&folder.join("scenario.json"), &scenario)?;
                        save(&folder.join("run.json"), &run)?;
                        failures.push(serde_json::json!({"seed": case_seed, "policy": policy, "error": error}));
                    }
                }
            }
            let receipt = serde_json::json!({"cases": cases, "policies": POLICIES.len(), "simulations": cases * POLICIES.len(), "first_seed": seed, "failures": failures});
            save(&out.join("receipt.json"), &receipt)?;
            println!("{receipt}");
            if !failures.is_empty() {
                return Err("fuzz invariants failed; replay artifacts saved".into());
            }
        }
        Command::Init { out } => {
            let profiles = scenarios::default_profiles();
            save(&out.join("profiles.json"), &profiles)?;
            save(
                &out.join("environment.json"),
                &inference_fleet_sim::CloudConfig::default(),
            )?;
            for scenario in scenarios::suite(&profiles) {
                save(&out.join(format!("{}.json", scenario.name)), &scenario)?;
            }
            println!(
                "Editable profiles and workload traces written to {}",
                out.display()
            );
        }
        Command::Suite {
            profiles,
            out,
            seed,
            seeds,
            scenario,
        } => {
            if seeds == 0 {
                return Err("--seeds must be positive".into());
            }
            let profiles = profiles
                .as_deref()
                .map(read)
                .transpose()?
                .unwrap_or_else(scenarios::default_profiles);
            let mut runs = vec![];
            let mut matched = false;
            for workload in scenarios::suite(&profiles) {
                if scenario.as_ref().is_some_and(|name| *name != workload.name) {
                    continue;
                }
                matched = true;
                for offset in 0..seeds {
                    let case_seed = seed.checked_add(offset as u64).ok_or("seed overflow")?;
                    for policy in POLICIES {
                        let run = simulate(&profiles, &workload, policy, case_seed)?;
                        save(
                            &out.join(format!("{}-{policy}-{case_seed}.json", workload.name)),
                            &run,
                        )?;
                        runs.push(run);
                    }
                }
            }
            if !matched {
                return Err("unknown scenario; run init to see available names".into());
            }
            fs::create_dir_all(&out)?;
            let table = report::markdown(&runs);
            fs::write(out.join("comparison.md"), &table)?;
            save(
                &out.join("summaries.json"),
                &runs.iter().map(|r| &r.summary).collect::<Vec<_>>(),
            )?;
            save(&out.join("profiles-used.json"), &profiles)?;
            print!("{table}");
        }
        Command::Run {
            profiles,
            scenario,
            policy,
            seed,
            out,
        } => {
            let profiles: Vec<Profile> = read(&profiles)?;
            let scenario: Scenario = read(&scenario)?;
            let run = simulate(&profiles, &scenario, policy, seed)?;
            save(&out, &run)?;
            print!("{}", report::markdown(&[run]));
        }
        Command::Validate { profiles, scenario } => {
            inference_fleet_sim::validate(
                &read::<Vec<Profile>>(&profiles)?,
                &read::<Scenario>(&scenario)?,
            )?;
            println!("Configuration is valid");
        }
    }
    Ok(())
}
