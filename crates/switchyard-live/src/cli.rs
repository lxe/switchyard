//! Reproducible live experiments and exact-ownership recovery.
use crate::config::DeploymentConfig;
use crate::{Error, LiveRuntime, Result, planner::Message};
use clap::{Args, Subcommand};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use switchyard_core::Policy;
use switchyard_provider::{Provider, RunPod, storage::Storage};

#[derive(Debug, Args)]
pub struct Settings {
    #[arg(long, default_value = "configs/runpod/qwen-small.json")]
    pub config: PathBuf,
    /// Read credentials locally without changing the process environment.
    #[arg(long)]
    pub env_file: Vec<PathBuf>,
    #[arg(long, default_value = "results/live")]
    pub out: PathBuf,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Query available GPUs and timestamped rental quotes; does not provision.
    Discover {
        #[command(flatten)]
        settings: Settings,
    },
    /// Check pinned archives, prompt tokenization, and available capacity without provisioning.
    Check {
        #[command(flatten)]
        settings: Settings,
    },
    /// Run configured workloads, record streaming usage and timing, and terminate owned pods.
    Gauntlet {
        #[command(flatten)]
        settings: Settings,
        #[arg(long, default_value = "configs/runpod/gauntlet.json")]
        workloads: PathBuf,
        /// Restrict this run to a configured model ID.
        #[arg(long)]
        model: Option<String>,
    },
    /// Prepare compiler caches on one GPU, release it, then publish the artifact.
    Bake {
        #[command(flatten)]
        settings: Settings,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long, default_value = "switchyard/artifacts/cache")]
        namespace: String,
    },
    /// Terminate only pods recorded in this run's ownership journals.
    Recover {
        #[command(flatten)]
        settings: Settings,
    },
}

pub use crate::workloads::{Case, Experiment};

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("partial");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
fn provider_inputs(
    settings: &Settings,
) -> Result<(DeploymentConfig, RunPod, BTreeMap<String, String>)> {
    let config: DeploymentConfig = read(&settings.config)?;
    config.validate()?;
    let mut values = BTreeMap::new();
    for file in &settings.env_file {
        let entries = dotenvy::from_path_iter(file)
            .map_err(|_| Error::Config("credential file cannot be read".into()))?;
        for entry in entries {
            let (key, value) =
                entry.map_err(|_| Error::Config("invalid credential file".into()))?;
            values.entry(key).or_insert(value);
        }
    }
    let provider =
        RunPod::with_config(&config.compute, &secret(&values, &config.compute.key_env)?)?;
    Ok((config, provider, values))
}
fn secret(values: &BTreeMap<String, String>, name: &str) -> Result<String> {
    std::env::var(name)
        .ok()
        .or_else(|| values.get(name).cloned())
        .ok_or_else(|| Error::Config(format!("missing credential variable {name}")))
}
fn clients(settings: &Settings) -> Result<(DeploymentConfig, RunPod, Storage)> {
    let (config, provider, values) = provider_inputs(settings)?;
    let storage = Storage::with_keys(
        &config.storage,
        secret(&values, &config.storage.access_key_env)?,
        secret(&values, &config.storage.secret_key_env)?,
    )?;
    Ok((config, provider, storage))
}

pub async fn run(command: Command) -> Result<()> {
    match command {
        Command::Discover { settings } => {
            let (_, provider, _) = provider_inputs(&settings)?;
            let offerings = provider.offers().await?;
            save(&settings.out.join("offerings.json"), &offerings)?;
            for offer in offerings {
                println!(
                    "{} / {}: ${:.4}/hour including configured disk",
                    offer.gpu,
                    offer.data_center,
                    offer.total_usd_per_hour()
                );
            }
        }
        Command::Check { settings } => {
            let (config, provider, storage) = clients(&settings)?;
            let runtime = LiveRuntime::new(
                provider,
                storage,
                config.execution(),
                Policy::Adaptive,
                &settings.out,
            )
            .await?;
            let mut checks = vec![];
            for model in runtime.model_ids() {
                let count = runtime.plan_input(
                    model,
                    &[Message {
                        role: "user".into(),
                        content: "Explain GPU inference batching in one sentence.".into(),
                    }],
                )?;
                checks.push(serde_json::json!({"model":model,"input_tokens":count}));
                println!("{model}: manifest validated, prompt plans {count} input tokens");
            }
            save(
                &settings.out.join("preflight.json"),
                &serde_json::json!({"provisioned":false,"checks":checks}),
            )?;
        }
        Command::Gauntlet {
            settings,
            workloads,
            model,
        } => {
            let experiment: Experiment = read(&workloads)?;
            experiment.validate()?;
            let (config, _, _) = clients(&settings)?;
            let experiment_cap = config
                .limits
                .max_experiment_usd
                .unwrap_or(config.limits.max_run_usd);
            let experiment_timeout = config
                .limits
                .experiment_timeout_s
                .unwrap_or(config.limits.run_timeout_s);
            let began = std::time::Instant::now();
            let mut total_rental = 0.0;
            let models = config
                .models
                .iter()
                .filter(|m| model.as_ref().is_none_or(|id| id == &m.id))
                .map(|m| m.id.clone())
                .collect::<Vec<_>>();
            if models.is_empty() {
                return Err(Error::Config(
                    "requested model is absent from fleet configuration".into(),
                ));
            }
            save(&settings.out.join("workloads.json"), &experiment)?;
            save(&settings.out.join("fleet-source.json"), &config)?;
            use sha2::Digest;
            let source_commit = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_owned());
            save(
                &settings.out.join("run-source.json"),
                &serde_json::json!({"version":env!("CARGO_PKG_VERSION"),"source_commit":source_commit,
                "fleet_sha256":format!("{:x}",sha2::Sha256::digest(std::fs::read(&settings.config)?)),
                "workloads_sha256":format!("{:x}",sha2::Sha256::digest(std::fs::read(&workloads)?))}),
            )?;
            for model in models {
                let (mut config, provider, storage) = clients(&settings)?;
                let remaining_usd = experiment_cap - total_rental;
                let remaining_seconds =
                    experiment_timeout.saturating_sub(began.elapsed().as_secs());
                if remaining_usd <= 0.0 || remaining_seconds == 0 {
                    return Err(Error::Limit("global experiment budget exhausted".into()));
                }
                config.limits.max_run_usd = config.limits.max_run_usd.min(remaining_usd);
                config.limits.run_timeout_s = config.limits.run_timeout_s.min(remaining_seconds);
                config.models.retain(|m| m.id == model);
                let out = settings.out.join(&model);
                let mut runtime = LiveRuntime::new(
                    provider,
                    storage,
                    config.execution(),
                    Policy::Adaptive,
                    &out,
                )
                .await?;
                let outcome = tokio::select! {
                    result = tokio::time::timeout(std::time::Duration::from_secs(experiment_timeout).saturating_sub(began.elapsed()),execute(&mut runtime, &experiment, &model)) =>
                        result.unwrap_or_else(|_|Err(Error::Limit("global experiment timeout".into()))),
                    _ = tokio::signal::ctrl_c() => Err(Error::Limit("interrupted by user".into())),
                };
                let cleanup = runtime.cleanup().await;
                total_rental += runtime.rental_usd();
                save(
                    &settings.out.join("experiment-receipt.json"),
                    &serde_json::json!({"estimated_rental_usd":total_rental,"elapsed_s":began.elapsed().as_secs_f64(),"max_experiment_usd":experiment_cap,"experiment_timeout_s":experiment_timeout}),
                )?;
                save(
                    &out.join("outcome.json"),
                    &serde_json::json!({"error":outcome.as_ref().err().map(ToString::to_string),"cleanup_error":cleanup.as_ref().err().map(ToString::to_string)}),
                )?;
                cleanup?;
                outcome?;
            }
        }
        Command::Recover { settings } => {
            let (_, provider, storage) = clients(&settings)?;
            recover(&provider, &storage, &settings.out).await?;
        }
        Command::Bake {
            settings,
            plan,
            namespace,
        } => {
            let plan: crate::bake::CacheBakePlan = read(&plan)?;
            let (mut config, provider, storage) = clients(&settings)?;
            plan.validate(&config.execution())?;
            if !config.compute.gpu_types.contains(&plan.gpu) {
                return Err(Error::Config(
                    "cache preparation GPU is absent from fleet configuration".into(),
                ));
            }
            config.models.retain(|model| model.id == plan.model);
            config.limits.max_instances = 1;
            config.limits.max_run_usd = config.limits.max_run_usd.min(
                config
                    .limits
                    .max_experiment_usd
                    .unwrap_or(config.limits.max_run_usd),
            );
            let timeout = config.limits.run_timeout_s;
            let mut runtime = LiveRuntime::new(
                provider,
                storage.clone(),
                config.execution(),
                Policy::WarmOnly,
                &settings.out,
            )
            .await?;
            let outcome = tokio::select! {
                result = tokio::time::timeout(std::time::Duration::from_secs(timeout), runtime.bake_cache(&plan, &namespace)) =>
                    result.unwrap_or_else(|_| Err(Error::Limit("cache preparation run timeout".into()))),
                _ = tokio::signal::ctrl_c() => Err(Error::Limit("interrupted by user".into())),
            };
            // No publication work may extend the lifetime of the paid host.
            let cleanup = runtime.cleanup().await;
            save(
                &settings.out.join("outcome.json"),
                &serde_json::json!({
                    "error":outcome.as_ref().err().map(ToString::to_string),
                    "cleanup_error":cleanup.as_ref().err().map(ToString::to_string),
                    "estimated_rental_usd":runtime.rental_usd(),
                }),
            )?;
            cleanup?;
            let prepared = outcome?;
            let publication = switchyard_artifact_tools::publish_directory(
                &storage,
                &prepared.directory,
                &namespace,
                plan.download_concurrency,
            )
            .await?;
            save(&settings.out.join("cache-publication.json"), &publication)?;
            save(
                &settings.out.join("cache-configuration.json"),
                &prepared.configuration,
            )?;
            println!(
                "Published cache {} after confirmed GPU release; estimated rent ${:.4}",
                publication.id,
                runtime.rental_usd()
            );
        }
    }
    Ok(())
}

async fn execute<P: Provider>(
    runtime: &mut LiveRuntime<P>,
    experiment: &Experiment,
    model: &str,
) -> Result<()> {
    for case in &experiment.cases {
        runtime.set_policy(case.policy);
        let calls = case.requests(model);
        runtime.capture_metrics(&case.name, "before").await?;
        println!(
            "{model} / {}: {} calls using {}",
            case.name, case.calls, case.policy
        );
        let report = runtime
            .run_workload(
                &case.name,
                &calls,
                case.desired_output_tps,
                case.completion_budget_s,
            )
            .await?;
        runtime.capture_metrics(&case.name, "after").await?;
        println!(
            "{}: {}/{} complete, {:.1} output tokens/s, TTFT p95 {:?}s, estimated rent ${:.4}",
            case.name,
            report.completed,
            report.requests,
            report.output_tokens_per_s,
            report.ttft_p95_s,
            report.estimated_rental_usd
        );
        if report.failed > 0
            || report.unfinished > 0
            || report.budget_exceeded
            || report.invalid_actions > 0
        {
            return Err(Error::Request(
                "live workload failed; detailed receipt saved".into(),
            ));
        }
    }
    Ok(())
}

async fn recover<P: Provider>(provider: &P, storage: &Storage, out: &Path) -> Result<()> {
    let path = out.join("owned-pods.json");
    if path.exists() {
        let pods: Vec<crate::runtime::OwnedPod> = read(&path)?;
        for owned in pods {
            if let Some(pod) = provider.get(&owned.pod.id).await? {
                if pod.name != owned.pod.name || !pod.name.starts_with("switchyard-") {
                    return Err(Error::Config(
                        "ownership mismatch in recovery journal".into(),
                    ));
                }
                provider.terminate(&pod.id).await?;
                if provider
                    .get(&pod.id)
                    .await?
                    .is_some_and(|p| !p.status.is_terminated())
                {
                    return Err(Error::Request(
                        "termination awaiting confirmation; rerun recovery".into(),
                    ));
                }
            }
            println!("{} termination confirmed", owned.pod.name);
        }
    }
    let path = out.join("pending-launches.json");
    if path.exists() {
        let launches: Vec<crate::runtime::PendingLaunch> = read(&path)?;
        let confirmed = if out.join("owned-pods.json").exists() {
            read::<Vec<crate::runtime::OwnedPod>>(&out.join("owned-pods.json"))?
                .into_iter()
                .map(|p| p.pod.name)
                .collect()
        } else {
            std::collections::BTreeSet::new()
        };
        let mut launches = launches;
        crate::ownership::reconcile(provider, storage, &mut launches, &confirmed).await?;
        save(&path, &launches)?;
    }
    save(
        &out.join("recovery-receipt.json"),
        &serde_json::json!({"termination_confirmed":true}),
    )?;
    Ok(())
}
