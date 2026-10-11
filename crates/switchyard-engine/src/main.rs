use clap::{ArgGroup, Parser, Subcommand};
use std::{net::SocketAddr, path::PathBuf};
use switchyard_engine::{
    Result,
    config::Config,
    engine::{Engine, Generate},
    models, server,
    telemetry::Telemetry,
    tokenizer::Tokenizer,
};

#[derive(Parser)]
#[command(group(ArgGroup::new("configuration").required(true).args(["config", "config_json"])))]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    /// Inline fixed configuration supplied by a serving adapter.
    #[arg(long)]
    config_json: Option<String>,
    /// Override the configured local weight directory after artifact hydration.
    #[arg(long)]
    model_root: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Probe {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        layers_output: Option<PathBuf>,
    },
    Prepare {
        #[arg(long)]
        output: PathBuf,
    },
    Tokenize {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 16)]
        max_tokens: usize,
    },
    Generate {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Serve {
        #[arg(long, default_value = "127.0.0.1:8100")]
        bind: SocketAddr,
    },
}
#[tokio::main]
async fn main() -> Result<()> {
    let telemetry = Telemetry::default();
    telemetry.event("process_started", serde_json::json!({"engine":"rust"}));
    let args = Args::parse();
    let mut config: Config = match (args.config, args.config_json) {
        (Some(path), None) => serde_json::from_slice(&std::fs::read(path)?)?,
        (None, Some(json)) => serde_json::from_str(&json)?,
        _ => unreachable!("Clap requires exactly one configuration source"),
    };
    if let Some(root) = args.model_root {
        config.model_root = root;
    }
    config.validate()?;
    if let Command::Prepare { output } = &args.command {
        switchyard_engine::weights::Weights::read(
            &config.model_root,
            config.max_weight_bytes,
            &telemetry,
        )?
        .prepare(&config.model_root, output)?;
        telemetry.event("weights_prepared", serde_json::json!({"output":output}));
        return Ok(());
    }
    if let Command::Tokenize {
        input,
        output,
        max_tokens,
    } = &args.command
    {
        let tokenizer = Tokenizer::load(&config.model_root)?;
        let chats: Vec<Vec<switchyard_engine::tokenizer::Message>> =
            serde_json::from_slice(&std::fs::read(input)?)?;
        let requests: Vec<Generate> = chats
            .iter()
            .map(|messages| {
                Ok(Generate {
                    prompt: tokenizer.chat(messages)?,
                    max_tokens: *max_tokens,
                    ignore_eos: true,
                })
            })
            .collect::<Result<_>>()?;
        std::fs::write(output, serde_json::to_vec_pretty(&requests)?)?;
        return Ok(());
    }
    let tokenizer_task = if matches!(args.command, Command::Serve { .. }) {
        let root = config.model_root.clone();
        let progress = telemetry.clone();
        Some(std::thread::spawn(move || {
            let tokenizer = Tokenizer::load(&root)?;
            progress.event("tokenizer_loaded", serde_json::json!({}));
            Ok::<_, switchyard_engine::Error>(tokenizer)
        }))
    } else {
        None
    };
    let model = models::load(&config, &telemetry)?;
    if let Command::Probe {
        input,
        output,
        layers_output,
    } = &args.command
    {
        let traces: Vec<switchyard_engine::probe::Trace> =
            serde_json::from_slice(&std::fs::read(input)?)?;
        let result = switchyard_engine::probe::probe(model.as_ref(), &config, &traces)?;
        std::fs::write(output, serde_json::to_vec_pretty(&result)?)?;
        if let Some(path) = layers_output {
            let first = traces
                .first()
                .ok_or_else(|| switchyard_engine::Error::Config("No diagnostic trace".into()))?;
            let tokens = candle_core::Tensor::from_vec(
                first.prompt.clone(),
                (1, first.prompt.len()),
                model.device(),
            )?;
            let layers = model
                .inspect_layers(&tokens)?
                .iter()
                .map(|(name, x)| {
                    Ok((
                        name.clone(),
                        x.to_dtype(candle_core::DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    ))
                })
                .collect::<candle_core::Result<std::collections::BTreeMap<_, _>>>()?;
            std::fs::write(path, serde_json::to_vec(&layers)?)?;
        }
        return Ok(());
    }
    let engine = Engine::new(config.clone(), model, telemetry.clone())?;
    match args.command {
        Command::Probe { .. } => unreachable!("Diagnostic probe returned before serving"),
        Command::Prepare { .. } | Command::Tokenize { .. } => {
            unreachable!("Tokenization returned before GPU initialization")
        }
        Command::Generate { input, output } => {
            let requests: Vec<Generate> = serde_json::from_slice(&std::fs::read(input)?)?;
            let result = engine.generate(&requests)?;
            let bytes = serde_json::to_vec_pretty(&result)?;
            if let Some(output) = output {
                std::fs::write(output, &bytes)?;
            } else {
                println!("{}", String::from_utf8_lossy(&bytes));
            }
        }
        Command::Serve { bind } => {
            let tokenizer = tokenizer_task
                .expect("Serving starts tokenizer preparation")
                .join()
                .map_err(|_| {
                    switchyard_engine::Error::Config("Tokenizer preparation panicked".into())
                })??;
            server::serve(engine, config, tokenizer, bind, telemetry).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exactly_one_config_source_is_required() {
        assert!(Args::try_parse_from(["engine", "serve"]).is_err());
        assert!(
            Args::try_parse_from([
                "engine",
                "--config",
                "a.json",
                "--config-json",
                "{}",
                "serve"
            ])
            .is_err()
        );
        assert!(Args::try_parse_from(["engine", "--config-json", "{}", "serve"]).is_ok());
        assert!(Args::try_parse_from(["engine", "--config", "a.json", "serve"]).is_ok());
    }
}
