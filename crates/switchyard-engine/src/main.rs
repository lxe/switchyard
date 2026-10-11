use clap::{Parser, Subcommand};
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
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// Override the configured local weight directory after artifact hydration.
    #[arg(long)]
    model_root: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
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
    let mut config: Config = serde_json::from_slice(&std::fs::read(&args.config)?)?;
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
    let engine = Engine::new(config.clone(), model, telemetry.clone());
    match args.command {
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
