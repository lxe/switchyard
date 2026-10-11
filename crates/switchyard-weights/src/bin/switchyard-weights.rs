//! Experimental local serving bridge. Transport and readiness run in Rust.
use clap::{Parser, Subcommand};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};
use switchyard_artifacts::{
    Error, HydrationMode, Metadata, Result,
    extents::{ExtentHydration, ExtentOptions, ExtentTicket},
};
use switchyard_weights::{WeightBridge, pack_weights, priority_extents};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Print the exact serving adapter fingerprint for a selected loader config.
    LoaderIdentity {
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Create a raw extent manifest and content-addressed objects; no upload.
    Pack {
        source: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t=128*1024*1024)]
        extent_bytes: u64,
    },
    /// Hydrate and serve one local tensor consumer over a private Unix socket.
    Serve {
        ticket: PathBuf,
        destination: PathBuf,
        socket: PathBuf,
        #[arg(long, default_value_t = 20)]
        concurrency: usize,
        #[arg(long)]
        trusted: bool,
        /// Defer transfer until the loader connects; useful for controlled overlap measurements.
        #[arg(long)]
        wait_for_consumer: bool,
        #[arg(long, default_value_t = 120)]
        timeout_seconds: u64,
    },
}
#[tokio::main]
async fn main() {
    if let Err(error) = execute().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
async fn execute() -> Result<()> {
    match Cli::parse().command {
        Command::LoaderIdentity { config } => {
            let loader: switchyard_weights::VllmLoaderConfig = match config {
                Some(path) => serde_json::from_slice(&fs::read(path)?)?,
                None => Default::default(),
            };
            println!("{}", loader.identity()?);
        }
        Command::Pack {
            source,
            output,
            extent_bytes,
        } => {
            let manifest = pack_weights(
                &source,
                &output,
                Metadata {
                    kind: "weights".into(),
                    compatibility: BTreeMap::new(),
                    attributes: BTreeMap::new(),
                },
                extent_bytes,
            )?;
            let file = output.join(format!("{}.manifest.json", manifest.id));
            fs::write(&file, serde_json::to_vec_pretty(&manifest)?)?;
            fs::write(
                output.join("manifest.json"),
                serde_json::to_vec_pretty(&manifest)?,
            )?;
            println!("{}", file.display());
        }
        Command::Serve {
            ticket,
            destination,
            socket,
            concurrency,
            trusted,
            wait_for_consumer,
            timeout_seconds,
        } => {
            let ticket: ExtentTicket = serde_json::from_slice(&fs::read(ticket)?)?;
            let priorities = priority_extents(&ticket.manifest);
            let hydration = ExtentHydration::prepare(
                ticket,
                &destination,
                ExtentOptions {
                    mode: if trusted {
                        HydrationMode::TrustedStreaming
                    } else {
                        HydrationMode::Verified
                    },
                    concurrency,
                    priority_extents: priorities,
                    request_timeout: Duration::from_secs(timeout_seconds),
                    ..ExtentOptions::default()
                },
            )?;
            // Bind never removes/replaces an existing socket. Caller owns this directory.
            let bridge = WeightBridge::bind(&socket)?;
            let start = Instant::now();
            let (connected, begin) = tokio::sync::oneshot::channel();
            let network = async {
                if wait_for_consumer {
                    begin.await.map_err(|_| Error::Cancelled)?;
                }
                let network_start = Instant::now();
                let report = hydration.run().await?;
                eprintln!(
                    "{}",
                    serde_json::json!({"event":"hydration_complete", "seconds":start.elapsed().as_secs_f64(), "hydration_seconds":network_start.elapsed().as_secs_f64(), "report":report})
                );
                Ok::<_, Error>(())
            };
            let serving = async {
                bridge.serve(&hydration,start,||{let _=connected.send(());},|p|{
                    eprintln!("{}",serde_json::json!({"event":"tensor_progress","seconds":start.elapsed().as_secs_f64(),"report":p}));
                }).await?;
                eprintln!(
                    "{}",
                    serde_json::json!({"event":"consumer_complete","seconds":start.elapsed().as_secs_f64()})
                );
                Ok::<_, Error>(())
            };
            tokio::time::timeout(Duration::from_secs(timeout_seconds), async {
                tokio::try_join!(network, serving).map(|_| ())
            })
            .await
            .map_err(|_| Error::Transfer)??;
        }
    }
    Ok(())
}
