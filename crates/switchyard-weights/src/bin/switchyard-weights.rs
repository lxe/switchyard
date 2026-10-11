//! Experimental local serving bridge. Transport and readiness run in Rust.
use clap::{Parser, Subcommand};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::{fs::PermissionsExt, io::AsRawFd},
    path::PathBuf,
    time::{Duration, Instant},
};
use switchyard_artifacts::{
    Error, HydrationMode, Metadata, Result,
    extents::{ExtentHydration, ExtentOptions, ExtentTicket, pack_extents},
};
use switchyard_weights::WeightStream;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader},
    net::UnixListener,
};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
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
struct SocketCleanup(PathBuf);
impl Drop for SocketCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
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
        Command::Pack {
            source,
            output,
            extent_bytes,
        } => {
            let manifest = pack_extents(
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
                    request_timeout: Duration::from_secs(timeout_seconds),
                    ..ExtentOptions::default()
                },
            )?;
            // Bind never removes/replaces an existing socket. Caller owns this directory.
            let listener = UnixListener::bind(&socket)?;
            let _cleanup = SocketCleanup(socket.clone());
            fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
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
                let (connection, _) = listener.accept().await?;
                let _ = connected.send(());
                let (input, mut output) = connection.into_split();
                let mut input = BufReader::new(input);
                let mut plan = WeightStream::new(&hydration, 100_000_000)?;
                let mut files = BTreeMap::new();
                loop {
                    let mut command = [0_u8; 5];
                    input.read_exact(&mut command).await?;
                    if &command != b"next\n" {
                        return Err(Error::Invalid("weights bridge command".into()));
                    }
                    // The protocol permits one outstanding request. Observe EOF
                    // while waiting for data, so a dead consumer cancels held HTTP.
                    let mut unexpected = [0_u8; 1];
                    let ready = tokio::select! {
                        result = plan.next_ready(&hydration) => result?,
                        result = input.read(&mut unexpected) => {
                            if result? == 0 { return Err(Error::Cancelled); }
                            return Err(Error::Invalid("pipelined weights bridge command".into()));
                        }
                    };
                    let Some(ready) = ready else {
                        output.write_all(b"{\"done\":true}\n").await?;
                        // Acknowledgment comes after all CPU/GPU reads and parameter assignment.
                        let mut command = [0_u8; 7];
                        input.read_exact(&mut command).await?;
                        if &command != b"finish\n" {
                            return Err(Error::Cancelled);
                        }
                        eprintln!(
                            "{}",
                            serde_json::json!({"event":"consumer_complete", "seconds":start.elapsed().as_secs_f64()})
                        );
                        return Ok(());
                    };
                    let file = match files.entry(ready.tensor.entry) {
                        std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
                        std::collections::btree_map::Entry::Vacant(e) => {
                            e.insert(ready.bytes.open_file()?)
                        }
                    };
                    // proc fd names the same inode across publication; the Rust owner
                    // retains it until the consumer acknowledges its completed loads.
                    let path = format!("/proc/{}/fd/{}", std::process::id(), file.as_raw_fd());
                    let message = serde_json::json!({"tensor":ready.tensor,"path":path,"ready_seconds":start.elapsed().as_secs_f64()});
                    output.write_all(&serde_json::to_vec(&message)?).await?;
                    output.write_all(b"\n").await?;
                }
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
