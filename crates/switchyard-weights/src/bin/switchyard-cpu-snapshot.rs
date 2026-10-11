//! Standalone engine-specific CPU checkpoint adapter; no generic agent API change.
#[cfg(target_os = "linux")]
fn main() {
    use clap::{Parser, Subcommand};
    use std::path::PathBuf;
    use switchyard_weights::cpu_snapshot;
    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        command: Action,
    }
    #[derive(Subcommand)]
    enum Action {
        Prepare {
            #[arg(long)]
            runtime_root: PathBuf,
            #[arg(long)]
            runtime_manifest: PathBuf,
            /// Exact native launch arguments and preparation environment JSON.
            #[arg(long)]
            serving_config: PathBuf,
            #[arg(long)]
            out: PathBuf,
            #[arg(long, default_value = "/usr/bin/criu")]
            criu: PathBuf,
        },
        Restore {
            #[arg(long)]
            snapshot: PathBuf,
            #[arg(long)]
            runtime_root: PathBuf,
            #[arg(long)]
            runtime_id: String,
            #[arg(long)]
            work: PathBuf,
            #[arg(long, default_value = "/usr/bin/criu")]
            criu: PathBuf,
            #[arg(long)]
            probe: bool,
            #[arg(last = true)]
            args: Vec<String>,
        },
    }
    let result = match Cli::parse().command {
        Action::Prepare {
            runtime_root,
            runtime_manifest,
            serving_config,
            out,
            criu,
        } => cpu_snapshot::prepare(cpu_snapshot::Prepare {
            runtime_root,
            runtime_manifest,
            serving_config,
            out,
            criu,
        }),
        Action::Restore {
            snapshot,
            runtime_root,
            runtime_id,
            work,
            criu,
            probe,
            args,
        } => cpu_snapshot::restore(cpu_snapshot::Restore {
            snapshot,
            runtime_root,
            runtime_id,
            work,
            criu,
            probe,
            args,
        }),
    };
    if let Err(error) = result {
        eprintln!("CPU snapshot: {error}");
        std::process::exit(1);
    }
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("CRIU CPU snapshots require Linux");
    std::process::exit(1);
}
