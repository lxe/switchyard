//! Pack, verify, publish, and hydrate independently of fleet scheduling.
use crate::{
    Error, ProbeOptions, Result, diagnostics::HydrationTiming, probe, probe_directory,
    publish_directory, publish_extent_directory,
};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use switchyard_artifacts::{HydrationOptions, Manifest, Metadata, PackOptions};
use switchyard_provider::{StorageConfig, storage::Storage};

#[derive(Debug, Args)]
pub struct StorageSettings {
    /// JSON object containing a storage block (a fleet configuration is accepted).
    #[arg(long)]
    pub config: PathBuf,
    /// Read credential variables locally without changing the process environment.
    #[arg(long)]
    pub env_file: Vec<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Build a fixed-prefix serving runtime in Docker and export its filesystem tree.
    BuildRuntime {
        #[arg(long, default_value = ".")]
        context: PathBuf,
        #[arg(long, default_value = "deploy/hydration/Dockerfile.builder")]
        dockerfile: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Compose verified boot and sibling compiler/resource artifacts from a complete runtime.
    PartitionRuntime {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        recipe: PathBuf,
        /// Fresh local composition directory, published atomically after acceptance.
        #[arg(long)]
        out: PathBuf,
        /// Optional pinned source-patch lock, applied through the runtime builder before partitioning.
        #[arg(long)]
        patch_lock: Option<PathBuf>,
        #[arg(long, default_value = "switchyard/runtime-components")]
        prefix: String,
        #[arg(long, default_value_t = 6)]
        zstd_level: i32,
        #[arg(long, default_value = "ghcr.io/lxe/switchyard-runtime:ubuntu24.04")]
        acceptance_image: String,
        /// Produce source inventory and native closure without component packing or import acceptance.
        #[arg(long)]
        plan_only: bool,
    },
    /// Pack a filesystem tree into deterministic, verified zstd tar shards.
    Pack {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        metadata: PathBuf,
        /// Soft shard target; an individual large file remains intact and is streamed.
        #[arg(long, default_value_t = 134217728)]
        shard_bytes: u64,
        #[arg(long, default_value_t = 6)]
        zstd_level: i32,
    },
    /// Preserve file bytes in independent raw objects with a separate extent manifest.
    PackExtents {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        metadata: PathBuf,
        #[arg(long, default_value_t = 8388608)]
        extent_bytes: u64,
    },
    /// Verify a hydrated tree against its complete manifest, without network access.
    Verify {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        directory: PathBuf,
    },
    /// Publish chunks first and the manifest last, without overwriting existing objects.
    Publish {
        #[command(flatten)]
        storage: StorageSettings,
        #[arg(long)]
        directory: PathBuf,
        /// The artifact identity is appended to this namespace automatically.
        #[arg(long, default_value = "switchyard/artifacts")]
        prefix: String,
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Publish raw extent objects first and their distinct manifest last.
    PublishExtents {
        #[command(flatten)]
        storage: StorageSettings,
        #[arg(long)]
        directory: PathBuf,
        #[arg(long, default_value = "switchyard/weights")]
        prefix: String,
        #[arg(long, default_value_t = 20)]
        concurrency: usize,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Measure object download throughput without decoding, hashing, or body disk writes.
    Probe {
        #[command(flatten)]
        storage: StorageSettings,
        #[arg(long)]
        prefix: String,
        #[arg(long)]
        id: String,
        #[arg(long, default_value_t = 20)]
        concurrency: usize,
        #[arg(long, default_value_t = 600)]
        attempt_timeout_s: u64,
        #[arg(long, default_value_t = 30)]
        stall_timeout_s: u64,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Resolve scoped download URLs and hydrate a published artifact locally.
    Hydrate {
        #[command(flatten)]
        storage: StorageSettings,
        #[arg(long)]
        prefix: String,
        #[arg(long)]
        id: String,
        #[arg(long)]
        destination: PathBuf,
        #[arg(long)]
        cache: PathBuf,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        #[arg(long, default_value_t = 4)]
        extraction_concurrency: usize,
        /// Stream trusted, pinned shards directly into their final staging files without hashing.
        #[arg(long)]
        trusted_streaming: bool,
        /// Reserve compressed storage independently of extraction; requires trusted streaming and zero retries.
        #[arg(long)]
        receive_buffer: Option<PathBuf>,
        #[arg(long, default_value_t = 64 * 1024 * 1024 * 1024)]
        max_receive_bytes: u64,
        /// Space reserved for other simultaneous allocations on the output filesystem.
        #[arg(long, default_value_t = 0)]
        additional_output_bytes: u64,
        /// Total deadline for one download attempt, including response streaming.
        #[arg(long, default_value_t = 600)]
        attempt_timeout_s: u64,
        /// Maximum time without receiving another response-body chunk.
        #[arg(long, default_value_t = 30)]
        stall_timeout_s: u64,
        /// Additional attempts after a failed download. Use zero for unambiguous timing.
        #[arg(long, default_value_t = 2)]
        retries: usize,
        /// Write safe transfer/publication timing observations to a JSON receipt.
        #[arg(long)]
        receipt: Option<PathBuf>,
    },
}

#[derive(Deserialize)]
struct StorageInput {
    storage: StorageConfig,
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn storage(settings: &StorageSettings) -> Result<Storage> {
    let config: StorageInput = read(&settings.config)?;
    let mut values = BTreeMap::new();
    for path in &settings.env_file {
        for entry in dotenvy::from_path_iter(path)
            .map_err(|_| Error::Config("credential file cannot be read".into()))?
        {
            let (key, value) =
                entry.map_err(|_| Error::Config("invalid credential file".into()))?;
            values.entry(key).or_insert(value);
        }
    }
    let secret = |name: &str| {
        std::env::var(name)
            .ok()
            .or_else(|| values.get(name).cloned())
            .ok_or_else(|| Error::Config(format!("missing credential variable {name}")))
    };
    Ok(Storage::with_keys(
        &config.storage,
        secret(&config.storage.access_key_env)?,
        secret(&config.storage.secret_key_env)?,
    )?)
}

pub async fn run(command: Command) -> Result<()> {
    match command {
        Command::BuildRuntime {
            context,
            dockerfile,
            out,
        } => {
            crate::runtime_build::build(&context, &dockerfile, &out).await?;
            println!("Runtime tree exported to {}", out.join("runtime").display());
        }
        Command::PartitionRuntime {
            source,
            manifest,
            recipe,
            out,
            patch_lock,
            prefix,
            zstd_level,
            acceptance_image,
            plan_only,
        } => {
            let result = tokio::task::spawn_blocking(move || {
                crate::runtime_partition::build(crate::runtime_partition::Options {
                    source,
                    manifest,
                    recipe,
                    out,
                    patch_lock,
                    prefix,
                    zstd_level,
                    acceptance_image,
                    plan_only,
                })
            })
            .await??;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Pack {
            source,
            out,
            metadata,
            shard_bytes,
            zstd_level,
        } => {
            let metadata: Metadata = read(&metadata)?;
            let target = out.clone();
            let manifest = tokio::task::spawn_blocking(move || {
                switchyard_artifacts::pack(
                    &source,
                    &target,
                    metadata,
                    PackOptions {
                        target_shard_bytes: shard_bytes,
                        zstd_level,
                    },
                )
            })
            .await??;
            write(&out.join("manifest.json"), &manifest)?;
            println!(
                "{}: {} entries, {} shards, {} compressed bytes",
                manifest.id,
                manifest.entries.len(),
                manifest.chunks.len(),
                manifest.chunks.iter().map(|c| c.size).sum::<u64>()
            );
        }
        Command::PackExtents {
            source,
            out,
            metadata,
            extent_bytes,
        } => {
            let metadata: Metadata = read(&metadata)?;
            let target = out.clone();
            let manifest = tokio::task::spawn_blocking(move || {
                switchyard_artifacts::extents::pack_extents(
                    &source,
                    &target,
                    metadata,
                    extent_bytes,
                )
            })
            .await??;
            write(&out.join("manifest.json"), &manifest)?;
            println!(
                "{}: {} entries, {} raw extents",
                manifest.id,
                manifest.entries.len(),
                manifest.extents.len()
            );
        }
        Command::Verify {
            manifest,
            directory,
        } => {
            let manifest: Manifest = read(&manifest)?;
            let id = manifest.id.clone();
            tokio::task::spawn_blocking(move || {
                switchyard_artifacts::verify(&directory, &manifest)
            })
            .await??;
            println!("{id}: complete tree verified");
        }
        Command::Publish {
            storage: settings,
            directory,
            prefix,
            concurrency,
            receipt,
        } => {
            let store = storage(&settings)?;
            let published = publish_directory(&store, &directory, &prefix, concurrency).await?;
            write(&receipt, &published)?;
            println!("{} published at {}", published.id, published.prefix);
        }
        Command::PublishExtents {
            storage: settings,
            directory,
            prefix,
            concurrency,
            receipt,
        } => {
            let store = storage(&settings)?;
            let published =
                publish_extent_directory(&store, &directory, &prefix, concurrency).await?;
            write(&receipt, &published)?;
            println!("{} published at {}", published.id, published.prefix);
        }
        Command::Hydrate {
            storage: settings,
            prefix,
            id,
            destination,
            cache,
            concurrency,
            extraction_concurrency,
            trusted_streaming,
            receive_buffer,
            max_receive_bytes,
            additional_output_bytes,
            attempt_timeout_s,
            stall_timeout_s,
            retries,
            receipt,
        } => {
            probe::validate_reference(&prefix, &id)?;
            ProbeOptions {
                concurrency,
                request_timeout_s: attempt_timeout_s,
                stall_timeout_s,
            }
            .validate()?;
            if !(1..=64).contains(&extraction_concurrency) || retries > 10 {
                return Err(Error::Config(
                    "invalid hydration reference or concurrency".into(),
                ));
            }
            if receive_buffer.is_some() && (!trusted_streaming || retries != 0) {
                return Err(Error::Config(
                    "reserved reception requires trusted streaming and zero retries".into(),
                ));
            }
            let controller = Instant::now();
            let store = storage(&settings)?;
            let ticket = probe::resolve(&store, &prefix, &id).await?;
            let declared_bytes = ticket.manifest.chunks.iter().map(|chunk| chunk.size).sum();
            let options = HydrationOptions {
                mode: if trusted_streaming {
                    switchyard_artifacts::HydrationMode::TrustedStreaming
                } else {
                    switchyard_artifacts::HydrationMode::Verified
                },
                cache_dir: cache,
                download_concurrency: concurrency,
                extraction_concurrency,
                receive_buffer: receive_buffer.map(|directory| {
                    let mut options = switchyard_artifacts::ReceiveBufferOptions::new(directory);
                    options.max_reserved_bytes = max_receive_bytes;
                    options.additional_output_bytes = additional_output_bytes;
                    options
                }),
                request_timeout_s: attempt_timeout_s,
                stall_timeout_s,
                retries,
                ..Default::default()
            };
            let timing = receipt.as_ref().map(|_| {
                Arc::new(HydrationTiming::new(
                    id.clone(),
                    prefix,
                    declared_bytes,
                    controller.elapsed().as_secs_f64(),
                    &options,
                ))
            });
            let observed = timing.clone();
            let report =
                switchyard_artifacts::hydrate(ticket, &destination, options, move |progress| {
                    if let Some(observed) = &observed {
                        observed.observe(progress);
                    }
                })
                .await?;
            if let (Some(receipt), Some(timing)) = (receipt, timing) {
                write(&receipt, &timing.receipt(report.clone()))?;
            }
            println!(
                "{id}: {} bytes transferred, {} cached shards, already present {}",
                report.downloaded_bytes, report.cache_hits, report.already_present
            );
        }
        Command::Probe {
            storage: settings,
            prefix,
            id,
            concurrency,
            attempt_timeout_s,
            stall_timeout_s,
            receipt,
        } => {
            probe::validate_reference(&prefix, &id)?;
            let options = ProbeOptions {
                concurrency,
                request_timeout_s: attempt_timeout_s,
                stall_timeout_s,
            };
            options.validate()?;
            let store = storage(&settings)?;
            let report = probe_directory(&store, &prefix, &id, options).await?;
            write(&receipt, &report)?;
            println!(
                "{id}: {} bytes in {:.3}s, {:.2} MB/s ({:.3} Gbit/s), last body byte {:.3}s",
                report.downloaded_bytes,
                report.elapsed_s,
                report.bytes_per_s / 1_000_000.0,
                report.bytes_per_s * 8.0 / 1_000_000_000.0,
                report.last_body_elapsed_s.unwrap_or(0.0)
            );
        }
    }
    Ok(())
}
