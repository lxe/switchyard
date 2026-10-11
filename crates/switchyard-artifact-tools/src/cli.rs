//! Pack, verify, publish, and hydrate independently of fleet scheduling.
use crate::{Error, Result, publication::Snapshot};
use clap::{Args, Subcommand};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use switchyard_artifacts::{ArtifactTicket, HydrationOptions, Manifest, Metadata, PackOptions};
use switchyard_provider::{StorageConfig, config::valid_key_prefix, storage::Storage};

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
        /// Total deadline for one download attempt, including response streaming.
        #[arg(long, default_value_t = 600)]
        attempt_timeout_s: u64,
        /// Maximum time without receiving another response-body chunk.
        #[arg(long, default_value_t = 30)]
        stall_timeout_s: u64,
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
            if !valid_key_prefix(&prefix) || !(1..=32).contains(&concurrency) {
                return Err(Error::Config(
                    "invalid publication prefix or concurrency".into(),
                ));
            }
            let snapshot = Snapshot::capture(&directory.join("manifest.json"))?;
            let manifest = &snapshot.manifest;
            let store = storage(&settings)?;
            let object_prefix = format!("{prefix}/{}", manifest.id);
            let operations = futures_util::stream::iter(manifest.chunks.iter().cloned())
                .map(|chunk| {
                    let store = store.clone();
                    let key = format!("{object_prefix}/{}", chunk.name);
                    let path = directory.join(&chunk.name);
                    async move {
                        Ok::<_, Error>((
                            chunk.name,
                            store
                                .put_file_immutable(&key, &path, &chunk.sha256, chunk.size)
                                .await?,
                        ))
                    }
                })
                .buffer_unordered(concurrency);
            tokio::pin!(operations);
            let mut chunks = BTreeMap::new();
            while let Some(result) = operations.next().await {
                let (name, created) = result?;
                chunks.insert(name, created);
            }
            let created = store
                .put_file_immutable(
                    &format!("{object_prefix}/manifest.json"),
                    &snapshot.path,
                    &snapshot.sha256,
                    snapshot.size,
                )
                .await?;
            write(
                &receipt,
                &serde_json::json!({"id":manifest.id,"prefix":object_prefix,"manifest_created":created,"chunks_created":chunks}),
            )?;
            println!("{} published at {object_prefix}", manifest.id);
        }
        Command::Hydrate {
            storage: settings,
            prefix,
            id,
            destination,
            cache,
            concurrency,
            extraction_concurrency,
            attempt_timeout_s,
            stall_timeout_s,
        } => {
            if !valid_key_prefix(&prefix)
                || id.len() != 64
                || !id.bytes().all(|c| c.is_ascii_hexdigit())
                || !(1..=64).contains(&concurrency)
                || !(1..=64).contains(&extraction_concurrency)
                || !(1..=86400).contains(&attempt_timeout_s)
                || !(1..=86400).contains(&stall_timeout_s)
            {
                return Err(Error::Config(
                    "invalid hydration reference or concurrency".into(),
                ));
            }
            let store = storage(&settings)?;
            let manifest: Manifest = store.get_json(&format!("{prefix}/manifest.json")).await?;
            manifest.validate()?;
            if manifest.id != id {
                return Err(Error::Config(
                    "published artifact differs from pinned identity".into(),
                ));
            }
            let urls = manifest
                .chunks
                .iter()
                .map(|c| {
                    Ok((
                        c.name.clone(),
                        store.presign(&format!("{prefix}/{}", c.name))?,
                    ))
                })
                .collect::<switchyard_provider::Result<_>>()?;
            let options = HydrationOptions {
                cache_dir: cache,
                download_concurrency: concurrency,
                extraction_concurrency,
                request_timeout_s: attempt_timeout_s,
                stall_timeout_s,
                ..Default::default()
            };
            let report = switchyard_artifacts::hydrate(
                ArtifactTicket { manifest, urls },
                &destination,
                options,
                |_| {},
            )
            .await?;
            println!(
                "{id}: {} bytes transferred, {} cached shards, already present {}",
                report.downloaded_bytes, report.cache_hits, report.already_present
            );
        }
    }
    Ok(())
}
