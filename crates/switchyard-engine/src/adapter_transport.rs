//! Bounded PEFT transport. Uploaded files are private and removed after parsing,
//! including on cancellation. GPU transfer belongs to the inference owner.
use crate::{Error, Result, config::LoraConfig, models::lora::Adapter, telemetry::Telemetry};
use axum::extract::Multipart;
use serde::Deserialize;
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};
use tokio::{io::AsyncWriteExt, sync::OwnedSemaphorePermit};

pub(crate) struct PreparedAdapter {
    pub adapter: Adapter,
    // Retained through parsing and GPU attachment, even when HTTP disconnects.
    pub _permit: OwnedSemaphorePermit,
}

const MAX_CONFIG_BYTES: u64 = 64 * 1024;
static NEXT_UPLOAD: AtomicU64 = AtomicU64::new(1);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LoadRequest {
    pub lora_name: String,
    pub lora_path: Option<PathBuf>,
    /// Extension for object-store adapters: inline PEFT config and a signed
    /// URL for the safetensors file. URLs never enter cache keys or telemetry.
    pub adapter_config: Option<Value>,
    pub adapter_url: Option<String>,
}

pub(crate) fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-+".contains(&b))
    {
        return Err(Error::Config(
            "Adapter name must contain 1–128 ASCII letters, digits, '.', '_', '-' or '+'".into(),
        ));
    }
    Ok(())
}

struct Staging(PathBuf);
impl Staging {
    async fn new(root: &Path) -> Result<Self> {
        // Establish cleanup ownership synchronously: canceling detached Tokio
        // filesystem work must not create a directory after this future drops.
        std::fs::create_dir_all(root)?;
        let path = root.join(format!(
            ".upload-{}-{}",
            std::process::id(),
            NEXT_UPLOAD.fetch_add(1, Ordering::Relaxed)
        ));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        // Only directories this request created; never a caller's adapter tree.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
async fn parse(
    path: PathBuf,
    staging: Option<Staging>,
    config: LoraConfig,
    permit: OwnedSemaphorePermit,
) -> Result<PreparedAdapter> {
    tokio::task::spawn_blocking(move || {
        let _staging = staging;
        let adapter = Adapter::read(&path, config.max_upload_bytes, config.max_rank, "")?;
        Ok(PreparedAdapter {
            adapter,
            _permit: permit,
        })
    })
    .await
    .map_err(|_| Error::Config("Adapter parser stopped".into()))?
}

pub(crate) async fn load(
    request: &LoadRequest,
    config: &LoraConfig,
    telemetry: &Telemetry,
    permit: OwnedSemaphorePermit,
) -> Result<PreparedAdapter> {
    validate_name(&request.lora_name)?;
    let began = Instant::now();
    let operation = async {
        match (
            &request.lora_path,
            &request.adapter_config,
            &request.adapter_url,
        ) {
            (Some(path), None, None) => {
                let root = tokio::fs::canonicalize(&config.root).await?;
                let path = tokio::fs::canonicalize(path).await?;
                if !path.starts_with(&root) || !path.is_dir() {
                    return Err(Error::Config(
                        "lora_path must be a directory inside the configured adapter root".into(),
                    ));
                }
                Ok((path, None))
            }
            (None, Some(metadata), Some(url)) => {
                let parsed = reqwest::Url::parse(url)
                    .map_err(|_| Error::Config("Invalid adapter URL".into()))?;
                if !matches!(parsed.scheme(), "https" | "http")
                    || !parsed.username().is_empty()
                    || parsed.password().is_some()
                {
                    return Err(Error::Config(
                        "Adapter URL must use HTTP(S) without embedded credentials".into(),
                    ));
                }
                let metadata = serde_json::to_vec(metadata)?;
                if metadata.len() as u64 > MAX_CONFIG_BYTES
                    || metadata.len() as u64 >= config.max_upload_bytes
                {
                    return Err(Error::Config("Adapter config exceeds byte limit".into()));
                }
                let staging = Staging::new(&config.root).await?;
                tokio::fs::write(staging.0.join("adapter_config.json"), &metadata).await?;
                let client = reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .map_err(|_| {
                        Error::Config("Adapter HTTP client initialization failed".into())
                    })?;
                let mut response = client
                    .get(parsed)
                    .send()
                    .await
                    .map_err(|_| Error::Config("Adapter download failed".into()))?;
                if !response.status().is_success() {
                    return Err(Error::Config(format!(
                        "Adapter download returned HTTP {}",
                        response.status().as_u16()
                    )));
                }
                let max_weights = config.max_upload_bytes - metadata.len() as u64;
                if response.content_length().is_some_and(|n| n > max_weights) {
                    return Err(Error::Config("Adapter download exceeds byte limit".into()));
                }
                let mut file =
                    tokio::fs::File::create(staging.0.join("adapter_model.safetensors")).await?;
                let mut bytes = 0u64;
                while let Some(chunk) = response
                    .chunk()
                    .await
                    .map_err(|_| Error::Config("Adapter stream interrupted".into()))?
                {
                    bytes = bytes
                        .checked_add(chunk.len() as u64)
                        .ok_or_else(|| Error::Config("Adapter byte count overflow".into()))?;
                    if bytes > max_weights {
                        return Err(Error::Config("Adapter stream exceeds byte limit".into()));
                    }
                    file.write_all(&chunk).await?;
                }
                file.flush().await?;
                drop(file);
                let path = staging.0.clone();
                Ok((path, Some(staging)))
            }
            _ => Err(Error::Config(
                "Provide lora_path, or adapter_config and adapter_url together".into(),
            )),
        }
    };
    let (path, staging) =
        tokio::time::timeout(Duration::from_secs(config.transfer_timeout_s), operation)
            .await
            .map_err(|_| Error::Config("Adapter transfer deadline exceeded".into()))??;
    let prepared = parse(path, staging, config.clone(), permit).await?;
    let adapter = &prepared.adapter;
    telemetry.event("adapter_received", serde_json::json!({"name":request.lora_name,"bytes":adapter.bytes(),"elapsed_s":began.elapsed().as_secs_f64(),"revision":adapter.revision()}));
    Ok(prepared)
}

pub(crate) async fn upload(
    name: &str,
    mut multipart: Multipart,
    config: &LoraConfig,
    telemetry: &Telemetry,
    permit: OwnedSemaphorePermit,
) -> Result<PreparedAdapter> {
    validate_name(name)?;
    let began = Instant::now();
    let operation = async {
        let staging = Staging::new(&config.root).await?;
        let mut seen_config = false;
        let mut seen_weights = false;
        let mut total = 0u64;
        while let Some(mut field) = multipart
            .next_field()
            .await
            .map_err(|_| Error::Config("Invalid adapter multipart stream".into()))?
        {
            let is_config = match field.name() {
                Some("adapter_config") if !seen_config => {
                    seen_config = true;
                    true
                }
                Some("adapter_model") if !seen_weights => {
                    seen_weights = true;
                    false
                }
                _ => {
                    return Err(Error::Config(
                        "Expected exactly adapter_config and adapter_model multipart fields".into(),
                    ));
                }
            };
            let path = staging.0.join(if is_config {
                "adapter_config.json"
            } else {
                "adapter_model.safetensors"
            });
            let mut file = tokio::fs::File::create(path).await?;
            let mut field_bytes = 0u64;
            while let Some(chunk) = field
                .chunk()
                .await
                .map_err(|_| Error::Config("Adapter upload interrupted".into()))?
            {
                field_bytes = field_bytes
                    .checked_add(chunk.len() as u64)
                    .ok_or_else(|| Error::Config("Adapter byte count overflow".into()))?;
                total = total
                    .checked_add(chunk.len() as u64)
                    .ok_or_else(|| Error::Config("Adapter byte count overflow".into()))?;
                if total > config.max_upload_bytes || (is_config && field_bytes > MAX_CONFIG_BYTES)
                {
                    return Err(Error::Config("Adapter upload exceeds byte limit".into()));
                }
                file.write_all(&chunk).await?;
            }
            file.flush().await?;
        }
        if !seen_config || !seen_weights {
            return Err(Error::Config(
                "Adapter upload requires config and safetensors".into(),
            ));
        }
        let path = staging.0.clone();
        Ok((path, Some(staging)))
    };
    let (path, staging) =
        tokio::time::timeout(Duration::from_secs(config.transfer_timeout_s), operation)
            .await
            .map_err(|_| Error::Config("Adapter upload deadline exceeded".into()))??;
    let prepared = parse(path, staging, config.clone(), permit).await?;
    let adapter = &prepared.adapter;
    telemetry.event("adapter_received", serde_json::json!({"name":name,"bytes":adapter.bytes(),"elapsed_s":began.elapsed().as_secs_f64(),"revision":adapter.revision()}));
    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, Bytes},
        http::{Response, StatusCode},
        routing::get,
    };
    use candle_core::{Device, Tensor};
    use std::{collections::HashMap, sync::Arc};
    use tokio::sync::{Notify, Semaphore};

    struct Fixture {
        path: PathBuf,
        root: PathBuf,
        metadata: Value,
        weights: Bytes,
    }
    impl Fixture {
        fn new() -> Self {
            static ID: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "switchyard-adapter-transport-{}-{}",
                std::process::id(),
                ID.fetch_add(1, Ordering::Relaxed)
            ));
            let root = path.join("allowed");
            std::fs::create_dir_all(root.join("local")).unwrap();
            std::fs::create_dir(path.join("outside")).unwrap();
            let metadata = serde_json::json!({"peft_type":"LORA","base_model_name_or_path":"fixture/base","r":1,"lora_alpha":1.,"target_modules":["q_proj"],"bias":"none"});
            std::fs::write(
                root.join("local/adapter_config.json"),
                serde_json::to_vec(&metadata).unwrap(),
            )
            .unwrap();
            let tensors = HashMap::from([
                (
                    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
                    Tensor::new(&[[0.25_f32, -0.5]], &Device::Cpu).unwrap(),
                ),
                (
                    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
                    Tensor::new(&[[0.75_f32], [1.]], &Device::Cpu).unwrap(),
                ),
            ]);
            candle_core::safetensors::save(&tensors, root.join("local/adapter_model.safetensors"))
                .unwrap();
            let weights =
                Bytes::from(std::fs::read(root.join("local/adapter_model.safetensors")).unwrap());
            Self {
                path,
                root,
                metadata,
                weights,
            }
        }
        fn config(&self) -> LoraConfig {
            LoraConfig {
                root: self.root.clone(),
                base_model_aliases: vec!["fixture/base".into()],
                max_loaded: 2,
                max_rank: 8,
                max_upload_bytes: 1 << 20,
                max_resident_bytes: 1 << 20,
                transfer_timeout_s: 2,
            }
        }
        fn request(&self, url: String) -> LoadRequest {
            LoadRequest {
                lora_name: "tiny-adapter".into(),
                lora_path: None,
                adapter_config: Some(self.metadata.clone()),
                adapter_url: Some(url),
            }
        }
        fn no_staging(&self) {
            assert!(
                !std::fs::read_dir(&self.root).unwrap().any(|entry| entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".upload-")),
                "owned staging directory was retained"
            );
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    struct Server {
        url: String,
        task: tokio::task::JoinHandle<()>,
        stalled: Arc<Notify>,
    }
    impl Server {
        async fn new(weights: Bytes) -> Self {
            let stalled = Arc::new(Notify::new());
            let app = Router::new()
                .route(
                    "/valid",
                    get(move || {
                        let weights = weights.clone();
                        async move {
                            Response::builder()
                                .header("content-length", weights.len())
                                .body(Body::from(weights))
                                .unwrap()
                        }
                    }),
                )
                .route(
                    "/declared-large",
                    get(|| async {
                        Response::builder()
                            .header("content-length", 256)
                            .body(Body::from(vec![0_u8; 256]))
                            .unwrap()
                    }),
                )
                .route(
                    "/streamed-large",
                    get(|| async {
                        Body::from_stream(futures_util::stream::iter([
                            Ok::<_, std::io::Error>(Bytes::from_static(b"small")),
                            Ok(Bytes::from(vec![0_u8; 256])),
                        ]))
                    }),
                )
                .route(
                    "/stalled",
                    get({
                        let stalled = stalled.clone();
                        move || {
                            let stalled = stalled.clone();
                            async move {
                                Body::from_stream(futures_util::stream::unfold(0_u8, move |step| {
                                    let stalled = stalled.clone();
                                    async move {
                                        if step == 0 {
                                            Some((
                                                Ok::<_, std::io::Error>(Bytes::from_static(
                                                    b"first-bytes",
                                                )),
                                                1,
                                            ))
                                        } else {
                                            stalled.notify_one();
                                            std::future::pending::<
                                                Option<(
                                                    std::result::Result<Bytes, std::io::Error>,
                                                    u8,
                                                )>,
                                            >()
                                            .await
                                        }
                                    }
                                }))
                            }
                        }
                    }),
                )
                .route(
                    "/redirect",
                    get(|| async {
                        Response::builder()
                            .status(StatusCode::FOUND)
                            .header("location", "/valid")
                            .body(Body::empty())
                            .unwrap()
                    }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            Self { url, task, stalled }
        }
    }
    impl Drop for Server {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    #[tokio::test]
    async fn valid_url_preserves_peft_identity_cleans_staging_and_holds_admission_permit() {
        let fixture = Fixture::new();
        let server = Server::new(fixture.weights.clone()).await;
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let request = fixture.request(format!("{}/valid?X-Amz-Signature=fixture", server.url));
        let prepared = load(&request, &fixture.config(), &Telemetry::default(), permit)
            .await
            .unwrap();
        let local = Adapter::read(&fixture.root.join("local"), 1 << 20, 8, "fixture/base").unwrap();
        assert_eq!(prepared.adapter.revision(), local.revision());
        assert_eq!(prepared.adapter.rank(), 1);
        assert_eq!(prepared.adapter.bytes(), 16);
        assert_eq!(semaphore.available_permits(), 0);
        fixture.no_staging();
        assert!(
            fixture
                .root
                .join("local/adapter_model.safetensors")
                .is_file()
        );
        drop(prepared);
        assert_eq!(semaphore.available_permits(), 1);
    }

    #[tokio::test]
    async fn declared_and_streamed_oversize_and_redirects_reject_without_retaining_staging() {
        let fixture = Fixture::new();
        let server = Server::new(fixture.weights.clone()).await;
        let mut config = fixture.config();
        config.max_upload_bytes = serde_json::to_vec(&fixture.metadata).unwrap().len() as u64 + 32;
        for (route, expected) in [
            ("declared-large", "download exceeds"),
            ("streamed-large", "stream exceeds"),
            ("redirect", "HTTP 302"),
        ] {
            let semaphore = Arc::new(Semaphore::new(1));
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let request = fixture.request(format!("{}/{route}", server.url));
            let error = load(&request, &config, &Telemetry::default(), permit)
                .await
                .err()
                .unwrap();
            assert!(error.to_string().contains(expected), "{route}: {error}");
            fixture.no_staging();
            assert_eq!(semaphore.available_permits(), 1);
        }
    }

    #[tokio::test]
    async fn local_directory_and_symlink_escapes_are_rejected_without_touching_caller_files() {
        let fixture = Fixture::new();
        let mut paths = vec![fixture.path.join("outside")];
        #[cfg(unix)]
        {
            let escape = fixture.root.join("escape");
            std::os::unix::fs::symlink(fixture.path.join("outside"), &escape).unwrap();
            paths.push(escape);
        }
        for path in paths {
            let semaphore = Arc::new(Semaphore::new(1));
            let request = LoadRequest {
                lora_name: "escape".into(),
                lora_path: Some(path),
                adapter_config: None,
                adapter_url: None,
            };
            let error = load(
                &request,
                &fixture.config(),
                &Telemetry::default(),
                semaphore.clone().acquire_owned().await.unwrap(),
            )
            .await
            .err()
            .unwrap();
            assert!(error.to_string().contains("inside the configured"));
            assert_eq!(semaphore.available_permits(), 1);
            assert!(fixture.path.join("outside").is_dir());
            fixture.no_staging();
        }
    }

    #[tokio::test]
    async fn canceling_a_live_download_removes_written_staging_and_releases_its_permit() {
        let fixture = Fixture::new();
        let server = Server::new(fixture.weights.clone()).await;
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let request = fixture.request(format!("{}/stalled", server.url));
        let config = fixture.config();
        let task =
            tokio::spawn(
                async move { load(&request, &config, &Telemetry::default(), permit).await },
            );
        tokio::time::timeout(Duration::from_secs(2), server.stalled.notified())
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let received = std::fs::read_dir(&fixture.root)
                    .unwrap()
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| entry.file_name().to_string_lossy().starts_with(".upload-"))
                    .any(|entry| {
                        std::fs::metadata(entry.path().join("adapter_model.safetensors"))
                            .is_ok_and(|metadata| metadata.len() > 0)
                    });
                if received {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(semaphore.available_permits(), 0);
        task.abort();
        assert!(matches!(task.await, Err(error) if error.is_cancelled()));
        fixture.no_staging();
        assert_eq!(semaphore.available_permits(), 1);
    }

    #[test]
    fn canceled_parser_waiter_cannot_release_a_still_owned_blocking_parse_permit() {
        let fixture = Fixture::new();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let started = Arc::new(Notify::new());
            let (release, blocked) = std::sync::mpsc::channel();
            let blocker = tokio::task::spawn_blocking({
                let started = started.clone();
                move || {
                    started.notify_one();
                    blocked.recv().unwrap();
                }
            });
            started.notified().await;
            let semaphore = Arc::new(Semaphore::new(1));
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let task = tokio::spawn(parse(
                fixture.root.join("local"),
                None,
                fixture.config(),
                permit,
            ));
            // The parser has moved its permit into a blocking closure queued
            // behind our controlled worker, before the waiter is canceled.
            tokio::task::yield_now().await;
            task.abort();
            assert!(matches!(task.await, Err(error) if error.is_cancelled()));
            assert_eq!(semaphore.available_permits(), 0);
            release.send(()).unwrap();
            blocker.await.unwrap();
            tokio::time::timeout(Duration::from_secs(2), async {
                while semaphore.available_permits() == 0 {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            assert_eq!(semaphore.available_permits(), 1);
        });
    }
}
