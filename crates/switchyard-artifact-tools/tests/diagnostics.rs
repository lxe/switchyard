//! Download-only measurements and hydration timing through a local S3-compatible server.
use axum::{
    Router,
    body::{Body, Bytes},
    extract::{Path as ObjectPath, State},
    routing::get,
};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use switchyard_artifact_tools::{
    HydrationReceipt, ProbeOptions, ProbeReceipt,
    cli::{Command, StorageSettings, run},
    probe_directory,
};
use switchyard_artifacts::{Manifest, Metadata, PackOptions, pack};
use switchyard_provider::{StorageConfig, storage::Storage};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let mut random = [0_u8; 12];
        getrandom::fill(&mut random).unwrap();
        let nonce = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let path = std::env::temp_dir().join(format!("switchyard-diagnostics-test-{nonce}"));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct Downloads {
    objects: Arc<BTreeMap<String, Vec<u8>>>,
    header_delay: Duration,
    body_delay: Duration,
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    requests: Arc<AtomicUsize>,
}
struct Active(Arc<AtomicUsize>);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}
async fn download(State(state): State<Downloads>, ObjectPath(key): ObjectPath<String>) -> Body {
    let bytes = state.objects[&key].clone();
    if key.ends_with("manifest.json") {
        return Body::from(bytes);
    }
    state.requests.fetch_add(1, Ordering::Relaxed);
    let active = state.active.fetch_add(1, Ordering::Relaxed) + 1;
    state.peak.fetch_max(active, Ordering::Relaxed);
    let ownership = Active(state.active.clone());
    tokio::time::sleep(state.header_delay).await;
    let tail = Bytes::copy_from_slice(&bytes[1..]);
    let head = Bytes::copy_from_slice(&bytes[..1]);
    let first = futures_util::stream::once(async move { Ok::<_, std::io::Error>(head) });
    let last = futures_util::stream::once(async move {
        tokio::time::sleep(state.body_delay).await;
        drop(ownership);
        Ok::<_, std::io::Error>(tail)
    });
    use futures_util::StreamExt;
    Body::from_stream(first.chain(last))
}

struct Fixture {
    scratch: Scratch,
    manifest: Manifest,
    storage: Storage,
    config: PathBuf,
    credentials: PathBuf,
    state: Downloads,
    server: tokio::task::JoinHandle<()>,
}
impl Fixture {
    async fn new(
        count: usize,
        header_delay: Duration,
        body_delay: Duration,
        size_adjustment: i8,
        corrupt: bool,
    ) -> Self {
        let scratch = Scratch::new();
        let source = scratch.0.join("source");
        fs::create_dir(&source).unwrap();
        for index in 0..count {
            fs::write(
                source.join(format!("payload-{index}")),
                format!("payload-{index}").repeat(64),
            )
            .unwrap();
        }
        let packed = scratch.0.join("packed");
        let manifest = pack(
            &source,
            &packed,
            Metadata {
                kind: "diagnostic-test".into(),
                compatibility: BTreeMap::new(),
                attributes: BTreeMap::new(),
            },
            PackOptions {
                target_shard_bytes: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(manifest.chunks.len(), count);
        let mut objects = BTreeMap::from([(
            "bucket/artifacts/manifest.json".into(),
            serde_json::to_vec(&manifest).unwrap(),
        )]);
        for chunk in &manifest.chunks {
            let mut bytes = fs::read(packed.join(&chunk.name)).unwrap();
            if corrupt {
                bytes.fill(b'x');
            }
            if size_adjustment < 0 {
                bytes.pop();
            }
            if size_adjustment > 0 {
                bytes.push(0);
            }
            objects.insert(format!("bucket/artifacts/{}", chunk.name), bytes);
        }
        let state = Downloads {
            objects: Arc::new(objects),
            header_delay,
            body_delay,
            active: Arc::default(),
            peak: Arc::default(),
            requests: Arc::default(),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let settings = StorageConfig {
            endpoint: format!("http://{}", listener.local_addr().unwrap()),
            region: "auto".into(),
            bucket: "bucket".into(),
            access_key_env: "SWITCHYARD_DIAGNOSTIC_TEST_ACCESS".into(),
            secret_key_env: "SWITCHYARD_DIAGNOSTIC_TEST_SECRET".into(),
            url_lifetime_s: 900,
        };
        let app = Router::new()
            .route("/{*key}", get(download))
            .with_state(state.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let storage =
            Storage::with_keys(&settings, "test-access".into(), "test-secret".into()).unwrap();
        let config = scratch.0.join("storage.json");
        fs::write(
            &config,
            serde_json::to_vec(&serde_json::json!({"storage":settings})).unwrap(),
        )
        .unwrap();
        let credentials = scratch.0.join("credentials.env");
        fs::write(&credentials, "SWITCHYARD_DIAGNOSTIC_TEST_ACCESS=test-access\nSWITCHYARD_DIAGNOSTIC_TEST_SECRET=test-secret\n").unwrap();
        Self {
            scratch,
            manifest,
            storage,
            config,
            credentials,
            state,
            server,
        }
    }
    fn settings(&self) -> StorageSettings {
        StorageSettings {
            config: self.config.clone(),
            env_file: vec![self.credentials.clone()],
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

#[tokio::test]
async fn memory_probe_runs_twenty_parallel_downloads_and_separates_header_and_body_timing() {
    // Equal-length invalid zstd bodies deliberately prove there is no hash or decoder work.
    let fixture = Fixture::new(
        24,
        Duration::from_millis(50),
        Duration::from_millis(100),
        0,
        true,
    )
    .await;
    let before = fs::read_dir(&fixture.scratch.0).unwrap().count();
    let receipt = probe_directory(
        &fixture.storage,
        "artifacts",
        &fixture.manifest.id,
        ProbeOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(receipt.shards.len(), 24);
    assert_eq!(receipt.downloaded_bytes, receipt.declared_bytes);
    assert_eq!(fixture.state.peak.load(Ordering::Relaxed), 20);
    assert_eq!(fixture.state.requests.load(Ordering::Relaxed), 24);
    assert_eq!(
        fs::read_dir(&fixture.scratch.0).unwrap().count(),
        before,
        "probe writes no files"
    );
    assert!(
        receipt
            .shards
            .iter()
            .all(|shard| shard.header_latency_s >= 0.035
                && shard.last_body_elapsed_s.unwrap() - shard.first_body_elapsed_s.unwrap()
                    >= 0.075)
    );
    assert!(receipt.first_headers_elapsed_s <= receipt.first_body_elapsed_s.unwrap());
    assert!(receipt.last_body_elapsed_s.unwrap() <= receipt.last_response_completion_elapsed_s);
    assert!(receipt.last_response_completion_elapsed_s <= receipt.elapsed_s);
    assert_eq!(
        receipt.bytes_per_s,
        receipt.downloaded_bytes as f64 / receipt.elapsed_s
    );
    let json = serde_json::to_string(&receipt).unwrap();
    assert!(
        !json.contains("test-access")
            && !json.contains("test-secret")
            && !json.contains("X-Amz")
            && !json.contains("http://")
    );
    assert_eq!(
        serde_json::from_str::<ProbeReceipt>(&json).unwrap(),
        receipt
    );
}

#[tokio::test]
async fn probe_cli_writes_receipt_and_rejects_pin_or_invalid_limits_before_chunk_downloads() {
    let fixture = Fixture::new(2, Duration::ZERO, Duration::ZERO, 0, true).await;
    for options in [
        ProbeOptions {
            concurrency: 0,
            ..Default::default()
        },
        ProbeOptions {
            request_timeout_s: 0,
            ..Default::default()
        },
        ProbeOptions {
            stall_timeout_s: 86401,
            ..Default::default()
        },
    ] {
        assert!(
            probe_directory(&fixture.storage, "artifacts", &fixture.manifest.id, options)
                .await
                .is_err()
        );
    }
    assert!(
        probe_directory(
            &fixture.storage,
            "../artifacts",
            &fixture.manifest.id,
            ProbeOptions::default()
        )
        .await
        .is_err()
    );
    assert!(
        probe_directory(
            &fixture.storage,
            "artifacts",
            &"a".repeat(64),
            ProbeOptions::default()
        )
        .await
        .is_err()
    );
    assert_eq!(fixture.state.requests.load(Ordering::Relaxed), 0);
    let path = fixture.scratch.0.join("probe.json");
    run(Command::Probe {
        storage: fixture.settings(),
        prefix: "artifacts".into(),
        id: fixture.manifest.id.clone(),
        concurrency: 20,
        attempt_timeout_s: 10,
        stall_timeout_s: 2,
        receipt: path.clone(),
    })
    .await
    .unwrap();
    let receipt: ProbeReceipt = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(receipt.id, fixture.manifest.id);
    assert_eq!(receipt.shards.len(), 2);
}

#[tokio::test]
async fn probe_enforces_exact_lengths_stall_timeout_and_complete_request_deadline() {
    async fn length(adjustment: i8) {
        let fixture = Fixture::new(1, Duration::ZERO, Duration::ZERO, adjustment, true).await;
        let error = probe_directory(
            &fixture.storage,
            "artifacts",
            &fixture.manifest.id,
            ProbeOptions::default(),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains(if adjustment < 0 { "shorter" } else { "exceeds" }));
        assert!(!error.contains("X-Amz") && !error.contains("test-secret"));
        assert_eq!(
            fixture.state.requests.load(Ordering::Relaxed),
            1,
            "probe does not retry"
        );
    }
    async fn timeout(header: Duration, body: Duration, attempt: u64, stall: u64, expected: &str) {
        let fixture = Fixture::new(1, header, body, 0, true).await;
        let error = tokio::time::timeout(
            Duration::from_secs(10),
            probe_directory(
                &fixture.storage,
                "artifacts",
                &fixture.manifest.id,
                ProbeOptions {
                    concurrency: 1,
                    request_timeout_s: attempt,
                    stall_timeout_s: stall,
                },
            ),
        )
        .await
        .expect("probe deadline must terminate while the server remains blocked")
        .unwrap_err()
        .to_string();
        assert!(
            error.contains(expected),
            "unexpected deadline failure: {error}"
        );
        assert_eq!(fixture.state.requests.load(Ordering::Relaxed), 1);
    }
    // A held server response distinguishes a real deadline from a late response.
    // The outer guard bounds regressions without a narrow wall-clock speed requirement.
    let held = Duration::from_secs(30);
    tokio::join!(
        length(-1),
        length(1),
        timeout(held, Duration::ZERO, 1, 5, "request failed or timed out"),
        timeout(held, Duration::ZERO, 5, 1, "response headers stalled"),
        timeout(Duration::ZERO, held, 5, 1, "response body stalled"),
        timeout(
            Duration::ZERO,
            held,
            1,
            5,
            "response body failed or timed out"
        )
    );
}

#[tokio::test]
async fn hydrate_cli_records_qualified_transfer_tail_and_warm_reuse_without_network() {
    let fixture = Fixture::new(
        2,
        Duration::from_millis(10),
        Duration::from_millis(30),
        0,
        false,
    )
    .await;
    let destination = fixture.scratch.0.join("hydrated");
    let cache = fixture.scratch.0.join("cache");
    let receipt = fixture.scratch.0.join("hydrate.json");
    for (repeat, retries) in [(false, 0), (true, 0), (false, 2)] {
        let target = if retries == 0 {
            destination.clone()
        } else {
            fixture.scratch.0.join("retry-enabled")
        };
        run(Command::Hydrate {
            storage: fixture.settings(),
            prefix: "artifacts".into(),
            id: fixture.manifest.id.clone(),
            destination: target,
            cache: cache.clone(),
            concurrency: 20,
            extraction_concurrency: 20,
            trusted_streaming: true,
            receive_buffer: None,
            max_receive_bytes: 64 << 30,
            additional_output_bytes: 0,
            attempt_timeout_s: 10,
            stall_timeout_s: 2,
            retries,
            receipt: Some(receipt.clone()),
        })
        .await
        .unwrap();
        let metrics: HydrationReceipt =
            serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(metrics.already_present, repeat);
        assert_eq!(metrics.verified_bytes, 0);
        assert_eq!(metrics.network_verified_bytes, 0);
        assert!(!cache.exists());
        if repeat {
            assert_eq!(metrics.downloaded_bytes, 0);
            assert!(metrics.first_network_progress_elapsed_s.is_none());
            assert!(metrics.all_network_bytes_observed_elapsed_s.is_none());
            assert!(metrics.post_transfer_observation_s.is_none());
        } else if retries > 0 {
            assert_eq!(metrics.downloaded_bytes, metrics.declared_bytes);
            assert!(metrics.first_network_progress_elapsed_s.is_some());
            assert!(metrics.all_network_bytes_observed_elapsed_s.is_none());
            assert!(metrics.post_transfer_observation_s.is_none());
            assert!(metrics.timing_qualification.contains("retries_enabled"));
        } else {
            assert_eq!(metrics.downloaded_bytes, metrics.declared_bytes);
            let first = metrics.first_network_progress_elapsed_s.unwrap();
            let complete = metrics.all_network_bytes_observed_elapsed_s.unwrap();
            assert!(first <= complete && complete <= metrics.hydration_elapsed_s);
            assert!(metrics.post_transfer_observation_s.unwrap() >= 0.0);
            assert!(metrics.timing_qualification.contains("upper_bound"));
        }
    }
    assert_eq!(fixture.state.requests.load(Ordering::Relaxed), 4);
    switchyard_artifacts::verify(&destination, &fixture.manifest).unwrap();
}

#[tokio::test]
async fn cancelling_probe_drops_its_transfer_queue_without_more_requests_or_disk_writes() {
    let fixture = Fixture::new(24, Duration::ZERO, Duration::from_secs(2), 0, true).await;
    let storage = fixture.storage.clone();
    let id = fixture.manifest.id.clone();
    let before = fs::read_dir(&fixture.scratch.0).unwrap().count();
    let task = tokio::spawn(async move {
        probe_directory(
            &storage,
            "artifacts",
            &id,
            ProbeOptions {
                concurrency: 3,
                ..Default::default()
            },
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(2), async {
        while fixture.state.requests.load(Ordering::Relaxed) != 3 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::sleep(Duration::from_millis(25)).await;
    assert_eq!(fixture.state.requests.load(Ordering::Relaxed), 3);
    assert_eq!(fs::read_dir(&fixture.scratch.0).unwrap().count(), before);
}

#[tokio::test]
async fn reserved_hydration_cli_records_independent_reception_and_reclamation() {
    let fixture = Fixture::new(
        2,
        Duration::from_millis(10),
        Duration::from_millis(30),
        0,
        false,
    )
    .await;
    let destination = fixture.scratch.0.join("buffered-output");
    let spool = fixture.scratch.0.join("receive-buffer");
    let receipt = fixture.scratch.0.join("buffered-receipt.json");
    run(Command::Hydrate {
        storage: fixture.settings(),
        prefix: "artifacts".into(),
        id: fixture.manifest.id.clone(),
        destination: destination.clone(),
        cache: fixture.scratch.0.join("unused-cache"),
        concurrency: 20,
        extraction_concurrency: 1,
        trusted_streaming: true,
        receive_buffer: Some(spool.clone()),
        max_receive_bytes: 16 << 20,
        additional_output_bytes: 0,
        attempt_timeout_s: 10,
        stall_timeout_s: 2,
        retries: 0,
        receipt: Some(receipt.clone()),
    })
    .await
    .unwrap();
    let receipt: HydrationReceipt = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
    let reception = receipt.reception.unwrap();
    assert_eq!(reception.network_received_bytes, receipt.declared_bytes);
    assert_eq!(reception.storage_written_bytes, receipt.declared_bytes);
    assert_eq!(reception.compressed_consumed_bytes, receipt.declared_bytes);
    assert_eq!(reception.reclaimed_bytes, reception.reserved_bytes);
    assert_eq!(reception.buffered_bytes, 0);
    assert!(reception.buffer_high_water_bytes > 0);
    assert!(reception.reception_complete_seconds.is_some());
    assert!(reception.extraction_complete_seconds.is_some());
    switchyard_artifacts::verify(&destination, &fixture.manifest).unwrap();
    assert_eq!(fs::read_dir(spool).unwrap().count(), 0);
}
