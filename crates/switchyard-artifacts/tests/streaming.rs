#![cfg(unix)]
//! Prove overlap and ownership at HTTP/filesystem boundaries, without cloud services.
use axum::{
    Router,
    body::{Body, Bytes},
    extract::{Path as HttpPath, State},
    http::HeaderMap,
    response::Response,
    routing::get,
};
use futures_util::stream;
use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use switchyard_artifacts::*;
use tokio::sync::{Semaphore, watch};

struct Fixture {
    root: PathBuf,
    manifest: Manifest,
}
impl Fixture {
    fn new(files: usize, bytes: usize) -> Self {
        let mut nonce = [0_u8; 12];
        getrandom::fill(&mut nonce).unwrap();
        let root = std::env::temp_dir().join(format!(
            "switchyard-streaming-{}",
            nonce.iter().map(|b| format!("{b:02x}")).collect::<String>()
        ));
        let source = root.join("source");
        fs::create_dir_all(&source).unwrap();
        for index in 0..files {
            let mut data = vec![0; bytes];
            getrandom::fill(&mut data).unwrap();
            fs::write(source.join(format!("file{index:03}")), data).unwrap();
        }
        let manifest = pack(
            &source,
            &root.join("packed"),
            Metadata {
                kind: "streaming-test".into(),
                compatibility: BTreeMap::new(),
                attributes: BTreeMap::new(),
            },
            PackOptions {
                target_shard_bytes: 1,
                zstd_level: 1,
            },
        )
        .unwrap();
        Self { root, manifest }
    }
    fn options(&self) -> HydrationOptions {
        HydrationOptions {
            mode: HydrationMode::TrustedStreaming,
            cache_dir: self.root.join("unused-cache"),
            download_concurrency: 20,
            extraction_concurrency: 20,
            retries: 0,
            request_timeout_s: 10,
            stall_timeout_s: 3,
            ..HydrationOptions::default()
        }
    }
    async fn serve(
        &self,
        held: bool,
        truncate_once: bool,
    ) -> (
        ArtifactTicket,
        HttpState,
        watch::Sender<bool>,
        tokio::task::JoinHandle<()>,
    ) {
        let (gate, released) = watch::channel(!held);
        let state = HttpState {
            bodies: Arc::new(
                self.manifest
                    .chunks
                    .iter()
                    .map(|c| {
                        (
                            c.name.clone(),
                            fs::read(self.root.join("packed").join(&c.name)).unwrap(),
                        )
                    })
                    .collect(),
            ),
            gate: released,
            truncate_once,
            requests: Arc::new(Mutex::new(vec![])),
            active: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            finished: Arc::new(AtomicUsize::new(0)),
        };
        let app = Router::new()
            .route("/{name}", get(response))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let ticket = ArtifactTicket {
            manifest: self.manifest.clone(),
            urls: self
                .manifest
                .chunks
                .iter()
                .map(|c| (c.name.clone(), format!("{url}/{}", c.name)))
                .collect(),
        };
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (ticket, state, gate, server)
    }
    fn staging(&self) -> Option<PathBuf> {
        fs::read_dir(&self.root).unwrap().find_map(|e| {
            let e = e.unwrap();
            e.file_name()
                .to_string_lossy()
                .ends_with(".staging")
                .then(|| e.path())
        })
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
#[derive(Clone)]
struct HttpState {
    bodies: Arc<BTreeMap<String, Vec<u8>>>,
    gate: watch::Receiver<bool>,
    truncate_once: bool,
    requests: Arc<Mutex<Vec<Option<String>>>>,
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    finished: Arc<AtomicUsize>,
}
struct Active(HttpState);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::Relaxed);
        self.0.finished.fetch_add(1, Ordering::Relaxed);
    }
}
async fn response(
    State(state): State<HttpState>,
    HttpPath(name): HttpPath<String>,
    headers: HeaderMap,
) -> Response {
    let attempt = {
        let mut requests = state.requests.lock().unwrap();
        requests.push(headers.get("range").map(|v| v.to_str().unwrap().into()));
        requests.len()
    };
    let active = state.active.fetch_add(1, Ordering::Relaxed) + 1;
    state.peak.fetch_max(active, Ordering::Relaxed);
    let body = &state.bodies[&name];
    let split = body.len() * 9 / 10;
    let mut pieces = VecDeque::from([
        Bytes::copy_from_slice(&body[..split]),
        Bytes::copy_from_slice(&body[split..]),
    ]);
    if state.truncate_once && attempt == 1 {
        pieces.pop_back();
    }
    let stream = stream::unfold(
        (pieces, state.gate.clone(), false, Active(state)),
        |(mut pieces, mut gate, sent, active)| async move {
            let bytes = pieces.pop_front()?;
            if sent {
                while !*gate.borrow() {
                    gate.changed().await.unwrap();
                }
            }
            Some((Ok::<_, std::io::Error>(bytes), (pieces, gate, true, active)))
        },
    );
    Response::new(Body::from_stream(stream))
}
async fn until(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn twenty_independent_streams_extract_before_any_response_finishes() {
    let fixture = Fixture::new(24, 256 * 1024);
    let (ticket, state, gate, server) = fixture.serve(true, false).await;
    let destination = fixture.root.join("hydrated");
    let target = destination.clone();
    let options = fixture.options();
    let progress = Arc::new(Mutex::new(vec![]));
    let recorded = progress.clone();
    let task = tokio::spawn(async move {
        hydrate(ticket, &target, options, move |p| {
            recorded.lock().unwrap().push(p)
        })
        .await
    });
    until(|| {
        state.active.load(Ordering::Relaxed) == 20
            && fixture.staging().is_some_and(|p| {
                fs::read_dir(p)
                    .unwrap()
                    .any(|e| e.unwrap().metadata().is_ok_and(|m| m.len() >= 64 * 1024))
            })
    })
    .await;
    assert_eq!(state.peak.load(Ordering::Relaxed), 20);
    assert_eq!(state.finished.load(Ordering::Relaxed), 0);
    assert!(!destination.exists());
    assert!(!fixture.options().cache_dir.exists());
    gate.send(true).unwrap();
    let report = task.await.unwrap().unwrap();
    assert_eq!(report.verified_bytes, 0);
    assert_eq!(report.network_verified_bytes, 0);
    assert_eq!(report.cache_hits, 0);
    assert_eq!(
        report.downloaded_bytes,
        fixture.manifest.chunks.iter().map(|c| c.size).sum::<u64>()
    );
    assert!(
        progress
            .lock()
            .unwrap()
            .iter()
            .all(|p| p.phase != Phase::Verifying && p.verified_bytes == 0)
    );
    verify(&destination, &fixture.manifest).unwrap();
    assert!(fixture.staging().is_none());
    server.abort();
}

#[tokio::test]
async fn trusted_mode_skips_hashes_and_reuses_receipt_without_tree_rescan() {
    let fixture = Fixture::new(1, 32 * 1024);
    let (mut ticket, state, _, server) = fixture.serve(false, false).await;
    let destination = fixture.root.join("hydrated");
    // Deliberately stale digest and identity: structural safety remains valid,
    // but this caller-selected mode must not perform content/manifest rehashes.
    if let Entry::File { sha256, .. } = &mut ticket.manifest.entries[0] {
        *sha256 = "0".repeat(64);
    }
    let original_name = ticket.manifest.chunks[0].name.clone();
    ticket.manifest.chunks[0].sha256 = "1".repeat(64);
    ticket.manifest.chunks[0].name = format!("{}.tar.zst", "1".repeat(64));
    ticket.urls = BTreeMap::from([(
        ticket.manifest.chunks[0].name.clone(),
        ticket.urls[&original_name].clone(),
    )]);
    assert!(ticket.manifest.validate().is_err());
    hydrate(ticket.clone(), &destination, fixture.options(), |_| {})
        .await
        .unwrap();
    fs::write(destination.join("file000"), "changed after installation").unwrap();
    fs::write(
        destination.join("extra"),
        "receipt reuse does not walk this tree",
    )
    .unwrap();
    let reused = hydrate(ticket, &destination, fixture.options(), |_| {})
        .await
        .unwrap();
    assert!(reused.already_present);
    assert_eq!(reused.verified_bytes, 0);
    assert_eq!(state.requests.lock().unwrap().len(), 1);
    assert!(verify(&destination, &fixture.manifest).is_err());
    server.abort();
}

#[tokio::test]
async fn truncated_stream_restarts_only_its_shard_after_removing_partial_files() {
    let fixture = Fixture::new(1, 256 * 1024);
    let (ticket, state, _, server) = fixture.serve(false, true).await;
    let mut options = fixture.options();
    options.retries = 1;
    let destination = fixture.root.join("hydrated");
    let report = hydrate(ticket, &destination, options, |_| {})
        .await
        .unwrap();
    assert_eq!(*state.requests.lock().unwrap(), vec![None, None]);
    assert!(report.downloaded_bytes > fixture.manifest.chunks[0].size);
    assert_eq!(report.verified_bytes, 0);
    verify(&destination, &fixture.manifest).unwrap();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_wakes_stalled_decoders_and_cleans_after_the_last_worker() {
    let fixture = Fixture::new(4, 256 * 1024);
    let (ticket, _, _gate, server) = fixture.serve(true, false).await;
    let destination = fixture.root.join("hydrated");
    let target = destination.clone();
    let limit = Arc::new(Semaphore::new(4));
    let mut options = fixture.options();
    options.download_limit = Some(limit.clone());
    options.stall_timeout_s = 60; // Cleanup must be driven by cancellation, not this timeout.
    options.request_timeout_s = 60;
    let task = tokio::spawn(async move { hydrate(ticket, &target, options, |_| {}).await });
    until(|| {
        fixture.staging().is_some_and(|p| {
            fs::read_dir(p)
                .unwrap()
                .any(|e| e.unwrap().metadata().is_ok_and(|m| m.len() > 0))
        })
    })
    .await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    until(|| fixture.staging().is_none() && limit.available_permits() == 4).await;
    assert!(!destination.exists());
    assert!(!fixture.options().cache_dir.exists());
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shared_transfer_budget_bounds_verified_and_streaming_hydration_together() {
    let fixture = Fixture::new(8, 32 * 1024);
    let (ticket, state, gate, server) = fixture.serve(true, false).await;
    let limit = Arc::new(Semaphore::new(4));
    let mut trusted = fixture.options();
    trusted.download_limit = Some(limit.clone());
    let mut verified = trusted.clone();
    verified.mode = HydrationMode::Verified;
    verified.cache_dir = fixture.root.join("strict-cache");
    let first = fixture.root.join("first");
    let second = fixture.root.join("second");
    let first_path = first.clone();
    let second_path = second.clone();
    let other_ticket = ticket.clone();
    let a = tokio::spawn(async move { hydrate(ticket, &first_path, trusted, |_| {}).await });
    let b =
        tokio::spawn(async move { hydrate(other_ticket, &second_path, verified, |_| {}).await });
    until(|| state.active.load(Ordering::Relaxed) == 4).await;
    assert_eq!(limit.available_permits(), 0);
    gate.send(true).unwrap();
    a.await.unwrap().unwrap();
    b.await.unwrap().unwrap();
    assert_eq!(state.peak.load(Ordering::Relaxed), 4);
    assert_eq!(limit.available_permits(), 4);
    verify(&first, &fixture.manifest).unwrap();
    verify(&second, &fixture.manifest).unwrap();
    server.abort();
}

#[tokio::test]
async fn declared_bounds_and_tar_metadata_still_fail_without_publication() {
    let fixture = Fixture::new(1, 32 * 1024);
    let (ticket, _, _, server) = fixture.serve(false, false).await;
    for case in 0..4 {
        let mut invalid = ticket.clone();
        match case {
            0 => invalid.manifest.chunks[0].size -= 1,
            1 => invalid.manifest.chunks[0].size += 1,
            2 => invalid.manifest.chunks[0].unpacked_size -= 512,
            3 => {
                if let Entry::File { path, .. } = &mut invalid.manifest.entries[0] {
                    *path = "different-file".into();
                }
            }
            _ => unreachable!(),
        }
        let destination = fixture.root.join(format!("invalid{case}"));
        assert!(
            hydrate(invalid, &destination, fixture.options(), |_| {})
                .await
                .is_err()
        );
        until(|| fixture.staging().is_none()).await;
        assert!(!destination.exists());
    }
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_blocking_worker_retains_staging_and_shared_permit_until_it_stops() {
    let fixture = Fixture::new(1, 2 * 1024 * 1024);
    let (ticket, _, _gate, server) = fixture.serve(true, false).await;
    let destination = fixture.root.join("hydrated");
    let target = destination.clone();
    let limit = Arc::new(Semaphore::new(1));
    let mut options = fixture.options();
    options.download_limit = Some(limit.clone());
    let entered = Arc::new(tokio::sync::Notify::new());
    let worker_entered = entered.clone();
    let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let worker_gate = gate.clone();
    let once = std::sync::atomic::AtomicBool::new(false);
    let task = tokio::spawn(async move {
        hydrate(ticket, &target, options, move |progress| {
            if progress.unpacked_bytes > 0 && !once.swap(true, Ordering::Relaxed) {
                worker_entered.notify_one();
                let (lock, ready) = &*worker_gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = ready.wait(released).unwrap();
                }
            }
        })
        .await
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(limit.available_permits(), 0);
    assert!(fixture.staging().is_some());
    assert!(!destination.exists());
    let (lock, ready) = &*gate;
    *lock.lock().unwrap() = true;
    ready.notify_all();
    until(|| fixture.staging().is_none() && limit.available_permits() == 1).await;
    assert!(!destination.exists());
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reserved_receivers_reach_eof_while_decoder_budget_is_blocked() {
    let fixture = Fixture::new(24, 256 * 1024);
    let (ticket, state, gate, server) = fixture.serve(true, false).await;
    let total = fixture.manifest.chunks.iter().map(|c| c.size).sum::<u64>();
    let destination = fixture.root.join("reserved-output");
    let target = destination.clone();
    let decoders = Arc::new(Semaphore::new(0));
    let limit = Arc::new(Semaphore::new(20));
    let mut buffer = ReceiveBufferOptions::new(fixture.root.join("spool"));
    buffer.extraction_limit = Some(decoders.clone());
    let options = HydrationOptions {
        receive_buffer: Some(buffer),
        download_limit: Some(limit.clone()),
        extraction_concurrency: 1,
        ..fixture.options()
    };
    let observed = Arc::new(Mutex::new(None));
    let receipt = observed.clone();
    let task = tokio::spawn(async move {
        hydrate(ticket, &target, options, move |p| {
            *receipt.lock().unwrap() = p.reception;
        })
        .await
    });
    until(|| state.active.load(Ordering::Relaxed) == 20).await;
    assert_eq!(limit.available_permits(), 0);
    gate.send(true).unwrap();
    until(|| {
        observed
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|p| p.storage_written_bytes == total)
    })
    .await;
    until(|| state.finished.load(Ordering::Relaxed) == 24 && limit.available_permits() == 20).await;
    let paused = observed.lock().unwrap().clone().unwrap();
    assert_eq!(paused.network_received_bytes, total);
    assert_eq!(paused.compressed_consumed_bytes, 0);
    assert_eq!(paused.decoded_bytes, 0);
    assert_eq!(paused.buffered_bytes, total);
    assert_eq!(paused.buffer_high_water_bytes, total);
    assert!(state.peak.load(Ordering::Relaxed) >= 20);
    assert!(!destination.exists());
    assert!(!task.is_finished());
    decoders.add_permits(1);
    let report = task.await.unwrap().unwrap();
    let progress = report.reception.unwrap();
    assert_eq!(progress.compressed_consumed_bytes, total);
    assert_eq!(progress.buffered_bytes, 0);
    assert_eq!(progress.file_written_bytes, 24 * 256 * 1024);
    assert_eq!(progress.reclaimed_bytes, progress.reserved_bytes);
    assert!(
        progress.reception_complete_seconds.unwrap()
            <= progress.extraction_complete_seconds.unwrap()
    );
    verify(&destination, &fixture.manifest).unwrap();
    assert_eq!(fs::read_dir(fixture.root.join("spool")).unwrap().count(), 0);
    server.abort();
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reserved_prefix_decodes_before_http_eof_and_cancellation_reclaims_backing() {
    let fixture = Fixture::new(2, 2 * 1024 * 1024);
    let (ticket, state, _gate, server) = fixture.serve(true, false).await;
    let destination = fixture.root.join("reserved-output");
    let target = destination.clone();
    let options = HydrationOptions {
        receive_buffer: Some(ReceiveBufferOptions::new(fixture.root.join("spool"))),
        ..fixture.options()
    };
    let observed = Arc::new(Mutex::new(None));
    let receipt = observed.clone();
    let task = tokio::spawn(async move {
        hydrate(ticket, &target, options, move |p| {
            *receipt.lock().unwrap() = p.reception;
        })
        .await
    });
    until(|| {
        observed
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|p| p.file_written_bytes > 1024 * 1024 && p.reclaimed_bytes > 0)
    })
    .await;
    assert_eq!(state.finished.load(Ordering::Relaxed), 0);
    task.abort();
    let _ = task.await;
    until(|| {
        fixture.staging().is_none()
            && fs::read_dir(fixture.root.join("spool")).unwrap().count() == 0
    })
    .await;
    assert!(!destination.exists());
    server.abort();
}
#[tokio::test]
async fn reserved_capacity_rejection_happens_before_http() {
    let fixture = Fixture::new(1, 65536);
    let (ticket, state, _gate, server) = fixture.serve(false, false).await;
    let mut buffer = ReceiveBufferOptions::new(fixture.root.join("spool"));
    buffer.max_reserved_bytes = 4096;
    let error = hydrate(
        ticket,
        &fixture.root.join("output"),
        HydrationOptions {
            receive_buffer: Some(buffer),
            ..fixture.options()
        },
        |_| {},
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("reserved budget"));
    assert!(state.requests.lock().unwrap().is_empty());
    assert!(fixture.staging().is_none());
    server.abort();
}
