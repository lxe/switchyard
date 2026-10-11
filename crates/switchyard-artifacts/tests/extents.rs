#![cfg(unix)]
use axum::{
    Router,
    body::{Body, Bytes},
    extract::{Path as HttpPath, State},
    http::{HeaderMap, StatusCode},
    response::Response,
    routing::get,
};
use futures_util::stream;
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
use switchyard_artifacts::{extents::*, *};
use tokio::sync::watch;

struct Fixture {
    root: PathBuf,
    manifest: ExtentManifest,
}
impl Fixture {
    fn new(bytes: usize, extent: u64) -> Self {
        let root = std::env::temp_dir().join(format!(
            "extent-test-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let source = root.join("source");
        fs::create_dir_all(source.join("nested")).unwrap();
        let mut state = 17_u64;
        let data = (0..bytes)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect::<Vec<_>>();
        fs::write(source.join("nested/large"), data).unwrap();
        fs::write(source.join("empty"), []).unwrap();
        let manifest = pack_extents(
            &source,
            &root.join("objects"),
            Metadata {
                kind: "test".into(),
                compatibility: BTreeMap::new(),
                attributes: BTreeMap::new(),
            },
            extent,
        )
        .unwrap();
        Self { root, manifest }
    }
    fn entry(&self) -> usize {
        self.manifest
            .entries
            .iter()
            .position(|e| e.path() == "nested/large")
            .unwrap()
    }
    fn options(&self) -> ExtentOptions {
        ExtentOptions {
            mode: HydrationMode::TrustedStreaming,
            request_timeout: Duration::from_secs(5),
            ..ExtentOptions::default()
        }
    }
    async fn server(
        &self,
        held: bool,
        fault: Fault,
    ) -> (
        ExtentTicket,
        HttpState,
        watch::Sender<bool>,
        tokio::task::JoinHandle<()>,
    ) {
        let (release, gate) = watch::channel(!held);
        let state = HttpState {
            objects: Arc::new(
                self.manifest
                    .extents
                    .iter()
                    .map(|e| {
                        (
                            e.object.clone(),
                            fs::read(self.root.join("objects").join(&e.object)).unwrap(),
                        )
                    })
                    .collect(),
            ),
            held: self.manifest.extents[0].object.clone(),
            gate,
            fault,
            requested: Arc::new(AtomicUsize::new(0)),
            finished: Arc::new(AtomicUsize::new(0)),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/{name}", get(response))
            .with_state(state.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let ticket = ExtentTicket {
            manifest: self.manifest.clone(),
            urls: self
                .manifest
                .extents
                .iter()
                .map(|e| (e.object.clone(), format!("{base}/{}", e.object)))
                .collect(),
        };
        (ticket, state, release, server)
    }
    fn staging_count(&self) -> usize {
        fs::read_dir(&self.root)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".staging")
            })
            .count()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
#[derive(Clone, Copy)]
enum Fault {
    None,
    Truncate,
    Oversize,
    Corrupt,
    BadRange,
}
#[derive(Clone)]
struct HttpState {
    objects: Arc<BTreeMap<String, Vec<u8>>>,
    held: String,
    gate: watch::Receiver<bool>,
    fault: Fault,
    requested: Arc<AtomicUsize>,
    finished: Arc<AtomicUsize>,
}
async fn response(
    State(state): State<HttpState>,
    HttpPath(name): HttpPath<String>,
    headers: HeaderMap,
) -> Response {
    state.requested.fetch_add(1, Ordering::Relaxed);
    let original = state.objects[&name].clone();
    let (mut bytes, ranged, start, end) = if let Some(range) = headers.get("range") {
        let (start, end) = range
            .to_str()
            .unwrap()
            .strip_prefix("bytes=")
            .unwrap()
            .split_once('-')
            .unwrap();
        let start: usize = start.parse().unwrap();
        let end: usize = end.parse().unwrap();
        (original[start..=end].to_vec(), true, start, end)
    } else {
        (original.clone(), false, 0, original.len() - 1)
    };
    match state.fault {
        Fault::Truncate => {
            bytes.pop();
        }
        Fault::Oversize => bytes.push(1),
        Fault::Corrupt => bytes[0] ^= 1,
        _ => {}
    }
    let body = stream::unfold(
        (state.clone(), Some(bytes), name),
        |(mut state, bytes, name)| async move {
            let bytes = bytes?;
            if name == state.held {
                while !*state.gate.borrow() {
                    if state.gate.changed().await.is_err() {
                        break;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
            state.finished.fetch_add(1, Ordering::Relaxed);
            Some((
                Ok::<_, std::io::Error>(Bytes::from(bytes)),
                (state, None, name),
            ))
        },
    );
    let mut result = Response::new(Body::from_stream(body));
    if ranged {
        *result.status_mut() = StatusCode::PARTIAL_CONTENT;
        let range = if matches!(state.fault, Fault::BadRange) {
            "bytes 0-0/1".into()
        } else {
            format!("bytes {start}-{end}/{}", original.len())
        };
        result
            .headers_mut()
            .insert("content-range", range.parse().unwrap());
    }
    result
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn twenty_useful_streams_and_completion_before_publication() {
    let fixture = Fixture::new(32 * 4096, 4096);
    let (ticket, state, release, server) = fixture.server(true, Fault::None).await;
    let destination = fixture.root.join("ready");
    let hydration =
        Arc::new(ExtentHydration::prepare(ticket, &destination, fixture.options()).unwrap());
    let working = hydration.clone();
    let task = tokio::spawn(async move { working.run().await });
    let entry = fixture.entry();
    let ready = tokio::time::timeout(
        Duration::from_secs(3),
        hydration.wait_range(entry, 19 * 4096, 4096),
    )
    .await
    .unwrap()
    .unwrap();
    let all_later = tokio::time::timeout(
        Duration::from_secs(3),
        hydration.wait_range(entry, 4096, 19 * 4096),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(!hydration.is_complete(entry, 0, 4096).unwrap());
    assert!(!destination.exists());
    assert!(state.requested.load(Ordering::Relaxed) >= 20);
    assert!(state.finished.load(Ordering::Relaxed) >= 19);
    let mut bytes = vec![0; 4096];
    ready.read_into(0, &mut bytes).unwrap();
    assert_eq!(
        bytes,
        fs::read(fixture.root.join("source/nested/large")).unwrap()[19 * 4096..20 * 4096]
    );
    assert!(ready.read_into(4090, &mut [0; 7]).is_err());
    release.send(true).unwrap();
    let report = task.await.unwrap().unwrap();
    assert_eq!(report.copy_buffer_bound, 20 * 64 * 1024);
    assert_eq!(report.downloaded_bytes, 32 * 4096);
    assert_eq!(
        fs::read(destination.join("nested/large")).unwrap(),
        fs::read(fixture.root.join("source/nested/large")).unwrap()
    );
    assert_eq!(fs::metadata(destination.join("empty")).unwrap().len(), 0);
    ready.read_into(0, &mut bytes).unwrap(); // Lease survives rename.
    drop(ready);
    drop(all_later);
    drop(hydration);
    assert_eq!(fixture.staging_count(), 0);
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_run_wakes_waiters_and_cleans_after_last_owner() {
    let fixture = Fixture::new(32 * 4096, 4096);
    let (ticket, state, _release, server) = fixture.server(true, Fault::None).await;
    let destination = fixture.root.join("ready");
    let hydration =
        Arc::new(ExtentHydration::prepare(ticket, &destination, fixture.options()).unwrap());
    let working = hydration.clone();
    let task = tokio::spawn(async move { working.run().await });
    while state.requested.load(Ordering::Relaxed) < 20 {
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    task.abort();
    let _ = task.await;
    assert!(matches!(
        hydration.wait_range(fixture.entry(), 0, 4096).await,
        Err(Error::Cancelled)
    ));
    assert!(!destination.exists());
    drop(hydration);
    tokio::time::timeout(Duration::from_secs(3), async {
        while fixture.staging_count() != 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exact_bodies_and_optional_hashes_without_retries() {
    for fault in [Fault::Truncate, Fault::Oversize, Fault::Corrupt] {
        let fixture = Fixture::new(4096, 4096);
        let (ticket, state, _release, server) = fixture.server(false, fault).await;
        let mut options = fixture.options();
        options.mode = HydrationMode::Verified;
        let hydration =
            ExtentHydration::prepare(ticket, &fixture.root.join("ready"), options).unwrap();
        assert!(hydration.run().await.is_err());
        assert_eq!(state.requested.load(Ordering::Relaxed), 1);
        assert!(!fixture.root.join("ready").exists());
        drop(hydration);
        assert_eq!(fixture.staging_count(), 0);
        server.abort();
    }
    let fixture = Fixture::new(4096, 4096);
    let (ticket, _, _release, server) = fixture.server(false, Fault::Corrupt).await;
    let hydration =
        ExtentHydration::prepare(ticket, &fixture.root.join("ready"), fixture.options()).unwrap();
    hydration.run().await.unwrap(); // Explicit trusted contract checks boundaries, not contents.
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_object_ranges_require_exact_content_range() {
    for fault in [Fault::None, Fault::BadRange] {
        let mut fixture = Fixture::new(8192, 4096);
        let full = fs::read(fixture.root.join("source/nested/large")).unwrap();
        fs::write(fixture.root.join("objects/full.raw"), &full).unwrap();
        // Server loads from the ticket's object names: all extents reference the full file.
        for e in &mut fixture.manifest.extents {
            e.object = "full.raw".into();
            e.object_offset = e.offset;
            e.object_size = 8192;
        }
        fixture.manifest.id = fixture.manifest.identity().unwrap();
        let (ticket, _, _release, server) = fixture.server(false, fault).await;
        let hydration =
            ExtentHydration::prepare(ticket, &fixture.root.join("ready"), fixture.options())
                .unwrap();
        let result = hydration.run().await;
        assert_eq!(result.is_ok(), matches!(fault, Fault::None));
        if result.is_ok() {
            assert_eq!(
                fs::read(fixture.root.join("ready/nested/large")).unwrap(),
                full
            );
        }
        server.abort();
    }
}

#[test]
fn content_identity_is_layout_independent_and_invalid_coverage_is_rejected() {
    let fixture = Fixture::new(8192, 4096);
    let different = pack_extents(
        &fixture.root.join("source"),
        &fixture.root.join("small"),
        fixture.manifest.metadata.clone(),
        1024,
    )
    .unwrap();
    assert_eq!(fixture.manifest.content_id, different.content_id);
    assert_ne!(fixture.manifest.id, different.id);
    for mutation in 0..5 {
        let mut broken = fixture.manifest.clone();
        match mutation {
            0 => broken.extents[0].offset = 1,
            1 => broken.extents[0].length = 0,
            2 => broken.extents[0].object_offset = u64::MAX,
            3 => broken.extents[0].object = "../escape".into(),
            _ => broken.extents.reverse(),
        }
        assert!(broken.validate_structure().is_err());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancellation_releases_shared_budget_after_writer_stops() {
    let fixture = Fixture::new(8192, 4096);
    let (ticket, state, _release, server) = fixture.server(true, Fault::None).await;
    let limit = Arc::new(tokio::sync::Semaphore::new(1));
    let mut options = fixture.options();
    options.download_limit = Some(limit.clone());
    let hydration =
        Arc::new(ExtentHydration::prepare(ticket, &fixture.root.join("ready"), options).unwrap());
    let working = hydration.clone();
    let task = tokio::spawn(async move { working.run().await });
    while state.requested.load(Ordering::Relaxed) == 0 {
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert_eq!(limit.available_permits(), 0);
    task.abort();
    let _ = task.await;
    tokio::time::timeout(Duration::from_secs(3), async {
        while limit.available_permits() != 1 {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    drop(hydration);
    assert_eq!(fixture.staging_count(), 0);
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explicit_cancel_wakes_a_transfer_waiting_for_another_calls_budget() {
    let fixture = Fixture::new(4096, 4096);
    let (ticket, state, _release, server) = fixture.server(false, Fault::None).await;
    let mut options = fixture.options();
    options.download_limit = Some(Arc::new(tokio::sync::Semaphore::new(0)));
    let hydration =
        Arc::new(ExtentHydration::prepare(ticket, &fixture.root.join("ready"), options).unwrap());
    let working = hydration.clone();
    let task = tokio::spawn(async move { working.run().await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    hydration.cancel();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap(),
        Err(Error::Cancelled)
    ));
    assert_eq!(state.requested.load(Ordering::Relaxed), 0);
    drop(hydration);
    assert_eq!(fixture.staging_count(), 0);
    server.abort();
}

#[test]
fn caller_prefix_boundary_keeps_canonical_content_identity() {
    let fixture = Fixture::new(5 * 4096, 4096);
    let prefix = pack_extents_with_prefixes(
        &fixture.root.join("source"),
        &fixture.root.join("prefixed"),
        fixture.manifest.metadata.clone(),
        4096,
        &BTreeMap::from([("nested/large".into(), 37)]),
    )
    .unwrap();
    assert_eq!(prefix.content_id, fixture.manifest.content_id);
    assert_ne!(prefix.id, fixture.manifest.id);
    let first = prefix
        .extents
        .iter()
        .find(|e| e.entry == fixture.entry())
        .unwrap();
    assert_eq!(first.offset, 0);
    assert_eq!(first.length, 37);
    assert_eq!(
        fs::read(fixture.root.join("prefixed").join(&first.object)).unwrap(),
        fs::read(fixture.root.join("source/nested/large")).unwrap()[..37]
    );
}
#[tokio::test]
async fn explicit_trusted_reuse_accepts_repacked_content_without_network() {
    let fixture = Fixture::new(5 * 4096, 4096);
    let (ticket, state, _gate, server) = fixture.server(false, Fault::None).await;
    let destination = fixture.root.join("ready");
    let mut options = fixture.options();
    options.mode = HydrationMode::TrustedStreaming;
    let hydration =
        ExtentHydration::prepare(ticket.clone(), &destination, options.clone()).unwrap();
    hydration.run().await.unwrap();
    drop(hydration);
    assert!(matches!(
        ExtentHydration::prepare(ticket.clone(), &destination, options.clone()),
        Err(Error::DestinationExists)
    ));
    let requested = state.requested.load(Ordering::Relaxed);
    options.reuse_existing = true;
    let reused = ExtentHydration::prepare(ticket, &destination, options).unwrap();
    let report = reused.run().await.unwrap();
    assert_eq!(report.downloaded_bytes, 0);
    assert_eq!(state.requested.load(Ordering::Relaxed), requested);
    let ready = reused.wait_range(fixture.entry(), 0, 37).await.unwrap();
    let mut bytes = [0; 37];
    ready.read_into(0, &mut bytes).unwrap();
    assert_eq!(
        bytes,
        fs::read(fixture.root.join("source/nested/large")).unwrap()[..37]
    );
    drop(ready);
    drop(reused);
    assert!(destination.exists());
    server.abort();
}
