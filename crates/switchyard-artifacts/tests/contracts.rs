#![cfg(unix)]
use axum::{
    Router,
    body::Body,
    extract::{Path as HttpPath, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime},
};
use switchyard_artifacts::*;

struct Tree(PathBuf);
impl Tree {
    fn new() -> Self {
        let mut random = [0_u8; 12];
        getrandom::fill(&mut random).unwrap();
        let name = random
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let path = std::env::temp_dir().join(format!("switchyard-artifacts-test-{name}"));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn source(&self) -> PathBuf {
        let source = self.0.join("source");
        fs::create_dir(&source).unwrap();
        source
    }
    fn options(&self) -> HydrationOptions {
        HydrationOptions {
            cache_dir: self.0.join("cache"),
            retries: 0,
            ..HydrationOptions::default()
        }
    }
}
impl Drop for Tree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn metadata() -> Metadata {
    Metadata {
        kind: "test-runtime".into(),
        compatibility: [("architecture".into(), "x86_64".into())].into(),
        attributes: BTreeMap::new(),
    }
}
fn reidentify(manifest: &mut Manifest) {
    let bytes = serde_json::to_vec(&(
        &manifest.format,
        &manifest.metadata,
        &manifest.entries,
        &manifest.chunks,
    ))
    .unwrap();
    manifest.id = format!("{:x}", Sha256::digest(bytes));
}
fn mode(path: &std::path::Path, mode: u32) {
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}
fn fixture(tree: &Tree) -> Manifest {
    let source = tree.source();
    fs::create_dir(source.join("bin")).unwrap();
    fs::create_dir(source.join("empty")).unwrap();
    fs::write(source.join("bin/tool"), b"#!/bin/sh\necho artifact\n").unwrap();
    fs::write(source.join("data"), vec![42; 16_384]).unwrap();
    mode(&source.join("bin"), 0o750);
    mode(&source.join("bin/tool"), 0o755);
    mode(&source.join("data"), 0o640);
    symlink("bin/tool", source.join("tool")).unwrap();
    pack(
        &source,
        &tree.0.join("packed"),
        metadata(),
        PackOptions {
            target_shard_bytes: 4096,
            ..PackOptions::default()
        },
    )
    .unwrap()
}

#[derive(Clone)]
struct HttpFixture {
    bodies: Arc<BTreeMap<String, Vec<u8>>>,
    requests: RequestLog,
    fail: Arc<AtomicBool>,
    fail_name: Option<String>,
    truncate_once: bool,
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}
type RequestLog = Arc<Mutex<Vec<(String, Option<String>)>>>;
async fn response(
    State(state): State<HttpFixture>,
    HttpPath(name): HttpPath<String>,
    headers: HeaderMap,
) -> Response {
    let range = headers
        .get("range")
        .and_then(|h| h.to_str().ok())
        .map(str::to_owned);
    let number = {
        let mut requests = state.requests.lock().unwrap();
        requests.push((name.clone(), range.clone()));
        requests.len()
    };
    let active = state.active.fetch_add(1, Ordering::Relaxed) + 1;
    state.peak.fetch_max(active, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(15)).await;
    state.active.fetch_sub(1, Ordering::Relaxed);
    if state.fail.load(Ordering::Relaxed) && state.fail_name.as_ref() == Some(&name) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "private URL response must not appear in errors",
        )
            .into_response();
    }
    let body = &state.bodies[&name];
    if state.truncate_once && number == 1 {
        return (StatusCode::OK, body[..body.len() / 2].to_vec()).into_response();
    }
    if let Some(range) = range {
        let start = range
            .strip_prefix("bytes=")
            .unwrap()
            .trim_end_matches('-')
            .parse::<usize>()
            .unwrap();
        return Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header(
                "content-range",
                format!("bytes {start}-{}/{}", body.len() - 1, body.len()),
            )
            .body(Body::from(body[start..].to_vec()))
            .unwrap();
    }
    (StatusCode::OK, body.clone()).into_response()
}
async fn server(
    manifest: &Manifest,
    packed: &std::path::Path,
    fail_name: Option<String>,
    truncate_once: bool,
) -> (ArtifactTicket, HttpFixture, tokio::task::JoinHandle<()>) {
    let state = HttpFixture {
        bodies: Arc::new(
            manifest
                .chunks
                .iter()
                .map(|chunk| {
                    (
                        chunk.name.clone(),
                        fs::read(packed.join(&chunk.name)).unwrap(),
                    )
                })
                .collect(),
        ),
        requests: Arc::new(Mutex::new(vec![])),
        fail: Arc::new(AtomicBool::new(fail_name.is_some())),
        fail_name,
        truncate_once,
        active: Arc::new(AtomicUsize::new(0)),
        peak: Arc::new(AtomicUsize::new(0)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/{name}", get(response))
        .with_state(state.clone());
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (
        ArtifactTicket {
            manifest: manifest.clone(),
            urls: manifest
                .chunks
                .iter()
                .map(|c| {
                    (
                        c.name.clone(),
                        format!("{base}/{}?signature=secret-test-signature", c.name),
                    )
                })
                .collect(),
        },
        state,
        handle,
    )
}

#[test]
fn packing_is_deterministic_and_metadata_changes_identity_without_changing_chunks() {
    let tree = Tree::new();
    let first = fixture(&tree);
    let source = tree.0.join("source");
    fs::File::open(source.join("data"))
        .unwrap()
        .set_times(
            fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(17)),
        )
        .unwrap();
    let second = pack(
        &source,
        &tree.0.join("packed-again"),
        metadata(),
        PackOptions {
            target_shard_bytes: 4096,
            ..PackOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(&first).unwrap(),
        serde_json::to_value(&second).unwrap()
    );
    assert!(first.chunks.iter().any(|c| c.unpacked_size > 4096));
    for chunk in &first.chunks {
        assert_eq!(
            fs::read(tree.0.join("packed").join(&chunk.name)).unwrap(),
            fs::read(tree.0.join("packed-again").join(&chunk.name)).unwrap()
        );
    }
    let mut modified = metadata();
    modified
        .attributes
        .insert("source-manifest".into(), "upstream-digest".into());
    let third = pack(
        &source,
        &tree.0.join("packed-third"),
        modified,
        PackOptions {
            target_shard_bytes: 4096,
            ..PackOptions::default()
        },
    )
    .unwrap();
    assert_ne!(third.id, first.id);
    assert_eq!(
        serde_json::to_value(third.chunks).unwrap(),
        serde_json::to_value(&first.chunks).unwrap()
    );
    verify(&source, &first).unwrap();
    first
        .check_compatibility(&[("architecture".into(), "x86_64".into())].into())
        .unwrap();
    assert!(
        first
            .check_compatibility(&[("architecture".into(), "aarch64".into())].into())
            .is_err()
    );
}

#[tokio::test]
async fn hydration_preserves_modes_and_links_and_publishes_only_a_verified_tree() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    let (ticket, state, server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let destination = tree.0.join("hydrated");
    let inspected = destination.clone();
    let phases = Arc::new(Mutex::new(vec![]));
    let recorded = phases.clone();
    let report = hydrate(ticket.clone(), &destination, tree.options(), move |p| {
        if p.phase != Phase::Ready {
            assert!(!inspected.exists());
        }
        recorded.lock().unwrap().push(p.phase);
    })
    .await
    .unwrap();
    assert!(!report.already_present);
    assert_eq!(report.cache_hits, 0);
    let compressed_bytes = manifest.chunks.iter().map(|c| c.size).sum::<u64>();
    assert_eq!(report.downloaded_bytes, compressed_bytes);
    assert_eq!(report.verified_bytes, compressed_bytes);
    assert_eq!(report.network_verified_bytes, compressed_bytes);
    verify(&destination, &manifest).unwrap();
    assert_eq!(
        fs::metadata(destination.join("bin/tool"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o755
    );
    assert_eq!(
        fs::metadata(destination.join("bin"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o750
    );
    assert_eq!(
        fs::read_link(destination.join("tool")).unwrap(),
        PathBuf::from("bin/tool")
    );
    assert_eq!(phases.lock().unwrap().last(), Some(&Phase::Ready));
    let requests = state.requests.lock().unwrap().len();
    let again = hydrate(ticket, &destination, tree.options(), |_| {})
        .await
        .unwrap();
    assert!(again.already_present);
    assert_eq!(again.downloaded_bytes, 0);
    assert_eq!(again.verified_bytes, compressed_bytes);
    assert_eq!(again.network_verified_bytes, 0);
    assert_eq!(state.requests.lock().unwrap().len(), requests);
    server.abort();
}

#[tokio::test]
async fn partial_response_retries_resume_and_verified_chunks_are_reused() {
    let tree = Tree::new();
    let source = tree.source();
    fs::write(source.join("file"), vec![91; 8192]).unwrap();
    let manifest = pack(
        &source,
        &tree.0.join("packed"),
        metadata(),
        PackOptions::default(),
    )
    .unwrap();
    let (ticket, state, server) = server(&manifest, &tree.0.join("packed"), None, true).await;
    let mut options = tree.options();
    options.retries = 1;
    hydrate(
        ticket.clone(),
        &tree.0.join("first"),
        options.clone(),
        |_| {},
    )
    .await
    .unwrap();
    {
        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].1.is_none());
        assert!(requests[1].1.as_ref().unwrap().starts_with("bytes="));
    }
    let second = hydrate(ticket, &tree.0.join("second"), options, |_| {})
        .await
        .unwrap();
    assert_eq!(second.cache_hits, manifest.chunks.len());
    assert_eq!(second.downloaded_bytes, 0);
    assert_eq!(second.network_verified_bytes, 0);
    assert_eq!(second.verified_bytes, manifest.chunks[0].size);
    assert_eq!(state.requests.lock().unwrap().len(), 2);
    server.abort();
}

#[tokio::test]
async fn failed_transfer_retains_complete_chunks_for_a_later_retry() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    let failed = manifest.chunks.last().unwrap().name.clone();
    let (ticket, state, server) =
        server(&manifest, &tree.0.join("packed"), Some(failed), false).await;
    let mut options = tree.options();
    options.download_concurrency = 1;
    let destination = tree.0.join("hydrated");
    let error = hydrate(ticket.clone(), &destination, options.clone(), |_| {})
        .await
        .unwrap_err();
    assert!(!error.to_string().contains("secret-test-signature"));
    assert!(!destination.exists());
    assert!(options.cache_dir.join(&manifest.chunks[0].name).is_file());
    state.fail.store(false, Ordering::Relaxed);
    let report = hydrate(ticket, &destination, options, |_| {})
        .await
        .unwrap();
    assert_eq!(report.cache_hits, manifest.chunks.len() - 1);
    verify(&destination, &manifest).unwrap();
    server.abort();
}

#[tokio::test]
async fn corrupted_cache_is_reverified_and_replaced_while_mutated_destinations_are_refused() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    let (ticket, state, server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let options = tree.options();
    fs::create_dir_all(&options.cache_dir).unwrap();
    fs::write(
        options.cache_dir.join(&manifest.chunks[0].name),
        b"corruption",
    )
    .unwrap();
    let destination = tree.0.join("hydrated");
    hydrate(ticket.clone(), &destination, options.clone(), |_| {})
        .await
        .unwrap();
    verify(&destination, &manifest).unwrap();
    fs::write(destination.join("data"), b"mutated").unwrap();
    assert!(verify(&destination, &manifest).is_err());
    let requests = state.requests.lock().unwrap().len();
    assert!(matches!(
        hydrate(ticket, &destination, options, |_| {}).await,
        Err(Error::DestinationExists)
    ));
    assert_eq!(state.requests.lock().unwrap().len(), requests);
    assert_eq!(fs::read(destination.join("data")).unwrap(), b"mutated");
    server.abort();
}

#[test]
fn manifests_reject_traversal_unsafe_modes_and_link_ancestors_even_with_recomputed_identity() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    for unsafe_path in [
        "../outside",
        "/absolute",
        "a/../data",
        "a\\escape",
        "a//data",
        "C:/drive",
    ] {
        let mut bad = manifest.clone();
        let entry = bad
            .entries
            .iter_mut()
            .find(|e| matches!(e, Entry::File { .. }))
            .unwrap();
        if let Entry::File { path, .. } = entry {
            *path = unsafe_path.into();
        }
        reidentify(&mut bad);
        assert!(bad.validate().is_err(), "{unsafe_path}");
    }
    let mut bad = manifest.clone();
    if let Entry::File { mode, .. } = bad
        .entries
        .iter_mut()
        .find(|e| matches!(e, Entry::File { .. }))
        .unwrap()
    {
        *mode = 0o4755;
    }
    reidentify(&mut bad);
    assert!(bad.validate().is_err());
    let mut bad = manifest.clone();
    if let Entry::Symlink { target, .. } = bad.entries.last_mut().unwrap() {
        *target = "../outside".into();
    }
    reidentify(&mut bad);
    assert!(bad.validate().is_err());
    let mut bad = manifest;
    bad.entries[0] = Entry::Symlink {
        path: "bin".into(),
        mode: 0o777,
        target: "empty".into(),
    };
    reidentify(&mut bad);
    assert!(bad.validate().is_err());
}

#[test]
fn pack_rejects_external_links_privileged_files_and_output_inside_source() {
    let tree = Tree::new();
    let source = tree.source();
    fs::write(source.join("file"), "data").unwrap();
    symlink("../../outside", source.join("escape")).unwrap();
    assert!(
        pack(
            &source,
            &tree.0.join("packed"),
            metadata(),
            PackOptions::default()
        )
        .is_err()
    );
    fs::remove_file(source.join("escape")).unwrap();
    mode(&source.join("file"), 0o4755);
    assert!(
        pack(
            &source,
            &tree.0.join("packed"),
            metadata(),
            PackOptions::default()
        )
        .is_err()
    );
    mode(&source.join("file"), 0o644);
    assert!(
        pack(
            &source,
            &source.join("output"),
            metadata(),
            PackOptions::default()
        )
        .is_err()
    );
}

#[tokio::test]
async fn existing_symlink_ancestors_and_cached_links_are_never_followed() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    let (ticket, state, server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let outside = tree.0.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("sentinel"), "untouched").unwrap();
    symlink(&outside, tree.0.join("link")).unwrap();
    assert!(
        hydrate(
            ticket.clone(),
            &tree.0.join("link/destination"),
            tree.options(),
            |_| {}
        )
        .await
        .is_err()
    );
    assert!(state.requests.lock().unwrap().is_empty());
    let options = tree.options();
    fs::create_dir_all(&options.cache_dir).unwrap();
    symlink(
        outside.join("sentinel"),
        options.cache_dir.join(&manifest.chunks[0].name),
    )
    .unwrap();
    assert!(
        hydrate(ticket, &tree.0.join("hydrated"), options, |_| {})
            .await
            .is_err()
    );
    assert_eq!(
        fs::read_to_string(outside.join("sentinel")).unwrap(),
        "untouched"
    );
    assert!(!outside.join("destination").exists());
    server.abort();
}

#[tokio::test]
async fn transfer_limits_and_download_concurrency_are_enforced() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    let (ticket, state, server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let mut options = tree.options();
    options.max_unpacked_bytes = 1;
    assert!(
        hydrate(ticket.clone(), &tree.0.join("refused"), options, |_| {})
            .await
            .is_err()
    );
    assert!(state.requests.lock().unwrap().is_empty());
    let mut options = tree.options();
    options.download_concurrency = 2;
    hydrate(ticket, &tree.0.join("hydrated"), options, |_| {})
        .await
        .unwrap();
    assert_eq!(state.peak.load(Ordering::Relaxed), 2);
    server.abort();
}

#[tokio::test]
async fn metadata_identity_prevents_reusing_a_tree_under_another_compatibility_contract() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    let (ticket, _, server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let destination = tree.0.join("hydrated");
    hydrate(ticket.clone(), &destination, tree.options(), |_| {})
        .await
        .unwrap();
    let mut changed = ticket;
    changed
        .manifest
        .metadata
        .compatibility
        .insert("architecture".into(), "aarch64".into());
    reidentify(&mut changed.manifest);
    assert!(
        hydrate(changed, &destination, tree.options(), |_| {})
            .await
            .is_err()
    );
    server.abort();
}

#[test]
fn link_resolution_rejects_cycles_and_escapes_through_internal_links() {
    let tree = Tree::new();
    let source = tree.source();
    fs::write(source.join("file"), "preserved").unwrap();
    symlink(".", source.join("alias")).unwrap();
    symlink("alias/../file", source.join("escape")).unwrap();
    assert!(matches!(
        pack(
            &source,
            &tree.0.join("packed"),
            metadata(),
            PackOptions::default()
        ),
        Err(Error::Invalid(_))
    ));
    fs::remove_file(source.join("escape")).unwrap();
    fs::remove_file(source.join("alias")).unwrap();
    symlink("b", source.join("a")).unwrap();
    symlink("a", source.join("b")).unwrap();
    assert!(matches!(
        pack(
            &source,
            &tree.0.join("packed"),
            metadata(),
            PackOptions::default()
        ),
        Err(Error::Invalid(_))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn extraction_is_parallel_and_bounded_independently_of_downloads() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    assert!(manifest.chunks.len() >= 3);
    let (ticket, _, server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let mut options = tree.options();
    options.download_concurrency = 1;
    options.extraction_concurrency = 2;
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let callback_peak = peak.clone();
    hydrate(ticket, &tree.0.join("hydrated"), options, move |progress| {
        if progress.phase == Phase::Extracting && progress.unpacked_bytes > 0 {
            let count = active.fetch_add(1, Ordering::Relaxed) + 1;
            callback_peak.fetch_max(count, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(30));
            active.fetch_sub(1, Ordering::Relaxed);
        }
    })
    .await
    .unwrap();
    assert_eq!(peak.load(Ordering::Relaxed), 2);
    verify(&tree.0.join("hydrated"), &manifest).unwrap();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_hydration_during_extraction_never_publishes_the_destination() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    let (ticket, _, first_server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let destination = tree.0.join("hydrated");
    let worker_destination = destination.clone();
    let arrived = Arc::new(tokio::sync::Notify::new());
    let worker_arrived = arrived.clone();
    let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let worker_gate = gate.clone();
    let completed = Arc::new(tokio::sync::Notify::new());
    let worker_completed = completed.clone();
    let entered = AtomicBool::new(false);
    let options = tree.options();
    let hydration = tokio::spawn(async move {
        hydrate(ticket, &worker_destination, options, move |progress| {
            if progress.phase == Phase::Extracting
                && progress.unpacked_bytes > 0
                && !entered.swap(true, Ordering::Relaxed)
            {
                worker_arrived.notify_one();
                let (lock, ready) = &*worker_gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = ready.wait(released).unwrap();
                }
                worker_completed.notify_one();
            }
        })
        .await
    });
    tokio::time::timeout(Duration::from_secs(5), arrived.notified())
        .await
        .unwrap();
    hydration.abort();
    assert!(hydration.await.unwrap_err().is_cancelled());
    let (lock, ready) = &*gate;
    *lock.lock().unwrap() = true;
    ready.notify_all();
    tokio::time::timeout(Duration::from_secs(5), completed.notified())
        .await
        .unwrap();
    assert!(!destination.exists());
    assert!(fs::read_dir(&tree.0).unwrap().all(|item| {
        !item
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".staging")
    }));
    // Verified immutable chunks remain reusable after cancellation.
    let (ticket, _, retry_server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let retry = hydrate(ticket, &destination, tree.options(), |_| {})
        .await
        .unwrap();
    assert_eq!(retry.cache_hits, manifest.chunks.len());
    verify(&destination, &manifest).unwrap();
    first_server.abort();
    retry_server.abort();
}

#[tokio::test]
async fn full_tree_verification_rejects_extra_files_modes_and_changed_links() {
    let tree = Tree::new();
    let manifest = fixture(&tree);
    let (ticket, _, server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let destination = tree.0.join("hydrated");
    hydrate(ticket, &destination, tree.options(), |_| {})
        .await
        .unwrap();
    fs::write(destination.join("undeclared"), "extra").unwrap();
    assert!(matches!(
        verify(&destination, &manifest),
        Err(Error::Integrity(_))
    ));
    fs::remove_file(destination.join("undeclared")).unwrap();
    mode(&destination.join("bin/tool"), 0o644);
    assert!(matches!(
        verify(&destination, &manifest),
        Err(Error::Integrity(_))
    ));
    mode(&destination.join("bin/tool"), 0o755);
    fs::remove_file(destination.join("tool")).unwrap();
    symlink("data", destination.join("tool")).unwrap();
    assert!(matches!(
        verify(&destination, &manifest),
        Err(Error::Integrity(_))
    ));
    server.abort();
}

#[tokio::test]
async fn recomputed_chunk_hash_cannot_authorize_tar_traversal_or_excess_decompression() {
    let tree = Tree::new();
    let mut manifest = fixture(&tree);
    let packed = tree.0.join("packed");
    let original = &manifest.chunks[0];
    let mut decoded =
        zstd::stream::decode_all(fs::File::open(packed.join(&original.name)).unwrap()).unwrap();
    let mut header = tar::Header::from_byte_slice(&decoded[..512]).clone();
    header.as_mut_bytes()[..100].fill(0);
    header.as_mut_bytes()[..10].copy_from_slice(b"../outside");
    header.set_cksum();
    decoded[..512].copy_from_slice(header.as_bytes());
    let compressed = zstd::stream::encode_all(decoded.as_slice(), 6).unwrap();
    let chunk = &mut manifest.chunks[0];
    chunk.sha256 = format!("{:x}", Sha256::digest(&compressed));
    chunk.name = format!("{}.tar.zst", chunk.sha256);
    chunk.size = compressed.len() as u64;
    fs::write(packed.join(&chunk.name), compressed).unwrap();
    reidentify(&mut manifest);
    manifest.validate().unwrap();
    let (ticket, _, server) = server(&manifest, &packed, None, false).await;
    let destination = tree.0.join("hydrated");
    assert!(
        hydrate(ticket, &destination, tree.options(), |_| {})
            .await
            .is_err()
    );
    assert!(!destination.exists());
    assert!(!tree.0.join("outside").exists());
    server.abort();

    // A different, hash-valid archive must also obey the declared decoded length.
    let source = tree.0.join("source");
    let mut manifest = pack(&source, &packed, metadata(), PackOptions::default()).unwrap();
    manifest.chunks[0].unpacked_size -= 1;
    reidentify(&mut manifest);
    manifest.validate().unwrap();
    let (ticket, _, second_server) = self::server(&manifest, &packed, None, false).await;
    assert!(
        hydrate(ticket, &destination, tree.options(), |_| {})
            .await
            .is_err()
    );
    assert!(!destination.exists());
    assert!(fs::read_dir(&tree.0).unwrap().all(|item| {
        !item
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".staging")
    }));
    second_server.abort();
}

#[tokio::test]
async fn internal_python_style_link_chains_and_parent_components_are_preserved() {
    let tree = Tree::new();
    let source = tree.source();
    fs::create_dir(source.join("bin")).unwrap();
    fs::create_dir(source.join("lib")).unwrap();
    fs::write(source.join("bin/python3.12"), "executable").unwrap();
    symlink("python3.12", source.join("bin/python3")).unwrap();
    symlink("python3", source.join("bin/python")).unwrap();
    symlink("../bin/python", source.join("lib/interpreter")).unwrap();
    symlink("lib", source.join("lib64")).unwrap();
    let manifest = pack(
        &source,
        &tree.0.join("packed"),
        metadata(),
        PackOptions::default(),
    )
    .unwrap();
    let (ticket, _, server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let destination = tree.0.join("hydrated");
    hydrate(ticket, &destination, tree.options(), |_| {})
        .await
        .unwrap();
    verify(&destination, &manifest).unwrap();
    assert_eq!(
        fs::read_to_string(destination.join("lib/interpreter")).unwrap(),
        "executable"
    );
    assert_eq!(
        fs::read_to_string(destination.join("lib64/interpreter")).unwrap(),
        "executable"
    );
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_removes_staging_after_read_only_directory_modes_are_applied() {
    let tree = Tree::new();
    let source = tree.source();
    fs::create_dir(source.join("readonly")).unwrap();
    fs::write(source.join("readonly/file"), "data").unwrap();
    mode(&source.join("readonly"), 0o500);
    let manifest = pack(
        &source,
        &tree.0.join("packed"),
        metadata(),
        PackOptions::default(),
    )
    .unwrap();
    let (ticket, _, server) = server(&manifest, &tree.0.join("packed"), None, false).await;
    let destination = tree.0.join("hydrated");
    let target = destination.clone();
    let arrived = Arc::new(tokio::sync::Notify::new());
    let notify = arrived.clone();
    let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let worker_gate = gate.clone();
    let options = tree.options();
    let hydration = tokio::spawn(async move {
        hydrate(ticket, &target, options, move |progress| {
            if progress.phase == Phase::Verifying {
                notify.notify_one();
                let (lock, ready) = &*worker_gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = ready.wait(released).unwrap();
                }
            }
        })
        .await
    });
    tokio::time::timeout(Duration::from_secs(5), arrived.notified())
        .await
        .unwrap();
    hydration.abort();
    assert!(hydration.await.unwrap_err().is_cancelled());
    let (lock, ready) = &*gate;
    *lock.lock().unwrap() = true;
    ready.notify_all();
    assert!(!destination.exists());
    assert!(fs::read_dir(&tree.0).unwrap().all(|item| {
        !item
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".staging")
    }));
    mode(&source.join("readonly"), 0o700);
    server.abort();
}

#[test]
fn manifest_resolution_exposes_final_file_permissions_and_directory_type() {
    let tree = Tree::new();
    let source = tree.source();
    fs::create_dir(source.join("bin")).unwrap();
    fs::write(source.join("bin/executable"), "executable").unwrap();
    fs::write(source.join("bin/data"), "nonexecutable").unwrap();
    mode(&source.join("bin/executable"), 0o755);
    mode(&source.join("bin/data"), 0o644);
    symlink("bin/executable", source.join("interpreter")).unwrap();
    symlink("bin/data", source.join("nonexecutable")).unwrap();
    symlink("nonexecutable", source.join("chain")).unwrap();
    symlink("bin", source.join("directory")).unwrap();
    symlink(".", source.join("root")).unwrap();
    let manifest = pack(
        &source,
        &tree.0.join("packed"),
        metadata(),
        PackOptions::default(),
    )
    .unwrap();
    let executable = manifest.resolve_entry("interpreter").unwrap().unwrap();
    assert!(matches!(executable, Entry::File { .. }));
    assert_eq!(executable.path(), "bin/executable");
    assert_eq!(executable.mode(), 0o755);
    let data = manifest.resolve_entry("chain").unwrap().unwrap();
    assert!(matches!(data, Entry::File { .. }));
    assert_eq!(data.path(), "bin/data");
    assert_eq!(data.mode() & 0o111, 0);
    let directory = manifest.resolve_entry("directory").unwrap().unwrap();
    assert!(matches!(directory, Entry::Directory { .. }));
    assert_eq!(directory.path(), "bin");
    assert!(manifest.resolve_entry("root").unwrap().is_none());
    assert_eq!(
        manifest.resolve_entry("bin/data").unwrap().unwrap().path(),
        "bin/data"
    );
    for invalid in [
        "missing",
        "../bin/data",
        "/bin/data",
        "bin/./data",
        "directory/data",
    ] {
        assert!(
            matches!(manifest.resolve_entry(invalid), Err(Error::Invalid(_))),
            "{invalid}"
        );
    }
    // Resolution validates the full identity rather than trusting caller-mutable entries.
    let mut changed = manifest;
    if let Entry::File { mode, .. } = changed
        .entries
        .iter_mut()
        .find(|entry| entry.path() == "bin/data")
        .unwrap()
    {
        *mode = 0o755;
    }
    assert!(matches!(
        changed.resolve_entry("chain"),
        Err(Error::Invalid(_))
    ));
}
