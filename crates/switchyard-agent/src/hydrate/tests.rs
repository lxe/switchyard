use super::*;
use crate::FileInfo;
use axum::{
    Router,
    body::Body,
    extract::{Path as RoutePath, State as AxumState},
    response::Response,
    routing::get,
};
use std::{path::PathBuf, sync::atomic::AtomicU64};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "switchyard-agent-{}-{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn tar_files(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut archive = tar::Builder::new(Vec::new());
    for (name, bytes) in files {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive.append_data(&mut header, name, *bytes).unwrap();
    }
    archive.into_inner().unwrap()
}

fn fixture(compress: &str) -> (Manifest, BTreeMap<String, Vec<u8>>) {
    let shards = [
        vec![("config.json", b"{\"model\":\"test\"}".as_slice())],
        vec![("weights/part.bin", b"0123456789abcdef".as_slice())],
    ];
    let mut chunks = Vec::new();
    let mut bodies = BTreeMap::new();
    let mut files = BTreeMap::new();
    for (shard, file_list) in shards.iter().enumerate() {
        for (path, bytes) in file_list {
            files.insert(
                (*path).into(),
                FileInfo {
                    size: bytes.len() as u64,
                    sha256: Some(sha(bytes)),
                },
            );
        }
        // Boundaries intentionally split tar headers and payloads.
        for (seq, raw) in tar_files(file_list).chunks(347).enumerate() {
            let body = if compress == "zstd" {
                zstd::stream::encode_all(raw, 1).unwrap()
            } else {
                raw.to_vec()
            };
            let name = format!("shard-{shard}-{seq}");
            chunks.push(Chunk {
                name: name.clone(),
                size: body.len() as u64,
                raw: raw.len() as u64,
                shard,
                seq,
                sha256: sha(&body),
            });
            bodies.insert(name, body);
        }
    }
    let identity = chunks.iter().map(|c| c.sha256.as_str()).collect::<String>();
    let manifest = Manifest {
        format: "shards-v1".into(),
        compress: compress.into(),
        total_size: chunks.iter().map(|c| c.size).sum(),
        raw_size: chunks.iter().map(|c| c.raw).sum(),
        sha256: sha(identity.as_bytes()),
        chunks,
        files,
    };
    (manifest, bodies)
}

#[derive(Clone)]
struct Source {
    bodies: Arc<BTreeMap<String, Vec<u8>>>,
    requests: Arc<AtomicUsize>,
    corrupt: Option<String>,
    fail_first: bool,
    attempts: Arc<Mutex<BTreeMap<String, usize>>>,
    frame_delay: Option<Duration>,
}
impl Source {
    fn new(bodies: BTreeMap<String, Vec<u8>>) -> Self {
        Self {
            bodies: Arc::new(bodies),
            requests: Arc::new(AtomicUsize::new(0)),
            corrupt: None,
            fail_first: false,
            attempts: Arc::new(Mutex::new(BTreeMap::new())),
            frame_delay: None,
        }
    }
}
struct SourceServer {
    base: String,
    handle: tokio::task::JoinHandle<()>,
}
impl Drop for SourceServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}
async fn source_server(source: Source) -> SourceServer {
    async fn chunk(
        AxumState(source): AxumState<Source>,
        RoutePath(name): RoutePath<String>,
    ) -> Response {
        source.requests.fetch_add(1, Ordering::Relaxed);
        let attempt = {
            let mut attempts = source.attempts.lock().unwrap();
            let a = attempts.entry(name.clone()).or_default();
            *a += 1;
            *a
        };
        let Some(body) = source.bodies.get(&name) else {
            return Response::builder().status(404).body(Body::empty()).unwrap();
        };
        let mut body = body.clone();
        if source.corrupt.as_deref() == Some(&name) && (!source.fail_first || attempt == 1) {
            body[0] ^= 1;
        }
        if let Some(delay) = source.frame_delay {
            let frames = body
                .chunks(64 << 10)
                .map(|bytes| bytes.to_vec())
                .collect::<Vec<_>>();
            let stream = futures_util::stream::iter(frames).then(move |bytes| async move {
                tokio::time::sleep(delay).await;
                Ok::<_, io::Error>(bytes)
            });
            Response::new(Body::from_stream(stream))
        } else {
            Response::new(Body::from(body))
        }
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/chunks/{name}", get(chunk))
                .with_state(source),
        )
        .await
        .unwrap();
    });
    SourceServer { base, handle }
}
fn ticket(manifest: Manifest, base: &str) -> Ticket {
    Ticket {
        model: "test-model".into(),
        revision: "test-revision".into(),
        urls: manifest
            .chunks
            .iter()
            .map(|c| {
                (
                    c.name.clone(),
                    format!("{base}/chunks/{}?signature=private-ticket", c.name),
                )
            })
            .collect(),
        manifest,
        engine_args: vec![],
        concurrency: 4,
        stall_s: 2,
        lifetime_s: 60,
        startup: Default::default(),
        runtime: None,
    }
}

#[tokio::test]
async fn hydrates_split_tar_shards_and_reuses_verified_cache() {
    let (manifest, bodies) = fixture("none");
    let source = Source::new(bodies);
    let server = source_server(source.clone()).await;
    let ticket = ticket(manifest, &server.base);
    let root = TempDir::new();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    hydrate(&ticket, &root.0, state.clone()).await.unwrap();
    assert_eq!(
        std::fs::read(root.0.join("model/weights/part.bin")).unwrap(),
        b"0123456789abcdef"
    );
    let observed = state.read().await;
    assert_eq!(observed.bytes_downloaded, observed.bytes_total);
    assert!(observed.volume_bytes_per_s.unwrap() > 0.0);
    assert!(!observed.model_cached);
    drop(observed);
    let before = source.requests.load(Ordering::Relaxed);
    let cached = Arc::new(RwLock::new(State::new(&ticket)));
    hydrate(&ticket, &root.0, cached.clone()).await.unwrap();
    assert!(cached.read().await.model_cached);
    assert_eq!(source.requests.load(Ordering::Relaxed), before);
}

#[tokio::test]
async fn hydrates_independently_compressed_chunks_and_replaces_corrupt_cache() {
    let (manifest, bodies) = fixture("zstd");
    let source = Source::new(bodies);
    let server = source_server(source.clone()).await;
    let ticket = ticket(manifest, &server.base);
    let root = TempDir::new();
    hydrate(&ticket, &root.0, Arc::new(RwLock::new(State::new(&ticket))))
        .await
        .unwrap();
    let before = source.requests.load(Ordering::Relaxed);
    std::fs::write(root.0.join("model/weights/part.bin"), b"fedcba9876543210").unwrap();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    hydrate(&ticket, &root.0, state.clone()).await.unwrap();
    assert!(!state.read().await.model_cached);
    assert!(source.requests.load(Ordering::Relaxed) > before);
    assert_eq!(
        std::fs::read(root.0.join("model/weights/part.bin")).unwrap(),
        b"0123456789abcdef"
    );
}

#[tokio::test]
async fn retry_progress_counts_retained_bytes_once() {
    let (manifest, bodies) = fixture("none");
    let mut source = Source::new(bodies);
    let failed_chunk = manifest.chunks[0].name.clone();
    source.corrupt = Some(failed_chunk.clone());
    source.fail_first = true;
    let server = source_server(source.clone()).await;
    let ticket = ticket(manifest, &server.base);
    let root = TempDir::new();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let operation = hydrate(&ticket, &root.0, state.clone());
    tokio::pin!(operation);
    loop {
        tokio::select! {
            result = &mut operation => { result.unwrap(); break; },
            _ = tokio::time::sleep(Duration::from_millis(2)) => {
                let st = state.read().await;
                assert!(st.bytes_downloaded <= st.bytes_total);
            }
        }
    }
    assert_eq!(
        state.read().await.bytes_downloaded,
        ticket.manifest.total_size
    );
    assert_eq!(source.attempts.lock().unwrap()[&failed_chunk], 2);
}

#[tokio::test]
async fn corrupt_chunk_never_publishes_and_errors_redact_the_url() {
    let (manifest, bodies) = fixture("none");
    let mut source = Source::new(bodies);
    source.corrupt = Some(manifest.chunks[0].name.clone());
    let server = source_server(source).await;
    let ticket = ticket(manifest, &server.base);
    let root = TempDir::new();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let error = hydrate(&ticket, &root.0, state.clone())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("digest or size mismatch"));
    assert!(!error.contains(&server.base));
    assert!(!error.contains("private-ticket"));
    assert!(!root.0.join("model").exists());
    assert!(state.read().await.bytes_downloaded <= ticket.manifest.total_size);
}

#[test]
fn validates_manifest_identity_and_sequences_before_transfer() {
    let (manifest, _) = fixture("none");
    assert!(manifest.validate().is_ok());
    let mut bad = manifest.clone();
    bad.total_size += 1;
    assert!(bad.validate().is_err());
    let mut bad = manifest.clone();
    bad.chunks[0].seq = 99;
    assert!(bad.validate().is_err());
    let mut bad = manifest.clone();
    bad.chunks[0].name = "../escape".into();
    assert!(bad.validate().is_err());
    let mut bad = manifest.clone();
    bad.files.insert(
        "../../escape".into(),
        FileInfo {
            size: 0,
            sha256: None,
        },
    );
    assert!(bad.validate().is_err());
    let mut bad = manifest.clone();
    bad.files.insert(
        "weights".into(),
        FileInfo {
            size: 0,
            sha256: None,
        },
    );
    assert!(bad.validate().is_err());
}

fn extract_fixture(bytes: Vec<u8>, manifest: &Manifest) -> Result<()> {
    let root = TempDir::new();
    let chunks = root.0.join("chunks");
    let dest = root.0.join("dest");
    std::fs::create_dir_all(&chunks)?;
    std::fs::create_dir_all(&dest)?;
    std::fs::write(chunks.join("payload"), &bytes)?;
    let mut manifest = manifest.clone();
    manifest.compress = "none".into();
    manifest.chunks = vec![Chunk {
        name: "payload".into(),
        size: bytes.len() as u64,
        raw: bytes.len() as u64,
        shard: 0,
        seq: 0,
        sha256: sha(&bytes),
    }];
    extract(&manifest, &chunks, &dest, 4)
}

#[test]
fn rejects_tar_traversal_links_duplicates_and_unexpected_files() {
    let (manifest, _) = fixture("none");
    let mut traversal = tar_files(&[("config.json", b"test")]);
    // Build the unsafe header directly because tar::Builder correctly rejects '..'.
    traversal[..100].fill(0);
    traversal[..9].copy_from_slice(b"../escape");
    let mut header = tar::Header::from_byte_slice(&traversal[..512]).clone();
    header.set_cksum();
    traversal[..512].copy_from_slice(header.as_bytes());
    assert!(extract_fixture(traversal, &manifest).is_err());
    let mut link = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    header.set_cksum();
    link.append_link(&mut header, "config.json", "../escape")
        .unwrap();
    assert!(extract_fixture(link.into_inner().unwrap(), &manifest).is_err());
    assert!(extract_fixture(tar_files(&[("unlisted.bin", b"test")]), &manifest).is_err());
    let file = b"{\"model\":\"test\"}".as_slice();
    assert!(
        extract_fixture(
            tar_files(&[("config.json", file), ("config.json", file)]),
            &manifest
        )
        .is_err()
    );
}

#[test]
fn exact_reader_detects_both_truncation_and_extra_raw_bytes() {
    let mut short = ExactReader {
        inner: b"ab".as_slice(),
        remaining: 3,
        checked_end: false,
    };
    assert!(io::copy(&mut short, &mut io::sink()).is_err());
    let mut long = ExactReader {
        inner: b"abcd".as_slice(),
        remaining: 3,
        checked_end: false,
    };
    assert!(io::copy(&mut long, &mut io::sink()).is_err());
}

#[cfg(unix)]
#[test]
fn cache_verification_rejects_symlinks_even_with_correct_size_and_hash() {
    let (mut manifest, _) = fixture("none");
    manifest.files.retain(|name, _| name == "config.json");
    let root = TempDir::new();
    let dest = root.0.join("model");
    std::fs::create_dir_all(&dest).unwrap();
    let outside = root.0.join("outside");
    std::fs::write(&outside, b"{\"model\":\"test\"}").unwrap();
    std::os::unix::fs::symlink(outside, dest.join("config.json")).unwrap();
    assert!(verify_files(&manifest, &dest).is_err());
}

#[tokio::test]
async fn trusted_streaming_extracts_split_zstd_without_hashing_and_reuses_receipt() {
    let (mut manifest, bodies) = fixture("zstd");
    // Syntactically valid but deliberately incorrect identities: this path trusts the pin.
    manifest.sha256 = "a".repeat(64);
    for chunk in &mut manifest.chunks {
        chunk.sha256 = "b".repeat(64);
    }
    for file in manifest.files.values_mut() {
        file.sha256 = Some("c".repeat(64));
    }
    let source = Source::new(bodies);
    let server = source_server(source.clone()).await;
    let mut ticket = ticket(manifest, &server.base);
    ticket.startup.mode = HydrationMode::TrustedStreaming;
    let root = TempDir::new();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    hydrate(&ticket, &root.0, state.clone()).await.unwrap();
    assert_eq!(
        std::fs::read(root.0.join("model/weights/part.bin")).unwrap(),
        b"0123456789abcdef"
    );
    assert_eq!(
        state.read().await.bytes_downloaded,
        ticket.manifest.total_size
    );
    assert!(
        std::fs::read_dir(&root.0)
            .unwrap()
            .all(|entry| entry.unwrap().file_name() == "model")
    );
    let before = source.requests.load(Ordering::Relaxed);
    std::fs::write(root.0.join("model/weights/part.bin"), b"caller-owned-cache").unwrap();
    hydrate(&ticket, &root.0, state.clone()).await.unwrap();
    assert!(state.read().await.model_cached);
    assert_eq!(source.requests.load(Ordering::Relaxed), before);
    assert_eq!(
        std::fs::read(root.0.join("model/weights/part.bin")).unwrap(),
        b"caller-owned-cache"
    );
}

fn large_fixture() -> (Manifest, BTreeMap<String, Vec<u8>>) {
    streaming_fixture(4 << 20, 128 << 10)
}

fn streaming_fixture(
    weight_size: usize,
    piece_size: usize,
) -> (Manifest, BTreeMap<String, Vec<u8>>) {
    let weights = vec![42; weight_size];
    let raw = tar_files(&[("config.json", b"{}"), ("weights.bin", &weights)]);
    let mut bodies = BTreeMap::new();
    let chunks = raw
        .chunks(piece_size)
        .enumerate()
        .map(|(seq, raw)| {
            let name = format!("piece-{seq:04}");
            bodies.insert(name.clone(), raw.to_vec());
            Chunk {
                name,
                size: raw.len() as u64,
                raw: raw.len() as u64,
                shard: 0,
                seq,
                sha256: sha(raw),
            }
        })
        .collect::<Vec<_>>();
    let identity = chunks
        .iter()
        .map(|chunk| chunk.sha256.as_str())
        .collect::<String>();
    (
        Manifest {
            format: "shards-v1".into(),
            compress: "none".into(),
            sha256: sha(identity.as_bytes()),
            total_size: chunks.iter().map(|chunk| chunk.size).sum(),
            raw_size: raw.len() as u64,
            files: BTreeMap::from([
                (
                    "config.json".into(),
                    FileInfo {
                        size: 2,
                        sha256: Some(sha(b"{}")),
                    },
                ),
                (
                    "weights.bin".into(),
                    FileInfo {
                        size: weights.len() as u64,
                        sha256: Some(sha(&weights)),
                    },
                ),
            ]),
            chunks,
        },
        bodies,
    )
}

#[tokio::test]
async fn trusted_single_shard_starts_twenty_streams_with_shared_budget() {
    #[derive(Clone)]
    struct ParallelSource {
        bodies: Arc<BTreeMap<String, Vec<u8>>>,
        entered: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        barrier: Arc<tokio::sync::Barrier>,
    }
    async fn piece(
        AxumState(source): AxumState<ParallelSource>,
        RoutePath(name): RoutePath<String>,
    ) -> Response {
        let count = source.entered.fetch_add(1, Ordering::Relaxed) + 1;
        source.peak.fetch_max(count, Ordering::Relaxed);
        if name.as_str() < "piece-0020" {
            source.barrier.wait().await;
        }
        // Delay the first wave together. A sequential downloader cannot pass the barrier.
        source.entered.fetch_sub(1, Ordering::Relaxed);
        Response::new(Body::from(source.bodies[&name].clone()))
    }
    let (manifest, bodies) = large_fixture();
    let source = ParallelSource {
        bodies: Arc::new(bodies),
        entered: Arc::new(AtomicUsize::new(0)),
        peak: Arc::new(AtomicUsize::new(0)),
        barrier: Arc::new(tokio::sync::Barrier::new(20)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let observed = source.peak.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/chunks/{name}", get(piece))
                .with_state(source),
        )
        .await
        .unwrap();
    });
    let mut ticket = ticket(manifest, &base);
    ticket.concurrency = 20;
    ticket.startup.mode = HydrationMode::TrustedStreaming;
    let root = TempDir::new();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let limit = Arc::new(tokio::sync::Semaphore::new(20));
    preflight(&ticket, &root.0, &state).await.unwrap();
    tokio::time::timeout(
        Duration::from_secs(5),
        hydrate_with_limit(&ticket, &root.0, state, limit.clone()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(observed.load(Ordering::Relaxed), 20);
    assert_eq!(limit.available_permits(), 20);
    server.abort();
}

#[tokio::test]
async fn trusted_extracts_before_final_transfer_and_cancellation_never_publishes() {
    #[derive(Clone)]
    struct DelayedSource {
        bodies: Arc<BTreeMap<String, Vec<u8>>>,
        final_name: String,
        waiting: Arc<tokio::sync::Notify>,
    }
    async fn piece(
        AxumState(source): AxumState<DelayedSource>,
        RoutePath(name): RoutePath<String>,
    ) -> Response {
        if name == source.final_name {
            source.waiting.notified().await;
        }
        Response::new(Body::from(source.bodies[&name].clone()))
    }
    let (manifest, bodies) = large_fixture();
    let source = DelayedSource {
        final_name: manifest.chunks.last().unwrap().name.clone(),
        bodies: Arc::new(bodies),
        waiting: Arc::new(tokio::sync::Notify::new()),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/chunks/{name}", get(piece))
                .with_state(source),
        )
        .await
        .unwrap();
    });
    let mut ticket = ticket(manifest, &base);
    ticket.startup.mode = HydrationMode::TrustedStreaming;
    let root = TempDir::new();
    let path = root.0.clone();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let limit = Arc::new(tokio::sync::Semaphore::new(4));
    let task_limit = limit.clone();
    preflight(&ticket, &root.0, &state).await.unwrap();
    let hydration =
        tokio::spawn(async move { hydrate_with_limit(&ticket, &path, state, task_limit).await });
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if std::fs::read_dir(&root.0).unwrap().any(|entry| {
                std::fs::read(entry.unwrap().path().join("model/config.json"))
                    .is_ok_and(|bytes| bytes == b"{}")
            }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(!root.0.join("model").exists());
    hydration.abort();
    let _ = hydration.await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if std::fs::read_dir(&root.0).unwrap().next().is_none()
                && limit.available_permits() == 4
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(!root.0.join("model").exists());
    server.abort();
}

#[tokio::test]
async fn trusted_shards_progress_with_contended_slots_and_out_of_order_responses() {
    #[derive(Clone)]
    struct ContendedSource {
        bodies: Arc<BTreeMap<String, Vec<u8>>>,
        arrivals: Arc<AtomicUsize>,
        active: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        first_wave: Arc<tokio::sync::Barrier>,
    }
    async fn piece(
        AxumState(source): AxumState<ContendedSource>,
        RoutePath(name): RoutePath<String>,
    ) -> Response {
        let active = source.active.fetch_add(1, Ordering::Relaxed) + 1;
        source.peak.fetch_max(active, Ordering::Relaxed);
        if source.arrivals.fetch_add(1, Ordering::Relaxed) < 5 {
            // Five slots remain after another artifact reserves fifteen. Network
            // requests must overlap even though extraction consumes each TAR in order.
            source.first_wave.wait().await;
        }
        if name.ends_with("-0000") {
            // Later pieces finish their HTTP bodies before their TAR predecessor.
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        source.active.fetch_sub(1, Ordering::Relaxed);
        Response::new(Body::from(source.bodies[&name].clone()))
    }
    let weights = vec![42; 2 << 20];
    let mut manifest = fixture("none").0;
    manifest.chunks.clear();
    manifest.files.clear();
    let mut bodies = BTreeMap::new();
    for shard in 0..2 {
        let path = format!("weights-{shard}.bin");
        manifest.files.insert(
            path.clone(),
            FileInfo {
                size: weights.len() as u64,
                sha256: Some(sha(&weights)),
            },
        );
        for (seq, raw) in tar_files(&[(&path, &weights)])
            .chunks(128 << 10)
            .enumerate()
        {
            let name = format!("piece-{shard}-{seq:04}");
            bodies.insert(name.clone(), raw.to_vec());
            manifest.chunks.push(Chunk {
                name,
                size: raw.len() as u64,
                raw: raw.len() as u64,
                shard,
                seq,
                sha256: sha(raw),
            });
        }
    }
    manifest.total_size = manifest.chunks.iter().map(|chunk| chunk.size).sum();
    manifest.raw_size = manifest.chunks.iter().map(|chunk| chunk.raw).sum();
    // Storage order is deliberately unhelpful; each shard must acquire slots in seq order.
    manifest.chunks.reverse();
    let source = ContendedSource {
        bodies: Arc::new(bodies),
        arrivals: Arc::new(AtomicUsize::new(0)),
        active: Arc::new(AtomicUsize::new(0)),
        peak: Arc::new(AtomicUsize::new(0)),
        first_wave: Arc::new(tokio::sync::Barrier::new(5)),
    };
    let observed = source.peak.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = SourceServer {
        base: base.clone(),
        handle: tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new()
                    .route("/chunks/{name}", get(piece))
                    .with_state(source),
            )
            .await
            .unwrap();
        }),
    };
    let mut ticket = ticket(manifest, &server.base);
    ticket.concurrency = 20;
    ticket.startup.mode = HydrationMode::TrustedStreaming;
    let root = TempDir::new();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let limit = Arc::new(tokio::sync::Semaphore::new(20));
    let other_artifact = limit.clone().acquire_many_owned(15).await.unwrap();
    preflight(&ticket, &root.0, &state).await.unwrap();
    tokio::time::timeout(
        Duration::from_secs(5),
        hydrate_with_limit(&ticket, &root.0, state, limit.clone()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(observed.load(Ordering::Relaxed), 5);
    assert_eq!(limit.available_permits(), 5);
    for shard in 0..2 {
        assert_eq!(
            std::fs::read(root.0.join(format!("model/weights-{shard}.bin"))).unwrap(),
            weights
        );
    }
    drop(other_artifact);
    assert_eq!(limit.available_permits(), 20);
}

#[tokio::test]
async fn trusted_large_pieces_refill_twenty_stream_window_across_multiple_rounds() {
    let (mut manifest, mut bodies) = streaming_fixture(80 << 20, 2 << 20);
    assert!(manifest.chunks.len() > 40);
    assert!(manifest.chunks[0].size > ((4 * 64) << 10));
    // Match a large weights TAR plus several small independent metadata TARs.
    for shard in 1..8 {
        let path = format!("metadata-{shard}.json");
        let raw = tar_files(&[(&path, b"{}")]);
        let name = format!("metadata-{shard}");
        manifest.chunks.push(Chunk {
            name: name.clone(),
            size: raw.len() as u64,
            raw: raw.len() as u64,
            shard,
            seq: 0,
            sha256: sha(&raw),
        });
        manifest.files.insert(
            path,
            FileInfo {
                size: 2,
                sha256: Some(sha(b"{}")),
            },
        );
        bodies.insert(name, raw);
    }
    manifest.total_size = manifest.chunks.iter().map(|chunk| chunk.size).sum();
    manifest.raw_size = manifest.chunks.iter().map(|chunk| chunk.raw).sum();
    let mut source = Source::new(bodies);
    source.frame_delay = Some(Duration::from_millis(1));
    let server = source_server(source.clone()).await;
    let mut ticket = ticket(manifest, &server.base);
    ticket.concurrency = 20;
    ticket.stall_s = 1;
    ticket.startup.mode = HydrationMode::TrustedStreaming;
    let root = TempDir::new();
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let limit = Arc::new(tokio::sync::Semaphore::new(20));
    preflight(&ticket, &root.0, &state).await.unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(8),
        hydrate_with_limit(&ticket, &root.0, state.clone(), limit.clone()),
    )
    .await;
    assert!(
        matches!(result, Ok(Ok(()))),
        "{result:?}; state={:?}; requests={:?}",
        state.read().await,
        source.attempts.lock().unwrap()
    );
    assert_eq!(
        std::fs::metadata(root.0.join("model/weights.bin"))
            .unwrap()
            .len(),
        80 << 20
    );
    assert_eq!(limit.available_permits(), 20);
    assert_eq!(
        state.read().await.bytes_downloaded,
        ticket.manifest.total_size
    );
}
