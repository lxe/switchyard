#![cfg(unix)]

use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
};
use futures_util::stream;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use switchyard_artifacts::{
    ArtifactTicket, Error, HydrationOptions, Manifest, Metadata, PackOptions, Phase, Progress,
    hydrate, pack,
};

struct Fixture {
    root: PathBuf,
    manifest: Manifest,
    body: Vec<u8>,
}
impl Fixture {
    fn new() -> Self {
        let mut random = [0_u8; 12];
        getrandom::fill(&mut random).unwrap();
        let nonce = random
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let root = std::env::temp_dir().join(format!("switchyard-artifacts-http-{nonce}"));
        let source = root.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("file"), "verified transport fixture").unwrap();
        let manifest = pack(
            &source,
            &root.join("packed"),
            Metadata {
                kind: "transport-test".into(),
                compatibility: BTreeMap::new(),
                attributes: BTreeMap::new(),
            },
            PackOptions::default(),
        )
        .unwrap();
        let body = fs::read(root.join("packed").join(&manifest.chunks[0].name)).unwrap();
        Self {
            root,
            manifest,
            body,
        }
    }
    fn options(&self) -> HydrationOptions {
        HydrationOptions {
            cache_dir: self.root.join("cache"),
            retries: 0,
            request_timeout_s: 5,
            stall_timeout_s: 1,
            ..HydrationOptions::default()
        }
    }
    async fn serve(&self, app: Router) -> (ArtifactTicket, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://{}/chunk?signature=must-remain-private",
            listener.local_addr().unwrap()
        );
        let ticket = ArtifactTicket {
            manifest: self.manifest.clone(),
            urls: [(self.manifest.chunks[0].name.clone(), url)].into(),
        };
        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (ticket, handle)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[derive(Clone)]
struct Stall {
    body: Arc<Vec<u8>>,
    attempts: Arc<AtomicUsize>,
    ranges: Arc<Mutex<Vec<Option<String>>>>,
}
async fn stall_then_resume(State(state): State<Stall>, headers: HeaderMap) -> Response {
    let range = headers.get("range").map(|v| v.to_str().unwrap().to_owned());
    state.ranges.lock().unwrap().push(range.clone());
    let attempt = state.attempts.fetch_add(1, Ordering::Relaxed);
    if attempt == 0 {
        let half = Bytes::copy_from_slice(&state.body[..state.body.len() / 2]);
        let streamed = stream::unfold(Some(half), |pending| async move {
            if let Some(bytes) = pending {
                Some((Ok::<_, std::io::Error>(bytes), None))
            } else {
                std::future::pending().await
            }
        });
        return Response::new(Body::from_stream(streamed));
    }
    let start = range
        .unwrap()
        .strip_prefix("bytes=")
        .unwrap()
        .trim_end_matches('-')
        .parse::<usize>()
        .unwrap();
    Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(
            "content-range",
            format!(
                "bytes {start}-{}/{}",
                state.body.len() - 1,
                state.body.len()
            ),
        )
        .body(Body::from(state.body[start..].to_vec()))
        .unwrap()
}

#[tokio::test]
async fn stalled_body_retries_only_its_missing_range() {
    let fixture = Fixture::new();
    let state = Stall {
        body: Arc::new(fixture.body.clone()),
        attempts: Arc::new(AtomicUsize::new(0)),
        ranges: Arc::new(Mutex::new(vec![])),
    };
    let (ticket, server) = fixture
        .serve(
            Router::new()
                .route("/chunk", get(stall_then_resume))
                .with_state(state.clone()),
        )
        .await;
    let mut options = fixture.options();
    options.retries = 1;
    let report = tokio::time::timeout(
        Duration::from_secs(4),
        hydrate(ticket, &fixture.root.join("destination"), options, |_| {}),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(state.attempts.load(Ordering::Relaxed), 2);
    assert_eq!(
        state.ranges.lock().unwrap().as_slice(),
        &[None, Some(format!("bytes={}-", fixture.body.len() / 2))]
    );
    assert_eq!(report.downloaded_bytes, fixture.body.len() as u64);
    assert_eq!(report.verified_bytes, report.downloaded_bytes);
    assert_eq!(report.network_verified_bytes, report.downloaded_bytes);
    server.abort();
}

async fn trickle(State(body): State<Arc<Vec<u8>>>) -> Response {
    let pieces = body
        .chunks(body.len().div_ceil(5))
        .map(Bytes::copy_from_slice)
        .collect::<Vec<_>>();
    let streamed = stream::unfold(pieces.into_iter(), |mut pieces| async move {
        let bytes = pieces.next()?;
        tokio::time::sleep(Duration::from_millis(450)).await;
        Some((Ok::<_, std::io::Error>(bytes), pieces))
    });
    Response::new(Body::from_stream(streamed))
}

#[tokio::test]
async fn total_attempt_deadline_expires_even_when_body_never_stalls() {
    let fixture = Fixture::new();
    let (ticket, server) = fixture
        .serve(
            Router::new()
                .route("/chunk", get(trickle))
                .with_state(Arc::new(fixture.body.clone())),
        )
        .await;
    let mut options = fixture.options();
    options.request_timeout_s = 1;
    options.stall_timeout_s = 2;
    let error = tokio::time::timeout(
        Duration::from_secs(2),
        hydrate(ticket, &fixture.root.join("destination"), options, |_| {}),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(matches!(error, Error::Transfer));
    assert!(!fixture.root.join("destination").exists());
    assert!(!error.to_string().contains("must-remain-private"));
    server.abort();
}

#[derive(Clone)]
struct Corruption {
    valid: Arc<Vec<u8>>,
    attempts: Arc<AtomicUsize>,
}
async fn corrupt_once(State(state): State<Corruption>) -> Response {
    let mut body = state.valid.as_ref().clone();
    if state.attempts.fetch_add(1, Ordering::Relaxed) == 0 {
        body[0] ^= 1;
    }
    (StatusCode::OK, body).into_response()
}

#[tokio::test]
async fn corrupt_download_is_retried_and_wire_bytes_include_the_rejected_attempt() {
    let fixture = Fixture::new();
    let state = Corruption {
        valid: Arc::new(fixture.body.clone()),
        attempts: Arc::new(AtomicUsize::new(0)),
    };
    let (ticket, server) = fixture
        .serve(
            Router::new()
                .route("/chunk", get(corrupt_once))
                .with_state(state.clone()),
        )
        .await;
    let mut options = fixture.options();
    options.retries = 1;
    let progress = Arc::new(Mutex::new(vec![]));
    let recorded = progress.clone();
    let report = hydrate(
        ticket,
        &fixture.root.join("destination"),
        options,
        move |p| recorded.lock().unwrap().push(p),
    )
    .await
    .unwrap();
    assert_eq!(state.attempts.load(Ordering::Relaxed), 2);
    assert_eq!(report.downloaded_bytes, fixture.body.len() as u64 * 2);
    assert_eq!(report.verified_bytes, fixture.body.len() as u64);
    assert_eq!(report.network_verified_bytes, report.verified_bytes);
    assert!(
        progress
            .lock()
            .unwrap()
            .iter()
            .all(|p: &Progress| p.verified_bytes <= p.total_bytes
                && p.network_verified_bytes <= p.verified_bytes)
    );
    server.abort();
}

#[tokio::test]
async fn response_exceeding_manifest_size_is_rejected_but_counted_as_network_traffic() {
    let fixture = Fixture::new();
    let size = fixture.body.len() + 1;
    let (ticket, server) = fixture
        .serve(Router::new().route("/chunk", get(move || async move { vec![0_u8; size] })))
        .await;
    let bytes = Arc::new(Mutex::new(0));
    let recorded = bytes.clone();
    let error = hydrate(
        ticket,
        &fixture.root.join("destination"),
        fixture.options(),
        move |p| {
            if p.phase == Phase::Downloading {
                *recorded.lock().unwrap() = p.downloaded_bytes;
            }
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::Integrity(_)));
    assert_eq!(*bytes.lock().unwrap(), size as u64);
    assert!(!fixture.root.join("destination").exists());
    server.abort();
}
