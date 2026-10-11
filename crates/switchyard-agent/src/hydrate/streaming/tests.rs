use super::*;
#[tokio::test]
async fn cancelled_downloader_keeps_permit_with_decoding_worker() {
    struct Paused(std::sync::mpsc::Receiver<()>);
    impl Read for Paused {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            self.0.recv().unwrap();
            Ok(0)
        }
    }
    let limit = Arc::new(Semaphore::new(1));
    let permit = Arc::new(Mutex::new(Some(
        limit.clone().acquire_owned().await.unwrap(),
    )));
    let (release, waiting) = std::sync::mpsc::channel();
    let (started, ready) = oneshot::channel();
    let (consumed, _completion) = oneshot::channel();
    let mut reader = DecodedPiece {
        inner: ExactReader {
            inner: Box::new(Paused(waiting)),
            remaining: 0,
            checked_end: false,
        },
        consumed: Some(consumed),
        permit: permit.clone(),
        trace: diagnostics(chunk(0)).transfer("piece"),
        shard_trace: diagnostics(chunk(0)).shard(0),
        seq: 0,
    };
    let decoding = tokio::task::spawn_blocking(move || {
        let _ = started.send(());
        assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    });
    ready.await.unwrap();
    // Drop the async downloader's ownership while the decoder is still blocked.
    drop(permit);
    decoding.abort();
    assert_eq!(limit.available_permits(), 0);
    release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while limit.available_permits() != 1 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

use crate::{Manifest, Ticket};
use axum::{Router, body::Body, response::Response, routing::get};

struct Server {
    url: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn server(router: Router) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/piece", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    Server { url, task }
}
fn ticket(lifetime_s: u64) -> Ticket {
    Ticket {
        source: crate::model::ModelSource::Legacy(crate::model::LegacySource {
            manifest: Manifest {
                format: "shards-v1".into(),
                compress: "none".into(),
                sha256: "a".repeat(64),
                total_size: 0,
                raw_size: 0,
                chunks: vec![],
                files: BTreeMap::new(),
            },
            urls: BTreeMap::new(),
        }),
        model: "test".into(),
        revision: "test".into(),
        engine_args: vec![],
        concurrency: 1,
        stall_s: 1,
        lifetime_s,
        runtime: None,
        startup: Default::default(),
    }
}
fn chunk(size: usize) -> Chunk {
    Chunk {
        name: "piece".into(),
        size: size as u64,
        raw: size as u64,
        shard: 0,
        seq: 0,
        sha256: "b".repeat(64),
    }
}

fn diagnostics(piece: Chunk) -> ModelDiagnostics {
    let mut manifest = ticket(10).legacy().unwrap().manifest.clone();
    manifest.chunks.push(piece);
    ModelDiagnostics::new(
        &manifest,
        &Arc::new(Semaphore::new(1)),
        Instant::now(),
        CHANNEL_PIECES,
    )
}

async fn paused_consumer(client: reqwest::Client) -> (Result<()>, usize) {
    const SIZE: usize = 2 << 20;
    let server =
        server(Router::new().route("/piece", get(|| async { Body::from(vec![42_u8; SIZE]) })))
            .await;
    let state = Arc::new(RwLock::new(State::new(&ticket(10))));
    let (sender, mut receiver) = mpsc::channel::<Vec<u8>>(CHANNEL_PIECES);
    let capacity = sender.clone();
    let diagnostic = diagnostics(chunk(SIZE));
    let observed = diagnostic.clone();
    let downloading = tokio::spawn(async move {
        transfer(
            &client,
            &server.url,
            &chunk(SIZE),
            sender,
            &state,
            &diagnostic,
            Duration::from_secs(1),
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while capacity.capacity() != 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    // Every consumer slot is full; the network transfer now waits on local extraction.
    tokio::time::sleep(Duration::from_millis(1250)).await;
    assert!(!downloading.is_finished());
    let snapshot = observed.snapshot();
    let piece = &snapshot.transfers[0];
    assert_eq!(piece.boundary, Boundary::ChannelSend);
    assert_eq!(piece.channel_available_slots, 0);
    assert!(piece.boundary_age_s > 1.0);
    assert!(piece.received_bytes > piece.delivered_bytes);
    assert!(piece.body_wait_s < 1.0);
    drop(capacity);
    let mut received = 0;
    while let Some(bytes) = receiver.recv().await {
        received += bytes.len();
    }
    (downloading.await.unwrap(), received)
}

#[tokio::test]
async fn consumer_backpressure_exceeding_stall_is_not_a_network_timeout() {
    let previous = reqwest::Client::builder()
        .read_timeout(Duration::from_secs(1))
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let (error, bytes) = paused_consumer(previous).await;
    assert!(error.unwrap_err().to_string().contains("piece interrupted"));
    assert!(bytes < 2 << 20);
    let (result, bytes) = paused_consumer(client(&ticket(10).legacy().unwrap()).unwrap()).await;
    result.unwrap();
    assert_eq!(bytes, 2 << 20);
}

#[tokio::test]
async fn total_request_deadline_still_bounds_a_paused_consumer() {
    let (result, bytes) = paused_consumer(client(&ticket(1).legacy().unwrap()).unwrap()).await;
    assert!(result.is_err());
    assert!(bytes < 2 << 20);
}

#[tokio::test]
async fn absent_response_headers_fail_the_network_stall_deadline() {
    let server = server(Router::new().route(
        "/piece",
        get(|| async {
            std::future::pending::<()>().await;
            Body::empty()
        }),
    ))
    .await;
    let state = RwLock::new(State::new(&ticket(10)));
    let (sender, _receiver) = mpsc::channel::<Vec<u8>>(CHANNEL_PIECES);
    let observed = diagnostics(chunk(128));
    let secret_url = format!("{}?secret=do-not-log-this-capability", server.url);
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        transfer(
            &client(&ticket(10).legacy().unwrap()).unwrap(),
            &secret_url,
            &chunk(128),
            sender,
            &state,
            &observed,
            Duration::from_secs(1),
        ),
    )
    .await
    .unwrap();
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("piece headers stalled")
    );
}

#[tokio::test]
async fn stalled_response_body_fails_without_local_backpressure() {
    let server = server(Router::new().route(
        "/piece",
        get(|| async {
            let body = futures_util::stream::once(async { Ok::<_, io::Error>(vec![42_u8; 64]) })
                .chain(futures_util::stream::pending::<io::Result<Vec<u8>>>());
            Response::new(Body::from_stream(body))
        }),
    ))
    .await;
    let state = RwLock::new(State::new(&ticket(10)));
    let (sender, mut receiver) = mpsc::channel::<Vec<u8>>(CHANNEL_PIECES);
    let draining = tokio::spawn(async move {
        let mut received = 0;
        while let Some(bytes) = receiver.recv().await {
            received += bytes.len();
        }
        received
    });
    let observed = diagnostics(chunk(128));
    let secret_url = format!("{}?secret=do-not-log-this-capability", server.url);
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        transfer(
            &client(&ticket(10).legacy().unwrap()).unwrap(),
            &secret_url,
            &chunk(128),
            sender,
            &state,
            &observed,
            Duration::from_secs(1),
        ),
    )
    .await
    .unwrap();
    assert!(result.unwrap_err().to_string().contains("piece stalled"));
    assert_eq!(draining.await.unwrap(), 64);
    let snapshot = observed.snapshot();
    let failure = snapshot.failure.unwrap();
    assert!(failure.host_capture_complete);
    assert_eq!(failure.error_class, "body_wait_deadline");
    let piece = &failure.transfers[0];
    assert_eq!(piece.boundary, Boundary::BodyWait);
    assert_eq!(piece.received_bytes, 64);
    assert_eq!(piece.delivered_bytes, 64);
    assert!(piece.channel_available_slots > 0);
    assert!(piece.longest_body_wait_s >= 1.0);
    assert!(piece.channel_wait_s < 0.1);
    assert!(failure.host.tasks.len() <= 64);
    assert!(failure.host.sockets.len() <= 128);
    let encoded = serde_json::to_string(&observed).unwrap();
    assert!(!encoded.contains("do-not-log-this-capability"));
    assert!(!encoded.contains("secret="));
    let restored: ModelDiagnostics = serde_json::from_str(&encoded).unwrap();
    assert_eq!(restored.snapshot().transfers[0].received_bytes, 64);
}

#[test]
fn diagnostic_record_counts_are_bounded_and_legacy_state_is_compatible() {
    let mut ticket = ticket(10);
    ticket.legacy_source_mut().manifest.chunks = (0..300)
        .map(|index| {
            let mut piece = chunk(1);
            piece.name = format!("piece-{index}");
            piece.shard = index;
            piece
        })
        .collect();
    let limit = Arc::new(Semaphore::new(20));
    let diagnostic = ModelDiagnostics::new(
        ticket.legacy().unwrap().manifest,
        &limit,
        Instant::now(),
        CHANNEL_PIECES,
    );
    let observed = diagnostic.snapshot();
    assert_eq!(observed.transfers.len(), 256);
    assert_eq!(observed.omitted_transfers, 44);
    assert_eq!(observed.shards.len(), 64);
    assert_eq!(observed.omitted_shards, 236);
    assert_eq!(observed.available_permits, Some(20));
    let state: State =
        serde_json::from_value(serde_json::to_value(State::new(&ticket)).unwrap()).unwrap();
    assert!(state.model_diagnostics.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_timeout_snapshot_is_complete_before_consumer_eof_failure() {
    let server = server(Router::new().route(
        "/piece",
        get(|| async {
            let body = futures_util::stream::once(async { Ok::<_, io::Error>(vec![42_u8; 64]) })
                .chain(futures_util::stream::pending::<io::Result<Vec<u8>>>());
            Response::new(Body::from_stream(body))
        }),
    ))
    .await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .unwrap();
    let observed = diagnostics(chunk(128));
    let (sender, mut receiver) = mpsc::channel::<Vec<u8>>(CHANNEL_PIECES);
    let (eof_observation, eof_seen) = oneshot::channel();
    let consumer_diagnostics = observed.clone();
    let decoder = tokio::task::spawn_blocking(move || {
        let mut read = 0;
        while let Some(bytes) = receiver.blocking_recv() {
            read += bytes.len();
        }
        // This is the decoder EOF boundary which used to cancel async host capture.
        let _ = eof_observation.send((read, consumer_diagnostics.snapshot().failure));
        Err::<(), _>(Error::Transfer("consumer decoding failed at EOF".into()))
    });
    let url = format!("{}?signed-capability=never-retain-this-value", server.url);
    let state = RwLock::new(State::new(&ticket(10)));
    let piece = chunk(128);
    let operation = transfer(
        &client,
        &url,
        &piece,
        sender,
        &state,
        &observed,
        Duration::from_secs(1),
    );
    let result = tokio::try_join!(operation, async { decoder.await.map_err(|_| Error::Task)? });
    assert!(result.is_err());
    let (read, failure) = eof_seen.await.unwrap();
    assert_eq!(read, 64);
    let failure = failure.expect("first transport failure must precede consumer EOF");
    assert!(failure.host_capture_complete);
    assert!(!failure.host.process.is_empty() || !failure.host.unavailable.is_empty());
    assert_eq!(failure.error_class, "reqwest_timeout");
    let details = failure.transfers[0].transport_error.as_ref().unwrap();
    assert!(details.reqwest_flags.timeout);
    assert!(details.reqwest_flags.decode || details.reqwest_flags.body);
    assert!(matches!(
        details.causes[0],
        super::super::diagnostics::ErrorCause::Reqwest { .. }
    ));
    observed.failure("other-piece", "consumer_closed");
    assert_eq!(
        observed.snapshot().failure.unwrap().error_class,
        "reqwest_timeout"
    );
    let encoded = serde_json::to_string(&observed).unwrap();
    assert!(!encoded.contains("never-retain-this-value"));
    assert!(!encoded.contains("signed-capability"));
}

#[tokio::test]
async fn truncated_http_body_retains_hyper_and_specific_io_causes_without_error_text() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let serving = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let _ = stream.read(&mut [0_u8; 2048]).await.unwrap();
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 128\r\n\r\nshort body")
            .await
            .unwrap();
        stream.shutdown().await.unwrap();
    });
    let observed = diagnostics(chunk(128));
    let (sender, mut receiver) = mpsc::channel::<Vec<u8>>(CHANNEL_PIECES);
    let draining = tokio::spawn(async move { while receiver.recv().await.is_some() {} });
    let result = transfer(
        &reqwest::Client::new(),
        &format!("http://{address}/piece?token=must-not-leak"),
        &chunk(128),
        sender,
        &RwLock::new(State::new(&ticket(10))),
        &observed,
        Duration::from_secs(1),
    )
    .await;
    assert!(result.is_err());
    serving.await.unwrap();
    draining.await.unwrap();
    let failure = observed.snapshot().failure.unwrap();
    assert!(failure.host_capture_complete);
    let details = failure.transfers[0].transport_error.as_ref().unwrap();
    assert!(!details.reqwest_flags.timeout);
    assert!(
        details
            .causes
            .iter()
            .any(|cause| matches!(cause, super::super::diagnostics::ErrorCause::Hyper { .. })),
        "{details:?}"
    );
    assert!(details.causes.iter().any(|cause| matches!(
        cause,
        super::super::diagnostics::ErrorCause::Io { kind, raw_os_error: None }
            if kind == "UnexpectedEof"
    )));
    assert!(
        !serde_json::to_string(&failure)
            .unwrap()
            .contains("must-not-leak")
    );
}

#[test]
fn permanent_channel_cancellation_escapes_standard_io_retry_loops() {
    struct Bounded<R> {
        inner: R,
        calls: usize,
    }
    impl<R: Read> Read for Bounded<R> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.calls > 8 {
                return Err(io::Error::other("retry limit reached"));
            }
            self.inner.read(buffer)
        }
    }
    for copy in [false, true] {
        let (_sender, receiver) = mpsc::channel(1);
        let channel = ChannelReader {
            receiver,
            pending: io::Cursor::new(vec![]),
            cancelled: Arc::new(AtomicBool::new(true)),
            trace: diagnostics(chunk(0)).transfer("piece"),
        };
        let mut reader = Bounded {
            inner: channel,
            calls: 0,
        };
        let error = if copy {
            io::copy(&mut reader, &mut io::sink()).unwrap_err()
        } else {
            reader.read_exact(&mut [0; 512]).unwrap_err()
        };
        assert_eq!(error.kind(), io::ErrorKind::ConnectionAborted);
        assert_eq!(
            reader.calls, 1,
            "permanent cancellation must not be retried by std::io"
        );
    }
}
