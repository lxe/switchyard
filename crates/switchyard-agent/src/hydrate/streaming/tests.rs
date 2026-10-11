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

use crate::Manifest;
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
        model: "test".into(),
        revision: "test".into(),
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

async fn paused_consumer(client: reqwest::Client) -> (Result<()>, usize) {
    const SIZE: usize = 2 << 20;
    let server =
        server(Router::new().route("/piece", get(|| async { Body::from(vec![42_u8; SIZE]) })))
            .await;
    let state = Arc::new(RwLock::new(State::new(&ticket(10))));
    let (sender, mut receiver) = mpsc::channel::<Vec<u8>>(CHANNEL_PIECES);
    let capacity = sender.clone();
    let downloading = tokio::spawn(async move {
        transfer(
            &client,
            &server.url,
            &chunk(SIZE),
            sender,
            &state,
            Instant::now(),
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
    let (result, bytes) = paused_consumer(client(&ticket(10)).unwrap()).await;
    result.unwrap();
    assert_eq!(bytes, 2 << 20);
}

#[tokio::test]
async fn total_request_deadline_still_bounds_a_paused_consumer() {
    let (result, bytes) = paused_consumer(client(&ticket(1)).unwrap()).await;
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
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        transfer(
            &client(&ticket(10)).unwrap(),
            &server.url,
            &chunk(128),
            sender,
            &state,
            Instant::now(),
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
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        transfer(
            &client(&ticket(10)).unwrap(),
            &server.url,
            &chunk(128),
            sender,
            &state,
            Instant::now(),
            Duration::from_secs(1),
        ),
    )
    .await
    .unwrap();
    assert!(result.unwrap_err().to_string().contains("piece stalled"));
    assert_eq!(draining.await.unwrap(), 64);
}
