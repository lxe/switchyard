#![cfg(target_os = "linux")]
use axum::{
    Router,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::Response,
    routing::get,
};
use std::future::IntoFuture;
use std::{
    collections::BTreeMap,
    fs,
    sync::{Arc, Mutex},
    time::Duration,
};
use switchyard_artifacts::{Metadata, extents::*, range_stream::*};
async fn response(
    State(objects): State<Arc<BTreeMap<String, Vec<u8>>>>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> Response {
    let data = &objects[&name];
    let (start, end) = headers
        .get("range")
        .map(|h| {
            h.to_str()
                .unwrap()
                .strip_prefix("bytes=")
                .unwrap()
                .split_once('-')
                .unwrap()
        })
        .map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()))
        .unwrap_or((0, data.len() - 1));
    // Intentionally complete requests in a different order from the artifact.
    if start == 0 {
        tokio::time::sleep(Duration::from_millis(60)).await;
    }
    Response::builder()
        .status(if headers.contains_key("range") {
            StatusCode::PARTIAL_CONTENT
        } else {
            StatusCode::OK
        })
        .header("content-length", end - start + 1)
        .header(
            "content-range",
            format!("bytes {start}-{end}/{}", data.len()),
        )
        .body(Body::from(data[start..=end].to_vec()))
        .unwrap()
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delayed_slow_consumer_reclaims_and_bounds_large_single_tensor() {
    let root = std::env::temp_dir().join(format!("bounded-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("src")).unwrap();
    // Represents one tensor 32 times the entire RAM budget. No consumer assembles it.
    let data = (0..(32 << 20))
        .map(|i| (i * 13 + 17) as u8)
        .collect::<Vec<_>>();
    fs::write(root.join("src/large.tensor"), &data).unwrap();
    let manifest = pack_extents(
        &root.join("src"),
        &root.join("objects"),
        Metadata {
            kind: "synthetic".into(),
            compatibility: BTreeMap::new(),
            attributes: BTreeMap::new(),
        },
        32 << 20,
    )
    .unwrap();
    let objects = Arc::new(
        manifest
            .extents
            .iter()
            .map(|e| {
                (
                    e.object.clone(),
                    fs::read(root.join("objects").join(&e.object)).unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>(),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(
        axum::serve(
            listener,
            Router::new()
                .route("/{name}", get(response))
                .with_state(objects),
        )
        .into_future(),
    );
    let ticket = ExtentTicket {
        urls: manifest
            .extents
            .iter()
            .map(|e| (e.object.clone(), format!("http://{addr}/{}", e.object)))
            .collect(),
        manifest,
    };
    let limits = BufferLimits {
        chunk_bytes: 256 << 10,
        ram_bytes: 1 << 20,
        spool_bytes: 2 << 20,
        download_concurrency: 6,
        download_buffer_bytes: 64 << 10,
        pinned_bytes: 2 << 20,
        transfer_concurrency: 2,
    };
    let events = Arc::new(Mutex::new(vec![]));
    let observed = events.clone();
    let receiver = RangeStream::new(
        &root.join("spool"),
        limits.clone(),
        Some(RangeObserver::new(move |e| {
            observed.lock().unwrap().push(e)
        })),
    )
    .unwrap();
    let downloader = receiver.clone();
    let run = tokio::spawn(async move { downloader.run(&ticket, Duration::from_secs(30)).await });
    tokio::time::sleep(Duration::from_millis(200)).await;
    let before = receiver.progress();
    assert_eq!(before.ram_bytes, limits.ram_bytes);
    assert_eq!(before.spool_bytes, limits.spool_bytes);
    assert!(!before.consumer_available);
    let mut consumed = 0;
    let mut order = vec![];
    while let Some(chunk) = receiver.next_ready().await.unwrap() {
        order.push(chunk.spec.id);
        let mut buffer = vec![0; chunk.spec.length as usize];
        chunk.copying();
        chunk.read_into(&mut buffer).unwrap();
        assert_eq!(
            &buffer,
            &data[chunk.spec.offset as usize..(chunk.spec.offset + chunk.spec.length) as usize]
        );
        consumed += buffer.len();
        let reader = chunk.clone();
        chunk.consumed();
        tokio::time::sleep(Duration::from_millis(2)).await;
        assert!(
            receiver.progress().ram_bytes + receiver.progress().spool_bytes >= chunk_size(&reader)
        );
        drop(reader);
    }
    run.await.unwrap().unwrap();
    let stats = receiver.progress();
    assert_eq!(consumed, data.len());
    assert_eq!(stats.ram_bytes + stats.spool_bytes, 0);
    assert_eq!(stats.reclaimed_bytes, consumed as u64);
    assert!(
        stats.peak_ram_bytes <= limits.ram_bytes && stats.peak_spool_bytes <= limits.spool_bytes
    );
    assert!(stats.exhausted_buffer_seconds > 0.0);
    assert_ne!(order, (0..order.len()).collect::<Vec<_>>());
    receiver.cancel();
    assert!(matches!(
        receiver.next_ready().await,
        Err(switchyard_artifacts::Error::Cancelled)
    ));
    drop(receiver);
    server.abort();
    fs::remove_dir_all(root).unwrap();
}

fn chunk_size(chunk: &ReadyRange) -> u64 {
    chunk.spec.padded_length
}
