#![cfg(unix)]
use axum::{
    Router,
    body::Body,
    extract::{Path as HttpPath, State},
    response::Response,
    routing::get,
};
use std::{
    collections::BTreeMap,
    fs,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use switchyard_artifacts::{
    HydrationMode, Metadata,
    extents::{ExtentHydration, ExtentOptions, ExtentTicket, pack_extents},
};
use switchyard_weights::WeightStream;
use tokio::sync::watch;

static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
#[derive(Clone)]
struct Http {
    data: Arc<BTreeMap<String, Vec<u8>>>,
    held: String,
    release: watch::Receiver<bool>,
}
async fn response(State(mut http): State<Http>, HttpPath(name): HttpPath<String>) -> Response {
    if name == http.held {
        while !*http.release.borrow() {
            http.release.changed().await.unwrap();
        }
    }
    Response::new(Body::from(http.data[&name].clone()))
}
fn tensor(name: &str) -> Vec<u8> {
    let mut header =
        format!("{{\"{name}\":{{\"dtype\":\"F32\",\"shape\":[1024],\"data_offsets\":[0,4096]}}}}");
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend(header.bytes());
    for i in 0..1024 {
        data.extend((i as f32).to_le_bytes());
    }
    data
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tensor_from_first_shard_is_ready_while_second_shard_header_is_held() {
    let root = std::env::temp_dir().join(format!(
        "weights-incremental-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(root.join("source")).unwrap();
    fs::write(root.join("source/a.safetensors"), tensor("a")).unwrap();
    fs::write(root.join("source/b.safetensors"), tensor("b")).unwrap();
    let manifest = pack_extents(
        &root.join("source"),
        &root.join("objects"),
        Metadata {
            kind: "weights-test".into(),
            compatibility: BTreeMap::new(),
            attributes: BTreeMap::new(),
        },
        1024,
    )
    .unwrap();
    let held = manifest
        .extents
        .iter()
        .find(|e| e.entry == 1 && e.offset == 0)
        .unwrap()
        .object
        .clone();
    let (release, gate) = watch::channel(false);
    let http = Http {
        data: Arc::new(
            manifest
                .extents
                .iter()
                .map(|e| {
                    (
                        e.object.clone(),
                        fs::read(root.join("objects").join(&e.object)).unwrap(),
                    )
                })
                .collect(),
        ),
        held,
        release: gate,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/{name}", get(response))
        .with_state(http);
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let ticket = ExtentTicket {
        urls: manifest
            .extents
            .iter()
            .map(|e| (e.object.clone(), format!("{base}/{}", e.object)))
            .collect(),
        manifest,
    };
    let h = Arc::new(
        ExtentHydration::prepare(
            ticket,
            &root.join("model"),
            ExtentOptions {
                mode: HydrationMode::TrustedStreaming,
                concurrency: 2,
                ..ExtentOptions::default()
            },
        )
        .unwrap(),
    );
    let running = h.clone();
    let task = tokio::spawn(async move { running.run().await });
    let mut weights = WeightStream::new(&h, 100_000).unwrap();
    let first = tokio::time::timeout(Duration::from_secs(3), weights.next_ready(&h))
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.tensor.name, "a");
    assert!(!h.is_complete(1, 0, 8).unwrap());
    assert!(!root.join("model").exists());
    let mut actual = vec![0_u8; 4096];
    first.bytes.read_into(0, &mut actual).unwrap();
    assert_eq!(actual, tensor("a")[first.tensor.offset as usize..]);
    release.send(true).unwrap();
    let second = weights.next_ready(&h).await.unwrap().unwrap();
    assert_eq!(second.tensor.name, "b");
    assert!(weights.next_ready(&h).await.unwrap().is_none());
    task.await.unwrap().unwrap();
    drop(first);
    drop(second);
    drop(h);
    server.abort();
    fs::remove_dir_all(root).unwrap();
}
