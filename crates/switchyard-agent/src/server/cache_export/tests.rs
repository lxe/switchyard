use super::*;
use crate::{export::CacheExportConfig, runtime::RuntimeTicket};
use axum::{Router, http::Request as HttpRequest, routing::get};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    sync::atomic::{AtomicUsize, Ordering},
};
use switchyard_artifacts::{ArtifactTicket, Metadata, PackOptions};
use tower::ServiceExt;

const TOKEN: &str = "test-token-with-at-least-32-characters";
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let mut random = [0_u8; 8];
        getrandom::fill(&mut random).unwrap();
        let path = std::env::temp_dir().join(format!(
            "switchyard-export-{:x}",
            u64::from_le_bytes(random)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Fixture {
    _temp: Temp,
    app: App,
    source: PathBuf,
    seeds: PathBuf,
    stopped: Arc<AtomicUsize>,
    owner: JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.owner.abort();
    }
}
fn fixture(engine_url: String, stop_succeeds: bool) -> Fixture {
    let temp = Temp::new();
    let roots = RootPaths {
        runtime: temp.0.join("runtime"),
        caches: temp.0.join("cache"),
        chunks: temp.0.join("artifacts"),
        models: temp.0.join("models"),
    };
    fs::create_dir(&roots.runtime).unwrap();
    fs::write(roots.runtime.join("engine"), "runtime").unwrap();
    let manifest = switchyard_artifacts::pack(
        &roots.runtime,
        &temp.0.join("runtime-pack"),
        Metadata {
            kind: "runtime".into(),
            compatibility: BTreeMap::new(),
            attributes: BTreeMap::new(),
        },
        PackOptions::default(),
    )
    .unwrap();
    let source = roots.caches.join(&manifest.id).join("compiled");
    let seeds = roots.caches.join(&manifest.id).join(".seeds/compiled");
    fs::create_dir_all(source.join("triton")).unwrap();
    fs::create_dir_all(&seeds).unwrap();
    fs::write(source.join("triton/kernel.bin"), b"compiled kernel output").unwrap();
    fs::write(source.join(".switchyard-cache-seed"), "a".repeat(64)).unwrap();
    fs::write(seeds.join("immutable.bin"), b"immutable seed").unwrap();
    let compatibility = BTreeMap::from([
        ("runtime_id".into(), manifest.id.clone()),
        ("model_revision".into(), "revision".into()),
        ("architecture".into(), std::env::consts::ARCH.into()),
        ("gpu".into(), "test-gpu".into()),
        ("engine_configuration".into(), "test-engine-profile".into()),
    ]);
    let ticket = Ticket {
        source: crate::model::ModelSource::Legacy(crate::model::LegacySource {
            manifest: crate::Manifest {
                format: "shards-v1".into(),
                compress: "none".into(),
                chunks: vec![],
                total_size: 0,
                raw_size: 0,
                sha256: "test".into(),
                files: BTreeMap::new(),
            },
            urls: BTreeMap::new(),
        }),
        model: "test".into(),
        revision: "revision".into(),
        engine_args: vec![],
        concurrency: 1,
        stall_s: 1,
        lifetime_s: 10,
        startup: Default::default(),
        runtime: Some(RuntimeTicket {
            components: vec![],
            hydration: None,
            artifact: ArtifactTicket {
                manifest,
                urls: BTreeMap::new(),
            },
            compatibility: BTreeMap::new(),
            program: "engine".into(),
            args: vec![],
            environment: BTreeMap::new(),
            caches: vec![],
            cache_export: Some(CacheExportConfig {
                name: "compiled".into(),
                compatibility,
                drain_timeout_s: 1,
                shutdown_timeout_s: 1,
            }),
        }),
    };
    let mut app = super::super::tests::app(Phase::Ready, engine_url);
    let (control, mut commands) = tokio::sync::mpsc::channel::<engine::Stop>(1);
    let stopped = Arc::new(AtomicUsize::new(0));
    let count = stopped.clone();
    let owner = tokio::spawn(async move {
        while let Some(command) = commands.recv().await {
            count.fetch_add(1, Ordering::SeqCst);
            let _ = command.reply.send(if stop_succeeds {
                Ok(())
            } else {
                Err(Error::Transfer("test shutdown failed".into()))
            });
        }
    });
    app.export = CacheExport::new(
        &ticket,
        &roots,
        app.admission.clone(),
        control,
        app.state.clone(),
    )
    .unwrap();
    Fixture {
        _temp: temp,
        app,
        source,
        seeds,
        stopped,
        owner,
    }
}
fn request(method: &str, path: &str) -> HttpRequest<Body> {
    HttpRequest::builder()
        .method(method)
        .uri(path)
        .header("authorization", format!("Bearer {TOKEN}"))
        .body(Body::empty())
        .unwrap()
}
async fn start(app: &App) -> Response {
    super::super::routes(app.clone())
        .oneshot(request("POST", "/cache/export"))
        .await
        .unwrap()
}
async fn finished(app: &App) -> ExportState {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let status = app.export.as_ref().unwrap().job.lock().await.status.clone();
            if matches!(status.phase, ExportPhase::Ready | ExportPhase::Failed) {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn capability_is_disabled_for_ordinary_tickets_and_requires_authentication() {
    let app = super::super::tests::app(Phase::Ready, "http://127.0.0.1:1".into());
    let routes = super::super::routes(app);
    for (method, path) in [
        ("POST", "/cache/export"),
        ("GET", "/cache/export"),
        ("GET", "/cache/export/chunks/file"),
    ] {
        assert_eq!(
            routes
                .clone()
                .oneshot(request(method, path))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
    }
    let fixture = fixture("http://127.0.0.1:1".into(), true);
    let unauthorized = HttpRequest::builder()
        .uri("/cache/export")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        super::super::routes(fixture.app.clone())
            .oneshot(unauthorized)
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn export_is_idempotent_and_downloads_match_the_manifest() {
    let fixture = fixture("http://127.0.0.1:1".into(), true);
    let (first, second) = tokio::join!(start(&fixture.app), start(&fixture.app));
    assert!(matches!(
        first.status(),
        StatusCode::ACCEPTED | StatusCode::OK
    ));
    assert!(matches!(
        second.status(),
        StatusCode::ACCEPTED | StatusCode::OK
    ));
    let status = finished(&fixture.app).await;
    assert_eq!(status.phase, ExportPhase::Ready);
    assert_eq!(fixture.stopped.load(Ordering::SeqCst), 1);
    assert_eq!(start(&fixture.app).await.status(), StatusCode::OK);
    let manifest = status.manifest.unwrap();
    manifest.validate().unwrap();
    assert!(
        manifest
            .entries
            .iter()
            .all(|entry| !entry.path().contains(".seeds")
                && entry.path() != ".switchyard-cache-seed")
    );
    assert_eq!(manifest.metadata.kind, "cache");
    assert_eq!(
        manifest.metadata.compatibility,
        fixture.app.export.as_ref().unwrap().config.compatibility
    );
    let routes = super::super::routes(fixture.app.clone());
    for chunk in &manifest.chunks {
        let response = routes
            .clone()
            .oneshot(request(
                "GET",
                &format!("/cache/export/chunks/{}", chunk.name),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-artifact-sha256"], chunk.sha256);
        assert_eq!(response.headers()["content-length"], chunk.size.to_string());
        let body = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        assert_eq!(body.len() as u64, chunk.size);
        assert_eq!(format!("{:x}", Sha256::digest(&body)), chunk.sha256);
        let mut archive = tar::Archive::new(zstd::stream::read::Decoder::new(&body[..]).unwrap());
        let mut found = false;
        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            if entry.path().unwrap().as_ref() == std::path::Path::new("triton/kernel.bin") {
                let mut bytes = vec![];
                std::io::Read::read_to_end(&mut entry, &mut bytes).unwrap();
                assert_eq!(bytes, b"compiled kernel output");
                found = true;
            }
        }
        assert!(found);
    }
    for name in [
        "not-listed.tar.zst",
        "..%2F.seeds%2Fcompiled",
        "%00",
        "manifest.json",
    ] {
        assert_eq!(
            routes
                .clone()
                .oneshot(request("GET", &format!("/cache/export/chunks/{name}")))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
    }
    assert_eq!(
        fs::read(fixture.seeds.join("immutable.bin")).unwrap(),
        b"immutable seed"
    );
    assert_eq!(
        fs::read_to_string(fixture.source.join(".switchyard-cache-seed")).unwrap(),
        "a".repeat(64)
    );
    assert!(matches!(
        fixture.app.state.read().await.phase,
        Phase::Exporting
    ));
    assert!(!fixture.app.state.read().await.inference_verified);
}

#[tokio::test]
async fn export_drains_streamed_bodies_until_eof_or_disconnect() {
    for disconnect in [false, true] {
        let (pieces, receiver) = tokio::sync::mpsc::channel::<Vec<u8>>(2);
        pieces.send(b"data: first\n\n".to_vec()).await.unwrap();
        let receiver = Arc::new(Mutex::new(Some(receiver)));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new().route(
            "/v1/models",
            get(move || {
                let receiver = receiver.clone();
                async move {
                    let receiver = receiver.lock().await.take().unwrap();
                    Body::from_stream(futures_util::stream::unfold(
                        receiver,
                        |mut receiver| async move {
                            receiver
                                .recv()
                                .await
                                .map(|bytes| (Ok::<_, std::io::Error>(bytes), receiver))
                        },
                    ))
                }
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let fixture = fixture(url, true);
        let response = super::super::routes(fixture.app.clone())
            .oneshot(request("GET", "/v1/models"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(start(&fixture.app).await.status(), StatusCode::ACCEPTED);
        assert_eq!(fixture.stopped.load(Ordering::SeqCst), 0);
        assert_eq!(
            fixture
                .app
                .export
                .as_ref()
                .unwrap()
                .job
                .lock()
                .await
                .status
                .phase,
            ExportPhase::Draining
        );
        assert_eq!(
            super::super::routes(fixture.app.clone())
                .oneshot(request("GET", "/v1/models"))
                .await
                .unwrap()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        if disconnect {
            drop(response);
        } else {
            let collecting = tokio::spawn(axum::body::to_bytes(response.into_body(), 4096));
            pieces.send(b"data: done\n\n".to_vec()).await.unwrap();
            drop(pieces);
            assert_eq!(
                &collecting.await.unwrap().unwrap()[..],
                b"data: first\n\ndata: done\n\n"
            );
        }
        assert_eq!(finished(&fixture.app).await.phase, ExportPhase::Ready);
        assert_eq!(fixture.stopped.load(Ordering::SeqCst), 1);
        server.abort();
    }
}

#[tokio::test]
async fn failed_shutdown_or_packing_never_publishes_an_export() {
    for bad_source in [false, true] {
        let fixture = fixture("http://127.0.0.1:1".into(), bad_source);
        if bad_source {
            std::os::unix::fs::symlink("../../secret-cache-path", fixture.source.join("outside"))
                .unwrap();
        }
        assert_eq!(start(&fixture.app).await.status(), StatusCode::ACCEPTED);
        let status = finished(&fixture.app).await;
        assert_eq!(status.phase, ExportPhase::Failed);
        assert!(status.manifest.is_none());
        assert!(!status.error.unwrap().contains("secret-cache-path"));
        assert!(
            fixture
                .app
                .export
                .as_ref()
                .unwrap()
                .job
                .lock()
                .await
                .output
                .is_none()
        );
        assert_eq!(start(&fixture.app).await.status(), StatusCode::CONFLICT);
        assert_eq!(
            super::super::routes(fixture.app.clone())
                .oneshot(request("GET", "/cache/export/chunks/anything"))
                .await
                .unwrap()
                .status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            fs::read(fixture.seeds.join("immutable.bin")).unwrap(),
            b"immutable seed"
        );
        let scratch = &fixture.app.export.as_ref().unwrap().scratch;
        assert!(!scratch.exists() || fs::read_dir(scratch).unwrap().next().is_none());
    }
}

#[tokio::test]
async fn export_rejects_payloads_and_startup_without_closing_admission() {
    let fixture = fixture("http://127.0.0.1:1".into(), true);
    let request = HttpRequest::builder()
        .method("POST")
        .uri("/cache/export")
        .header("authorization", format!("Bearer {TOKEN}"))
        .body(Body::from("{\"name\":\".seeds\"}"))
        .unwrap();
    assert_eq!(
        super::super::routes(fixture.app.clone())
            .oneshot(request)
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    fixture.app.state.write().await.phase = Phase::Initializing;
    assert_eq!(start(&fixture.app).await.status(), StatusCode::CONFLICT);
    assert!(fixture.app.admission.enter().await.is_some());
    assert_eq!(fixture.stopped.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn drain_timeout_preserves_failure_and_does_not_pack_or_stop_active_work() {
    let fixture = fixture("http://127.0.0.1:1".into(), true);
    let active = fixture.app.admission.enter().await.unwrap();
    assert_eq!(start(&fixture.app).await.status(), StatusCode::ACCEPTED);
    let status = finished(&fixture.app).await;
    assert_eq!(status.phase, ExportPhase::Failed);
    assert!(status.manifest.is_none());
    assert_eq!(fixture.stopped.load(Ordering::SeqCst), 0);
    drop(active);
    assert_eq!(start(&fixture.app).await.status(), StatusCode::CONFLICT);
}
