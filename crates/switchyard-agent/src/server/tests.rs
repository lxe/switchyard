use super::*;
use crate::{FileInfo, Manifest};
use axum::{extract::Query, http::Request as HttpRequest, routing::post};
use std::{collections::BTreeMap, io};
use tower::ServiceExt;

const TOKEN: &str = "test-token-with-at-least-32-characters";

pub(super) fn app(phase: Phase, engine_url: String) -> App {
    let ticket = Ticket {
        source: crate::model::ModelSource::Legacy(crate::model::LegacySource {
            manifest: Manifest {
                format: "shards-v1".into(),
                compress: "none".into(),
                chunks: vec![],
                total_size: 0,
                raw_size: 0,
                sha256: "test".into(),
                files: BTreeMap::<String, FileInfo>::new(),
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
        runtime: None,
    };
    let mut state = State::new(&ticket);
    state.phase = phase;
    App {
        state: Arc::new(RwLock::new(state)),
        client: reqwest::Client::new(),
        authorization_hash: Sha256::digest(format!("Bearer {TOKEN}").as_bytes()).into(),
        engine_url,
        admission: admission::Admission::default(),
        export: None,
    }
}
fn request(path: &str, authorized: bool) -> HttpRequest<Body> {
    let mut request = HttpRequest::builder().uri(path);
    if authorized {
        request = request.header("authorization", format!("Bearer {TOKEN}"));
    }
    request.body(Body::empty()).unwrap()
}

#[tokio::test]
async fn observations_require_authentication_and_preserve_unknown_image_cache() {
    let routes = routes(app(Phase::Hydrating, "http://127.0.0.1:1".into()));
    assert_eq!(
        routes
            .clone()
            .oneshot(request("/state", false))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let response = routes
        .clone()
        .oneshot(request("/probe", true))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    let state: State = serde_json::from_slice(&body).unwrap();
    assert!(state.image_cached.is_none());
    assert!(matches!(state.phase, Phase::Hydrating));
    assert!(!String::from_utf8_lossy(&body).contains(TOKEN));
    assert_eq!(
        routes
            .oneshot(request("/v1/models", true))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn streaming_proxy_preserves_query_and_usage_without_forwarding_agent_token() {
    async fn engine(Query(query): Query<BTreeMap<String, String>>, request: Request) -> Response {
        assert!(request.headers().get("authorization").is_none());
        assert_eq!(query.get("mode").map(String::as_str), Some("test"));
        let body = axum::body::to_bytes(request.into_body(), 4096)
            .await
            .unwrap();
        assert_eq!(&body[..], b"{\"stream\":true}");
        let pieces = [
            b"data: {\"choices\":[]}".as_slice(),
            b"\n\ndata: [DONE]\n\n".as_slice(),
        ];
        Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(futures_util::stream::iter(
                pieces.map(Ok::<_, io::Error>),
            )))
            .unwrap()
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async {
        axum::serve(
            listener,
            Router::new().route("/v1/chat/completions", post(engine)),
        )
        .await
        .unwrap();
    });
    let routes = routes(app(Phase::Ready, url));
    let request = HttpRequest::builder()
        .method("POST")
        .uri("/v1/chat/completions?mode=test")
        .header("authorization", format!("Bearer {TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from("{\"stream\":true}"))
        .unwrap();
    let response = routes.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(response.headers()["x-accel-buffering"], "no");
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    assert_eq!(&body[..], b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n");
    assert_eq!(
        routes
            .oneshot(super::tests::request("/private", true))
            .await
            .unwrap()
            .status(),
        StatusCode::NOT_FOUND
    );
    server.abort();
}

#[tokio::test]
async fn proxy_bounds_request_size_and_redacts_upstream_failures() {
    let routes = routes(app(Phase::Ready, "http://127.0.0.1:1".into()));
    let large = HttpRequest::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {TOKEN}"))
        .body(Body::from(vec![0; (2 << 20) + 1]))
        .unwrap();
    assert_eq!(
        routes.clone().oneshot(large).await.unwrap().status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let response = routes.oneshot(request("/health", true)).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(
        axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap()
            .is_empty()
    );
}
