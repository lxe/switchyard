use super::*;
use crate::{FileInfo, Manifest};
use axum::{extract::Query, http::Request as HttpRequest, routing::post};
use std::{collections::BTreeMap, io};
use tower::ServiceExt;

const TOKEN: &str = "test-token-with-at-least-32-characters";

#[tokio::test]
async fn readiness_requires_tokenization_and_decoding_after_health() {
    async fn tokenize(
        axum::Json(body): axum::Json<serde_json::Value>,
    ) -> axum::Json<serde_json::Value> {
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        axum::Json(serde_json::json!({"count":6,"tokens":[1,2,3,4,5,6]}))
    }
    async fn completion(
        axum::Json(body): axum::Json<serde_json::Value>,
    ) -> axum::Json<serde_json::Value> {
        assert_eq!(body["max_tokens"], 1);
        assert_eq!(body["stream"], false);
        axum::Json(
            serde_json::json!({"choices":[{"message":{"role":"assistant","content":"Yes"}}],"usage":{"prompt_tokens":6,"completion_tokens":1}}),
        )
    }
    async fn invalid() -> axum::Json<serde_json::Value> {
        axum::Json(
            serde_json::json!({"choices":[{"message":{"content":"secret-upstream-output"}}],"usage":{"prompt_tokens":6,"completion_tokens":0}}),
        )
    }
    for (path, should_pass) in [
        ("/v1/chat/completions", true),
        ("/invalid/v1/chat/completions", false),
    ] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route("/tokenize", post(tokenize))
            .route("/invalid/tokenize", post(tokenize))
            .route("/v1/chat/completions", post(completion))
            .route("/invalid/v1/chat/completions", post(invalid));
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let base = if path.starts_with("/invalid") {
            format!("http://{address}/invalid")
        } else {
            format!("http://{address}")
        };
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        assert!(
            client
                .get(format!("http://{address}/health"))
                .send()
                .await
                .unwrap()
                .status()
                .is_success()
        );
        let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let result = readiness::verify_observed(&client, &base, "test-model", |event| {
            observed
                .lock()
                .unwrap()
                .push(serde_json::to_value(event).unwrap());
            async {}
        })
        .await;
        assert_eq!(
            *observed.lock().unwrap(),
            vec![
                serde_json::json!("tokenization_started"),
                serde_json::json!("tokenization_response_received"),
                serde_json::json!("first_decode_started"),
                serde_json::json!("first_decode_response_received"),
            ]
        );
        assert_eq!(result.is_ok(), should_pass);
        if let Err(error) = result {
            assert!(!error.to_string().contains("secret-upstream-output"));
        }
        task.abort();
    }
}

#[tokio::test]
async fn readiness_rejects_empty_tokenization_and_oversized_responses() {
    async fn empty() -> axum::Json<serde_json::Value> {
        axum::Json(serde_json::json!({"count":0}))
    }
    async fn oversized() -> String {
        "x".repeat((1 << 20) + 1)
    }
    for router in [
        Router::new().route("/tokenize", post(empty)),
        Router::new().route("/tokenize", post(oversized)),
    ] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        assert!(
            readiness::verify(&reqwest::Client::new(), &base, "test")
                .await
                .is_err()
        );
        task.abort();
    }
}
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
