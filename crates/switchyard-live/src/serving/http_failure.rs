//! Bounded HTTP error evidence without retaining server bodies or credentials.
use crate::Error;
use futures_util::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::time::Duration;

const CAPTURE_BYTES: usize = 4096;

#[derive(Serialize)]
struct Evidence {
    server_kind: &'static str,
    content_kind: &'static str,
    body_kind: &'static str,
    captured_bytes: usize,
    body_capture_complete: bool,
    body_prefix_sha256: String,
}

pub(super) async fn failure(operation: &str, response: reqwest::Response) -> Error {
    let status = response.status().as_u16();
    let server_kind = match response
        .headers()
        .get("server")
        .and_then(|v| v.to_str().ok())
    {
        Some(value) if value.starts_with("nginx") => "nginx",
        Some(value) if value.starts_with("cloudflare") => "cloudflare",
        Some(value) if value.starts_with("envoy") => "envoy",
        Some(_) => "other",
        None => "absent",
    };
    let content_kind = match response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
    {
        Some(value) if value.starts_with("application/json") => "json",
        Some(value) if value.starts_with("text/html") => "html",
        Some(value) if value.starts_with("text/plain") => "text",
        Some(_) => "other",
        None => "absent",
    };
    let mut bytes = Vec::with_capacity(CAPTURE_BYTES);
    let mut complete = false;
    let mut stream = response.bytes_stream();
    let capture = async {
        while let Some(piece) = stream.next().await {
            let Ok(piece) = piece else { return };
            let count = piece.len().min(CAPTURE_BYTES - bytes.len());
            bytes.extend_from_slice(&piece[..count]);
            if bytes.len() == CAPTURE_BYTES {
                return;
            }
        }
        complete = true;
    };
    // Error diagnostics must not create an unbounded read or hide a stalled
    // endpoint behind a retry. The response is dropped after this observation.
    let _ = tokio::time::timeout(Duration::from_secs(2), capture).await;
    let body_kind = if bytes.is_empty() {
        "empty"
    } else if serde_json::from_slice::<serde_json::Value>(&bytes).is_ok() {
        "json"
    } else if content_kind == "html" {
        "html"
    } else {
        "other"
    };
    let evidence = Evidence {
        server_kind,
        content_kind,
        body_kind,
        captured_bytes: bytes.len(),
        body_capture_complete: complete,
        body_prefix_sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    Error::Request(format!(
        "{operation} HTTP {status}; response_evidence={}",
        serde_json::to_string(&evidence).expect("fixed HTTP evidence is serializable")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{Response, StatusCode},
        routing::get,
    };

    async fn fixture(
        body: Vec<u8>,
        headers: bool,
    ) -> (reqwest::Response, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/",
            get(move || {
                let body = body.clone();
                async move {
                    let mut response = Response::builder().status(StatusCode::NOT_FOUND);
                    if headers {
                        response = response
                            .header("server", "nginx/private-value")
                            .header("content-type", "text/html");
                    }
                    response.body(Body::from(body)).unwrap()
                }
            }),
        );
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let response = reqwest::get(format!("http://{address}/")).await.unwrap();
        (response, task)
    }

    #[tokio::test]
    async fn retains_html_origin_and_hash_without_private_body_or_header_values() {
        let (response, task) = fixture(b"<html>private-credential</html>".to_vec(), true).await;
        let error = failure("tokenizer", response).await.to_string();
        task.abort();
        assert!(error.contains("tokenizer HTTP 404"));
        assert!(error.contains("\"server_kind\":\"nginx\""));
        assert!(error.contains("\"body_kind\":\"html\""));
        assert!(error.contains("\"body_capture_complete\":true"));
        assert!(!error.contains("private-"));
    }

    #[tokio::test]
    async fn caps_large_bodies_and_distinguishes_empty_route_errors() {
        let (response, task) = fixture(vec![b'x'; CAPTURE_BYTES * 3], false).await;
        let error = failure("inference", response).await.to_string();
        task.abort();
        assert!(error.contains("\"captured_bytes\":4096"));
        assert!(error.contains("\"body_capture_complete\":false"));
        let (response, task) = fixture(vec![], false).await;
        let error = failure("tokenizer", response).await.to_string();
        task.abort();
        assert!(error.contains("\"body_kind\":\"empty\""));
        assert!(error.contains("\"captured_bytes\":0"));
    }
}
