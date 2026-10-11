//! Conditional, streaming publication for immutable content-addressed artifacts.
use super::{SignableBody, SignableRequest, SigningSettings, Storage, sign, v4};
use crate::{Error, Result};
use futures_util::StreamExt;
use reqwest::StatusCode;
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Duration, SystemTime},
};
use tokio::io::AsyncReadExt;

impl Storage {
    /// Publish an immutable object after validating the local payload.
    ///
    /// Returns true when created, false when an existing object's bytes match.
    /// An existing different object is an error and is never overwritten. Upload
    /// and collision verification stream through bounded buffers, including large shards.
    pub async fn put_file_immutable(
        &self,
        key: &str,
        path: &Path,
        expected_sha256: &str,
        expected_size: u64,
    ) -> Result<bool> {
        let mut file = tokio::fs::File::open(path)
            .await
            .map_err(|_| Error::Config("artifact file cannot be opened".into()))?;
        let mut hash = Sha256::new();
        let mut size = 0u64;
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            let n = file
                .read(&mut buffer)
                .await
                .map_err(|_| Error::Config("artifact file cannot be read".into()))?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
            size = size
                .checked_add(n as u64)
                .ok_or_else(|| Error::Config("artifact size overflow".into()))?;
        }
        if size != expected_size || format!("{:x}", hash.finalize()) != expected_sha256 {
            return Err(Error::Config(
                "artifact file differs from its manifest".into(),
            ));
        }
        let url = self.url(key)?;
        let mut last_error = None;
        for attempt in 0..3 {
            let identity = self.credentials.clone().into();
            let mut settings = SigningSettings::default();
            settings.percent_encoding_mode = super::PercentEncodingMode::Single;
            settings.uri_path_normalization_mode = super::UriPathNormalizationMode::Disabled;
            settings.payload_checksum_kind =
                aws_sigv4::http_request::PayloadChecksumKind::XAmzSha256;
            let params = v4::SigningParams::builder()
                .identity(&identity)
                .region(&self.config.region)
                .name("s3")
                .time(SystemTime::now())
                .settings(settings)
                .build()
                .map_err(|_| Error::Config("storage signing settings".into()))?
                .into();
            let signable = SignableRequest::new(
                "PUT",
                url.as_str(),
                [("if-none-match", "*")].into_iter(),
                SignableBody::Precomputed(expected_sha256.into()),
            )
            .map_err(|_| Error::Config("signable artifact object".into()))?;
            let (instructions, _) = sign(signable, &params)
                .map_err(|_| Error::Transport("storage signing".into()))?
                .into_parts();
            let mut headers = reqwest::header::HeaderMap::new();
            for (name, value) in instructions.headers() {
                let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| Error::Config("signed header".into()))?;
                let mut value = reqwest::header::HeaderValue::from_str(value)
                    .map_err(|_| Error::Config("signed header".into()))?;
                value.set_sensitive(true);
                headers.insert(name, value);
            }
            headers.insert(
                "if-none-match",
                reqwest::header::HeaderValue::from_static("*"),
            );
            let file = tokio::fs::File::open(path)
                .await
                .map_err(|_| Error::Config("artifact file cannot be opened".into()))?;
            let stream = futures_util::stream::try_unfold(file, |mut file| async move {
                let mut buffer = vec![0u8; 64 * 1024];
                let count = file.read(&mut buffer).await?;
                if count == 0 {
                    Ok::<_, std::io::Error>(None)
                } else {
                    buffer.truncate(count);
                    Ok(Some((buffer, file)))
                }
            });
            let sent = self
                .client
                .put(url.clone())
                .headers(headers)
                .header(reqwest::header::CONTENT_LENGTH, expected_size)
                .timeout(Duration::from_secs(600))
                .body(reqwest::Body::wrap_stream(stream))
                .send()
                .await;
            match sent {
                Ok(response) if response.status().is_success() => return Ok(true),
                Ok(response) if response.status() == StatusCode::PRECONDITION_FAILED => {
                    self.verify_object(key, expected_sha256, expected_size)
                        .await?;
                    return Ok(false);
                }
                Ok(response)
                    if response.status().is_server_error()
                        || response.status() == StatusCode::TOO_MANY_REQUESTS =>
                {
                    last_error = Some(Error::Http {
                        status: response.status().as_u16(),
                        operation: "artifact publication".into(),
                    });
                }
                Ok(response) => {
                    return Err(Error::Http {
                        status: response.status().as_u16(),
                        operation: "artifact publication".into(),
                    });
                }
                Err(_) => last_error = Some(Error::Transport("artifact publication".into())),
            }
            if attempt < 2 {
                tokio::time::sleep(Duration::from_millis(250 * (attempt + 1))).await;
            }
        }
        Err(last_error.expect("publication attempted"))
    }

    async fn verify_object(
        &self,
        key: &str,
        expected_sha256: &str,
        expected_size: u64,
    ) -> Result<()> {
        let response = self
            .client
            .get(self.presign(key)?)
            .timeout(Duration::from_secs(600))
            .send()
            .await
            .map_err(|_| Error::Transport("artifact collision verification".into()))?;
        if !response.status().is_success() {
            return Err(Error::Http {
                status: response.status().as_u16(),
                operation: "artifact collision verification".into(),
            });
        }
        let mut bytes = 0u64;
        let mut hash = Sha256::new();
        let mut stream = response.bytes_stream();
        while let Some(piece) = stream.next().await {
            let piece =
                piece.map_err(|_| Error::Transport("artifact collision verification".into()))?;
            bytes = bytes
                .checked_add(piece.len() as u64)
                .ok_or_else(|| Error::Response("object size overflow".into()))?;
            if bytes > expected_size {
                return Err(Error::Response("immutable artifact object differs".into()));
            }
            hash.update(&piece);
        }
        if bytes != expected_size || format!("{:x}", hash.finalize()) != expected_sha256 {
            return Err(Error::Response("immutable artifact object differs".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Bytes, extract::State, http::HeaderMap, routing::put};
    use std::{
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };
    use tokio::sync::Mutex;

    #[derive(Clone, Default)]
    struct Objects {
        bytes: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
    }
    async fn upload(
        State(objects): State<Objects>,
        axum::extract::Path(key): axum::extract::Path<String>,
        headers: HeaderMap,
        bytes: Bytes,
    ) -> StatusCode {
        assert_eq!(headers.get("if-none-match").unwrap(), "*");
        assert!(
            headers
                .get("authorization")
                .unwrap()
                .to_str()
                .unwrap()
                .contains("if-none-match")
        );
        assert_eq!(
            headers
                .get("x-amz-content-sha256")
                .unwrap()
                .to_str()
                .unwrap(),
            format!("{:x}", Sha256::digest(&bytes))
        );
        let mut map = objects.bytes.lock().await;
        if map.contains_key(&key) {
            return StatusCode::PRECONDITION_FAILED;
        }
        map.insert(key, bytes.to_vec());
        StatusCode::CREATED
    }
    async fn download(
        State(objects): State<Objects>,
        axum::extract::Path(key): axum::extract::Path<String>,
    ) -> (StatusCode, Vec<u8>) {
        match objects.bytes.lock().await.get(&key) {
            Some(bytes) => (StatusCode::OK, bytes.clone()),
            None => (StatusCode::NOT_FOUND, vec![]),
        }
    }
    struct TempFile(std::path::PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn file(bytes: &[u8]) -> TempFile {
        static ID: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "switchyard-upload-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        TempFile(path)
    }
    #[tokio::test]
    async fn streaming_publication_is_verified_and_never_overwrites_existing_content() {
        let objects = Objects::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new()
            .route("/{*key}", put(upload).get(download))
            .layer(axum::extract::DefaultBodyLimit::max(4 << 20))
            .with_state(objects.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let store = Storage::with_keys(
            &crate::StorageConfig {
                endpoint,
                region: "auto".into(),
                bucket: "bucket".into(),
                access_key_env: "EXAMPLE_KEY".into(),
                secret_key_env: "EXAMPLE_SECRET".into(),
                url_lifetime_s: 900,
            },
            "example".into(),
            "example-secret".into(),
        )
        .unwrap();
        let bytes = vec![17u8; 2 * 1024 * 1024];
        let source = file(&bytes);
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert!(
            store
                .put_file_immutable("artifact/chunk", &source.0, &digest, bytes.len() as u64)
                .await
                .unwrap()
        );
        assert!(
            !store
                .put_file_immutable("artifact/chunk", &source.0, &digest, bytes.len() as u64)
                .await
                .unwrap()
        );
        let other = b"different";
        let conflicting = file(other);
        assert!(
            store
                .put_file_immutable(
                    "artifact/chunk",
                    &conflicting.0,
                    &format!("{:x}", Sha256::digest(other)),
                    other.len() as u64
                )
                .await
                .is_err()
        );
        assert!(
            store
                .put_file_immutable(
                    "artifact/bad",
                    &source.0,
                    &"0".repeat(64),
                    bytes.len() as u64
                )
                .await
                .is_err()
        );
        let stored = objects.bytes.lock().await;
        assert_eq!(stored["bucket/artifact/chunk"], bytes);
        assert!(!stored.contains_key("bucket/artifact/bad"));
        server.abort();
    }
}
