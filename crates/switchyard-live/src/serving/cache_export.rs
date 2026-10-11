//! Bounded, authenticated cache export transport; no cloud or object-store policy.
use super::Endpoint;
use crate::{Error, Result};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};
use switchyard_agent::export::ExportState;
use switchyard_artifacts::{Chunk, Manifest};
use tokio::io::AsyncWriteExt;

impl Endpoint {
    pub(crate) async fn start_cache_export(&self) -> Result<ExportState> {
        let response = self
            .client
            .post(format!("{}/cache/export", self.url))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|_| Error::Request("cache export start transport".into()))?;
        decode_status(response).await
    }

    pub(crate) async fn cache_export_status(&self) -> Result<ExportState> {
        let response = self
            .client
            .get(format!("{}/cache/export", self.url))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|_| Error::Request("cache export observation transport".into()))?;
        decode_status(response).await
    }

    /// Download exactly the validated manifest's chunks and publish its local
    /// manifest only after every length and digest has matched.
    pub(crate) async fn download_cache_export(
        &self,
        manifest: &Manifest,
        requirements: &BTreeMap<String, String>,
        directory: &Path,
        max_bytes: u64,
        concurrency: usize,
    ) -> Result<()> {
        manifest.validate()?;
        manifest.check_compatibility(requirements)?;
        if manifest.metadata.kind != "cache" || !(1..=32).contains(&concurrency) {
            return Err(Error::Config(
                "export is not a compatible serving cache".into(),
            ));
        }
        let bytes = manifest
            .chunks
            .iter()
            .try_fold(0_u64, |sum, chunk| sum.checked_add(chunk.size));
        if bytes.is_none_or(|bytes| bytes > max_bytes) {
            return Err(Error::Limit(
                "cache export size exceeds preparation limit".into(),
            ));
        }
        tokio::fs::create_dir(directory).await?;
        let operations = futures_util::stream::iter(manifest.chunks.iter())
            .map(|chunk| self.download_export_chunk(chunk, directory))
            .buffer_unordered(concurrency);
        tokio::pin!(operations);
        while let Some(result) = operations.next().await {
            result?;
        }
        let manifest_path = directory.join("manifest.json");
        let partial = Partial {
            path: manifest_path.with_extension("partial"),
        };
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial.path)
            .await?;
        file.write_all(&serde_json::to_vec_pretty(manifest)?)
            .await?;
        file.sync_all().await?;
        drop(file);
        // Once synchronized, publish without an async cancellation boundary.
        std::fs::rename(&partial.path, manifest_path)?;
        Ok(())
    }

    async fn download_export_chunk(&self, chunk: &Chunk, directory: &Path) -> Result<()> {
        let response = self
            .client
            .get(format!("{}/cache/export/chunks/{}", self.url, chunk.name))
            .send()
            .await
            .map_err(|_| Error::Request("cache chunk download transport".into()))?;
        if !response.status().is_success()
            || response.content_length() != Some(chunk.size)
            || response
                .headers()
                .get("x-artifact-sha256")
                .and_then(|value| value.to_str().ok())
                != Some(chunk.sha256.as_str())
        {
            return Err(Error::Request(
                "cache chunk response length or digest header differs".into(),
            ));
        }
        let path = directory.join(&chunk.name);
        let partial = Partial {
            path: path.with_extension("partial"),
        };
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial.path)
            .await?;
        let mut digest = Sha256::new();
        let mut size = 0_u64;
        let mut stream = response.bytes_stream();
        while let Some(bytes) = stream.next().await {
            let bytes =
                bytes.map_err(|_| Error::Request("cache chunk stream interrupted".into()))?;
            size = size
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| Error::Request("cache chunk size overflow".into()))?;
            if size > chunk.size {
                return Err(Error::Request("cache chunk exceeds manifest length".into()));
            }
            digest.update(&bytes);
            file.write_all(&bytes).await?;
        }
        if size != chunk.size || format!("{:x}", digest.finalize()) != chunk.sha256 {
            return Err(Error::Request(
                "cache chunk content integrity failed".into(),
            ));
        }
        file.sync_all().await?;
        drop(file);
        tokio::fs::rename(&partial.path, path).await?;
        Ok(())
    }
}

async fn decode_status(response: reqwest::Response) -> Result<ExportState> {
    if !response.status().is_success() {
        return Err(Error::Request(format!(
            "cache export HTTP {}",
            response.status().as_u16()
        )));
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(part) = stream.next().await {
        let part =
            part.map_err(|_| Error::Request("cache export status stream interrupted".into()))?;
        if bytes.len().saturating_add(part.len()) > 16 << 20 {
            return Err(Error::Limit("cache export status size limit".into()));
        }
        bytes.extend_from_slice(&part);
    }
    serde_json::from_slice(&bytes).map_err(|_| Error::Request("cache export status schema".into()))
}

struct Partial {
    path: PathBuf,
}
impl Drop for Partial {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{HeaderMap, Response, StatusCode},
        routing::get,
    };
    use switchyard_artifacts::{Metadata, PackOptions, pack};

    async fn fixture(corrupt: bool) -> (Endpoint, Manifest, PathBuf, tokio::task::JoinHandle<()>) {
        let root = std::env::temp_dir().join(format!(
            "switchyard-export-client-{}",
            crate::runtime::nonce().unwrap()
        ));
        let source = root.join("source");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("kernel.cubin"), b"prepared kernel").unwrap();
        let manifest = pack(
            &source,
            &root.join("packed"),
            Metadata {
                kind: "cache".into(),
                compatibility: BTreeMap::from([("architecture".into(), "x86_64".into())]),
                attributes: BTreeMap::new(),
            },
            PackOptions::default(),
        )
        .unwrap();
        let chunk = manifest.chunks[0].clone();
        let mut bytes = std::fs::read(root.join("packed").join(&chunk.name)).unwrap();
        if corrupt {
            bytes[0] ^= 1;
        }
        let app = Router::new().route(
            &format!("/cache/export/chunks/{}", chunk.name),
            get(move |headers: HeaderMap| {
                let bytes = bytes.clone();
                let chunk = chunk.clone();
                async move {
                    if headers.get("authorization").unwrap() != "Bearer test" {
                        return Response::builder()
                            .status(StatusCode::UNAUTHORIZED)
                            .body(Body::empty())
                            .unwrap();
                    }
                    Response::builder()
                        .header("content-length", chunk.size)
                        .header("x-artifact-sha256", chunk.sha256)
                        .body(Body::from(bytes))
                        .unwrap()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = Endpoint::new(
            format!("http://{}", listener.local_addr().unwrap()),
            "test",
            5,
        )
        .unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (endpoint, manifest, root, server)
    }

    #[tokio::test]
    async fn export_download_authenticates_and_checks_original_bytes_before_manifest() {
        let (endpoint, manifest, root, server) = fixture(false).await;
        let directory = root.join("download");
        endpoint
            .download_cache_export(
                &manifest,
                &manifest.metadata.compatibility,
                &directory,
                1 << 20,
                2,
            )
            .await
            .unwrap();
        let downloaded: Manifest =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest.id, downloaded.id);
        let chunk = &manifest.chunks[0];
        assert_eq!(
            std::fs::read(directory.join(&chunk.name)).unwrap(),
            std::fs::read(root.join("packed").join(&chunk.name)).unwrap()
        );
        server.abort();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn corrupt_export_never_publishes_a_manifest_or_partial_chunk() {
        let (endpoint, manifest, root, server) = fixture(true).await;
        let directory = root.join("download");
        assert!(
            endpoint
                .download_cache_export(
                    &manifest,
                    &manifest.metadata.compatibility,
                    &directory,
                    1 << 20,
                    2
                )
                .await
                .is_err()
        );
        assert!(!directory.join("manifest.json").exists());
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        server.abort();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn incompatible_or_oversized_export_is_rejected_before_download() {
        let (endpoint, manifest, root, server) = fixture(false).await;
        let directory = root.join("download");
        let wrong = BTreeMap::from([("architecture".into(), "aarch64".into())]);
        assert!(
            endpoint
                .download_cache_export(&manifest, &wrong, &directory, 1 << 20, 2)
                .await
                .is_err()
        );
        assert!(
            endpoint
                .download_cache_export(
                    &manifest,
                    &manifest.metadata.compatibility,
                    &directory,
                    1,
                    2
                )
                .await
                .is_err()
        );
        assert!(!directory.exists());
        server.abort();
        std::fs::remove_dir_all(root).unwrap();
    }
}
