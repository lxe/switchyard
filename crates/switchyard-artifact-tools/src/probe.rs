//! Measure scoped object transfers without decoding, hashing, or writing their bodies.
use crate::{Error, Result};
use futures_util::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use switchyard_artifacts::{ArtifactTicket, Chunk, Manifest};
use switchyard_provider::{config::valid_key_prefix, storage::Storage};

#[derive(Debug, Clone, Copy)]
/// Concurrency is 1..=64; both deadlines are seconds in 1..=86400.
pub struct ProbeOptions {
    pub concurrency: usize,
    /// Deadline for each complete request, including headers and body.
    pub request_timeout_s: u64,
    /// Maximum wait for initial response headers or the next response-body frame.
    pub stall_timeout_s: u64,
}
impl Default for ProbeOptions {
    fn default() -> Self {
        Self {
            concurrency: 20,
            request_timeout_s: 600,
            stall_timeout_s: 30,
        }
    }
}
impl ProbeOptions {
    pub(crate) fn validate(self) -> Result<()> {
        if !(1..=64).contains(&self.concurrency)
            || !(1..=86400).contains(&self.request_timeout_s)
            || !(1..=86400).contains(&self.stall_timeout_s)
        {
            return Err(Error::Config("invalid download limits".into()));
        }
        Ok(())
    }
}

/// Per-shard monotonic observations; elapsed fields share the transfer epoch.
/// Header latency starts when this request is issued, excluding its concurrency queue.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeShard {
    pub name: String,
    pub downloaded_bytes: u64,
    pub header_latency_s: f64,
    pub headers_elapsed_s: f64,
    pub first_body_elapsed_s: Option<f64>,
    pub last_body_elapsed_s: Option<f64>,
    pub response_completion_elapsed_s: f64,
}

/// A single download-only measurement. Bodies have declared lengths checked, but
/// their hashes and compressed contents are deliberately not inspected. Receipts
/// contain object names and pinned references, never capabilities or credentials.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeReceipt {
    pub id: String,
    pub prefix: String,
    pub concurrency: usize,
    pub declared_bytes: u64,
    pub downloaded_bytes: u64,
    /// Manifest resolution and capability signing, excluded from transfer timing.
    pub controller_elapsed_s: f64,
    pub elapsed_s: f64,
    pub first_headers_elapsed_s: f64,
    pub first_body_elapsed_s: Option<f64>,
    pub last_body_elapsed_s: Option<f64>,
    pub last_response_completion_elapsed_s: f64,
    /// Bytes divided by complete transfer elapsed time, including request headers.
    pub bytes_per_s: f64,
    pub shards: Vec<ProbeShard>,
}

pub(crate) fn validate_reference(prefix: &str, id: &str) -> Result<()> {
    if !valid_key_prefix(prefix) || id.len() != 64 || !id.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(Error::Config("invalid artifact reference".into()));
    }
    Ok(())
}

pub(crate) async fn resolve(storage: &Storage, prefix: &str, id: &str) -> Result<ArtifactTicket> {
    validate_reference(prefix, id)?;
    let manifest: Manifest = storage.get_json(&format!("{prefix}/manifest.json")).await?;
    // This controller checks manifest structure and identity, not shard contents.
    manifest.validate()?;
    if manifest.id != id {
        return Err(Error::Config(
            "published artifact differs from pinned identity".into(),
        ));
    }
    let urls = manifest
        .chunks
        .iter()
        .map(|chunk| {
            Ok((
                chunk.name.clone(),
                storage.presign(&format!("{prefix}/{}", chunk.name))?,
            ))
        })
        .collect::<switchyard_provider::Result<_>>()?;
    Ok(ArtifactTicket { manifest, urls })
}

/// Download a pinned publication through bounded, concurrent, memory-only GETs.
/// Each body frame is counted and immediately discarded. There is no retry,
/// decompressor, extraction, body hash, chunk cache, or filesystem publication.
/// Dropping the future drops its in-flight response streams. A successful receipt
/// is a measurement of this caller's host and connection, not a throughput promise.
pub async fn probe_directory(
    storage: &Storage,
    prefix: &str,
    id: &str,
    options: ProbeOptions,
) -> Result<ProbeReceipt> {
    options.validate()?;
    let controller = Instant::now();
    let ticket = resolve(storage, prefix, id).await?;
    let controller_elapsed_s = controller.elapsed().as_secs_f64();
    let client = reqwest::Client::builder()
        .user_agent("Switchyard/0.1")
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(options.request_timeout_s))
        .build()
        .map_err(|_| Error::Probe("client initialization failed".into()))?;
    let started = Instant::now();
    let mut shards = futures_util::stream::iter(ticket.manifest.chunks.iter().cloned())
        .map(|chunk| {
            let client = client.clone();
            let url = ticket.urls[&chunk.name].clone();
            async move { download(&client, &url, &chunk, options, started).await }
        })
        .buffer_unordered(options.concurrency)
        .try_collect::<Vec<_>>()
        .await?;
    let elapsed_s = started.elapsed().as_secs_f64();
    shards.sort_by(|left, right| left.name.cmp(&right.name));
    let downloaded_bytes = shards
        .iter()
        .map(|shard| shard.downloaded_bytes)
        .sum::<u64>();
    let first_body_elapsed_s = shards
        .iter()
        .filter_map(|shard| shard.first_body_elapsed_s)
        .min_by(f64::total_cmp);
    let last_body_elapsed_s = shards
        .iter()
        .filter_map(|shard| shard.last_body_elapsed_s)
        .max_by(f64::total_cmp);
    Ok(ProbeReceipt {
        id: id.to_owned(),
        prefix: prefix.to_owned(),
        concurrency: options.concurrency,
        declared_bytes: ticket.manifest.chunks.iter().map(|chunk| chunk.size).sum(),
        downloaded_bytes,
        controller_elapsed_s,
        elapsed_s,
        first_headers_elapsed_s: shards
            .iter()
            .map(|shard| shard.headers_elapsed_s)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0),
        first_body_elapsed_s,
        last_body_elapsed_s,
        last_response_completion_elapsed_s: shards
            .iter()
            .map(|shard| shard.response_completion_elapsed_s)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0),
        bytes_per_s: downloaded_bytes as f64 / elapsed_s.max(f64::EPSILON),
        shards,
    })
}

async fn download(
    client: &reqwest::Client,
    url: &str,
    chunk: &Chunk,
    options: ProbeOptions,
    started: Instant,
) -> Result<ProbeShard> {
    let fail = |reason| Error::Probe(format!("{}: {reason}", chunk.name));
    let request_started = Instant::now();
    let response = tokio::time::timeout(
        Duration::from_secs(options.stall_timeout_s),
        client.get(url).send(),
    )
    .await
    .map_err(|_| fail("response headers stalled"))?
    .map_err(|_| fail("request failed or timed out"))?;
    if response.status() != reqwest::StatusCode::OK {
        return Err(fail("unexpected HTTP status"));
    }
    let header_latency_s = request_started.elapsed().as_secs_f64();
    let headers_elapsed_s = started.elapsed().as_secs_f64();
    let mut body = response.bytes_stream();
    let mut downloaded_bytes = 0_u64;
    let (mut first_body_elapsed_s, mut last_body_elapsed_s) = (None, None);
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(options.stall_timeout_s), body.next())
            .await
            .map_err(|_| fail("response body stalled"))?;
        let Some(frame) = frame else {
            break;
        };
        let frame = frame.map_err(|_| fail("response body failed or timed out"))?;
        downloaded_bytes = downloaded_bytes
            .checked_add(frame.len() as u64)
            .filter(|bytes| *bytes <= chunk.size)
            .ok_or_else(|| fail("response exceeds declared size"))?;
        if !frame.is_empty() {
            let elapsed = started.elapsed().as_secs_f64();
            first_body_elapsed_s.get_or_insert(elapsed);
            last_body_elapsed_s = Some(elapsed);
        }
    }
    if downloaded_bytes != chunk.size {
        return Err(fail("response is shorter than declared size"));
    }
    Ok(ProbeShard {
        name: chunk.name.clone(),
        downloaded_bytes,
        header_latency_s,
        headers_elapsed_s,
        first_body_elapsed_s,
        last_body_elapsed_s,
        response_completion_elapsed_s: started.elapsed().as_secs_f64(),
    })
}
