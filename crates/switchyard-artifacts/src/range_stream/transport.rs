//! Concurrent HTTP range downloads into reserved backing.
use super::*;
use crate::extents::{Extent, ExtentTicket};
use aligned_vec::{AVec, ConstAlign};
use futures_util::{StreamExt, stream};
impl RangeStream {
    pub async fn run(&self, ticket: &ExtentTicket, timeout: Duration) -> Result<()> {
        self.run_prioritized(ticket, timeout, &[]).await
    }
    pub async fn run_prioritized(
        &self,
        ticket: &ExtentTicket,
        timeout: Duration,
        priority_entries: &[usize],
    ) -> Result<()> {
        ticket.manifest.validate_structure()?;
        let mut requests = vec![];
        let mut id = 0;
        for extent in &ticket.manifest.extents {
            let mut n = 0;
            while n < extent.length {
                let length = (extent.length - n).min(self.shared.limits.chunk_bytes as u64);
                let spec = RangeSpec {
                    id,
                    entry: extent.entry,
                    offset: extent.offset + n,
                    length,
                    padded_length: length.div_ceil(ALIGN as u64) * ALIGN as u64,
                };
                requests.push((spec, extent.clone(), n));
                n += length;
                id += 1;
            }
        }
        requests.sort_by_key(|(spec, _, _)| !priority_entries.contains(&spec.entry));
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|_| Error::Transfer)?;
        let result = async {
            let mut work = stream::iter(requests)
                .map(|(spec, extent, relative)| {
                    let client = &client;
                    async move {
                        let range = self.reserve(spec).await?;
                        self.receive(client, ticket, extent, relative, range).await
                    }
                })
                .buffer_unordered(self.shared.limits.download_concurrency);
            while let Some(result) = work.next().await {
                result?;
            }
            Ok::<_, Error>(())
        }
        .await;
        if result.is_err() {
            self.cancel()
        } else {
            self.shared.state.lock().unwrap().finished = true;
            self.shared.wake.notify_waiters();
        }
        result
    }
    async fn receive(
        &self,
        client: &reqwest::Client,
        ticket: &ExtentTicket,
        extent: Extent,
        relative: u64,
        range: ReadyRange,
    ) -> Result<()> {
        self.shared
            .emit(Some(range.spec.id), Some(Lifecycle::Downloading));
        let offset = extent.object_offset + relative;
        let size = range.spec.length;
        let ranged = offset != 0 || size != extent.object_size;
        let url = ticket
            .urls
            .get(&extent.object)
            .ok_or_else(|| Error::Invalid("missing range URL".into()))?;
        let mut request = client.get(url).header("Accept-Encoding", "identity");
        if ranged {
            request = request.header("Range", format!("bytes={offset}-{}", offset + size - 1));
        }
        let response = request.send().await.map_err(|_| Error::Transfer)?;
        if response.status().as_u16() != if ranged { 206 } else { 200 } {
            return Err(Error::Http(response.status().as_u16()));
        }
        if response
            .headers()
            .get("content-encoding")
            .is_some_and(|h| h != "identity")
            || response.content_length().is_some_and(|l| l != size)
            || ranged
                && response
                    .headers()
                    .get("content-range")
                    .and_then(|h| h.to_str().ok())
                    != Some(
                        format!(
                            "bytes {offset}-{}/{}",
                            offset + size - 1,
                            extent.object_size
                        )
                        .as_str(),
                    )
        {
            return Err(Error::Integrity("range HTTP boundary".into()));
        }
        let shared = self.shared.clone();
        let body = stream::try_unfold(response.bytes_stream(), move |mut body| {
            let shared = shared.clone();
            async move {
                let began = Instant::now();
                let next = body.next().await;
                shared.state.lock().unwrap().stats.network_wait_seconds +=
                    began.elapsed().as_secs_f64();
                shared
                    .cancelled()
                    .map_err(|_| std::io::Error::other("cancelled"))?;
                match next {
                    Some(Ok(bytes)) => {
                        shared.state.lock().unwrap().stats.received_bytes += bytes.len() as u64;
                        shared.emit(None, None);
                        Ok(Some((bytes, body)))
                    }
                    Some(Err(_)) => Err(std::io::Error::other("range body failed")),
                    None => Ok(None),
                }
            }
        });
        let shared = self.shared.clone();
        let expected_digest = (relative == 0 && size == extent.length).then_some(extent.sha256);
        let range = tokio::task::spawn_blocking(move || -> Result<ReadyRange> {
            use sha2::{Digest, Sha256};
            use std::io::Read;
            use std::os::unix::fs::FileExt;
            let mut input = tokio_util::io::SyncIoBridge::new(tokio_util::io::StreamReader::new(
                Box::pin(body),
            ));
            let mut buffer = AVec::<u8, ConstAlign<ALIGN>>::new(ALIGN);
            buffer.resize(shared.limits.download_buffer_bytes, 0);
            let mut received = 0_u64;
            let mut hash = Sha256::new();
            while received < size {
                shared.cancelled()?;
                let count = ((size - received) as usize).min(buffer.len());
                input
                    .read_exact(&mut buffer[..count])
                    .map_err(|_| Error::Transfer)?;
                hash.update(&buffer[..count]);
                let padded = count.div_ceil(ALIGN) * ALIGN;
                buffer[count..padded].fill(0);
                let began = Instant::now();
                range
                    .backing
                    .file
                    .write_all_at(&buffer[..padded], received)?;
                if matches!(range.location(), Location::Spool) {
                    shared.state.lock().unwrap().stats.spool_write_seconds +=
                        began.elapsed().as_secs_f64();
                }
                received += count as u64;
            }
            if input.read(&mut [0])? != 0 {
                return Err(Error::Integrity(
                    "range body exceeds declared boundary".into(),
                ));
            }
            if expected_digest.is_some_and(|expected| format!("{:x}", hash.finalize()) != expected)
            {
                return Err(Error::Integrity("range chunk digest".into()));
            }
            shared.cancelled()?;
            Ok(range)
        })
        .await
        .map_err(|_| Error::Worker)??;
        self.shared
            .emit(Some(range.spec.id), Some(Lifecycle::Ready));
        self.shared.state.lock().unwrap().ready.push_back(range);
        self.shared.wake.notify_waiters();
        Ok(())
    }
}
