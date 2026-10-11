//! Exact HTTP objects/ranges feed bounded positional writers without retries.
use super::ExtentHydration;
use crate::{Error, HydrationMode, Result, extents::Extent};
use futures_util::{StreamExt, stream};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, sync::atomic::Ordering};
use tokio_util::io::{StreamReader, SyncIoBridge};

impl ExtentHydration {
    pub(super) async fn transfer(
        &self,
        client: &reqwest::Client,
        index: usize,
        extent: Extent,
    ) -> Result<()> {
        let shared = self.shared.clone();
        let permit = match &self.options.download_limit {
            Some(limit) => Some(tokio::select! {
                permit = limit.clone().acquire_owned() => permit.map_err(|_| Error::Cancelled)?,
                _ = shared.wait_cancelled() => return Err(Error::Cancelled),
            }),
            None => None,
        };
        if shared.cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let ranged = extent.object_offset != 0 || extent.length != extent.object_size;
        let mut request = client
            .get(&self.ticket.urls[&extent.object])
            .header("Accept-Encoding", "identity");
        if ranged {
            request = request.header(
                "Range",
                format!(
                    "bytes={}-{}",
                    extent.object_offset,
                    extent.object_offset + extent.length - 1
                ),
            );
        }
        let sending = request.send();
        tokio::pin!(sending);
        let response = loop {
            let notified = shared.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if shared.cancelled.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            tokio::select! {
                result = &mut sending => break result.map_err(|_| Error::Transfer)?,
                _ = &mut notified => {},
            }
        };
        let expected_status = if ranged {
            reqwest::StatusCode::PARTIAL_CONTENT
        } else {
            reqwest::StatusCode::OK
        };
        if response.status() != expected_status {
            return Err(Error::Http(response.status().as_u16()));
        }
        if response
            .headers()
            .get("content-encoding")
            .is_some_and(|h| h != "identity")
            || response
                .content_length()
                .is_some_and(|length| length != extent.length)
        {
            return Err(Error::Integrity("extent HTTP length or encoding".into()));
        }
        if ranged {
            let expected = format!(
                "bytes {}-{}/{}",
                extent.object_offset,
                extent.object_offset + extent.length - 1,
                extent.object_size
            );
            if response
                .headers()
                .get("content-range")
                .and_then(|h| h.to_str().ok())
                != Some(expected.as_str())
            {
                return Err(Error::Integrity("extent HTTP content-range".into()));
            }
        }
        let body_shared = shared.clone();
        let body = stream::try_unfold(response.bytes_stream(), move |mut body| {
            let shared = body_shared.clone();
            async move {
                loop {
                    let notified = shared.wake.notified();
                    tokio::pin!(notified);
                    notified.as_mut().enable();
                    if shared.cancelled.load(Ordering::Acquire) {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::ConnectionAborted,
                            "cancelled",
                        ));
                    }
                    tokio::select! {
                        next = body.next() => return match next {
                            Some(Ok(bytes)) => { shared.received.fetch_add(bytes.len() as u64, Ordering::Relaxed); Ok(Some((bytes, body))) },
                            Some(Err(_)) => Err(std::io::Error::other("extent transfer failed")),
                            None => Ok(None),
                        },
                        _ = &mut notified => {},
                    }
                }
            }
        });
        let buffer_bytes = self.options.buffer_bytes;
        let verified = self.options.mode == HydrationMode::Verified;
        tokio::task::spawn_blocking(move || -> Result<()> {
            // Permit and owner survive async cancellation until the blocking writer exits.
            let _permit = permit;
            let mut reader = SyncIoBridge::new(StreamReader::new(Box::pin(body)));
            let file = File::options().write(true).open(
                shared
                    .staging
                    .join(shared.manifest.entries[extent.entry].path()),
            )?;
            let mut buffer = vec![0_u8; buffer_bytes];
            let mut hash = verified.then(Sha256::new);
            let mut written = 0;
            loop {
                if shared.cancelled.load(Ordering::Acquire) {
                    return Err(Error::Cancelled);
                }
                let count = reader.read(&mut buffer).map_err(|_| Error::Transfer)?;
                if count == 0 {
                    break;
                }
                if written + count as u64 > extent.length {
                    return Err(Error::Integrity("extent body exceeds boundary".into()));
                }
                write_at(&file, &buffer[..count], extent.offset + written)?;
                if let Some(hash) = &mut hash {
                    hash.update(&buffer[..count]);
                }
                written += count as u64;
            }
            if written != extent.length {
                return Err(Error::Integrity("truncated extent body".into()));
            }
            if let Some(hash) = hash
                && format!("{:x}", hash.finalize()) != extent.sha256
            {
                return Err(Error::Integrity("extent digest".into()));
            }
            if shared.cancelled.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            // Completion is published only after exact EOF, optional digest, and all writes.
            shared.state.lock().unwrap().done[index] = true;
            shared.wake.notify_waiters();
            Ok(())
        })
        .await
        .map_err(|_| Error::Worker)?
    }
}

#[cfg(unix)]
fn write_at(file: &File, bytes: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(bytes, offset)
}
#[cfg(not(unix))]
fn write_at(_file: &File, _bytes: &[u8], _offset: u64) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "extent writes require Unix positional IO",
    ))
}
