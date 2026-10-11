//! Explicit host, download, and CUDA staging budgets.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
const ALIGN: usize = 4096;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferLimits {
    pub chunk_bytes: usize,
    pub ram_bytes: u64,
    pub spool_bytes: u64,
    pub download_concurrency: usize,
    pub download_buffer_bytes: usize,
    pub pinned_bytes: usize,
    pub transfer_concurrency: usize,
}
impl Default for BufferLimits {
    fn default() -> Self {
        Self {
            chunk_bytes: 4 << 20,
            ram_bytes: 64 << 20,
            spool_bytes: 4 << 30,
            download_concurrency: 16,
            download_buffer_bytes: 64 << 10,
            pinned_bytes: 20 << 20,
            transfer_concurrency: 4,
        }
    }
}
impl BufferLimits {
    pub fn validate(&self) -> Result<()> {
        if self.chunk_bytes == 0
            || self.chunk_bytes > 64 << 20
            || self.pinned_bytes > 4 << 30
            || !self.chunk_bytes.is_multiple_of(ALIGN)
            || self.download_buffer_bytes < ALIGN
            || !self.download_buffer_bytes.is_multiple_of(ALIGN)
            || self.download_buffer_bytes > self.chunk_bytes
            || self.ram_bytes < self.chunk_bytes as u64
            || self.spool_bytes < self.chunk_bytes as u64
            || !(1..=128).contains(&self.download_concurrency)
            || !(1..=64).contains(&self.transfer_concurrency)
            || self.pinned_bytes / self.transfer_concurrency < self.chunk_bytes + ALIGN
        {
            return Err(Error::Invalid("bounded range buffer limits".into()));
        }
        Ok(())
    }
}
