//! Native paged attention kernels. The public API accepts borrowed immutable
//! tensors; device pointers remain private to the audited launch module.
use candle_core::{Result, Tensor};

#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
mod cuda;

/// One sequence's compact, ordered KV pages. Every non-final page is full.
pub struct PagedRow<'a> {
    pub pages: Vec<(&'a Tensor, &'a Tensor)>,
    pub len: usize,
    pub page_tokens: usize,
}

/// Decode one query token per sequence directly against immutable KV pages.
/// Unsupported devices, dtypes or head dimensions return `None`, allowing a
/// portable implementation to handle them. Invalid page structure is an error.
pub fn paged_decode(
    query: &Tensor,
    rows: &[PagedRow<'_>],
    num_kv_groups: usize,
) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    {
        cuda::attention(query, rows, num_kv_groups)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (query, rows, num_kv_groups);
        Ok(None)
    }
}
