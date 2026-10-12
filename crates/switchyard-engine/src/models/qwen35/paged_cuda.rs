//! Safe adapter from model-owned immutable pages to the CUDA launch boundary.
use super::paged::PagedKv;
use candle_core::{Result, Tensor};
use switchyard_engine_kernels::{PagedRow, paged_decode};

pub(super) fn attention(
    query: &Tensor,
    rows: &[PagedKv],
    num_kv_groups: usize,
) -> Result<Option<Tensor>> {
    let pages = rows
        .iter()
        .map(|row| PagedRow {
            pages: row
                .pages
                .iter()
                .map(|page| (&page.key, &page.value))
                .collect(),
            len: row.len,
            page_tokens: row.page_tokens,
        })
        .collect::<Vec<_>>();
    paged_decode(query, &pages, num_kv_groups)
}
