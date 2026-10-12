//! Immutable per-sequence KV pages. Only a partial tail is copied on append;
//! full pages remain shared by restored prefixes and independent continuations.
use super::repeat_interleave;
use candle_core::{D, DType, Result, Tensor};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct KvPage {
    pub(super) key: Tensor,
    pub(super) value: Tensor,
}

#[derive(Debug, Clone)]
pub(super) struct PagedKv {
    pub(super) pages: Vec<Arc<KvPage>>,
    pub(super) len: usize,
    pub(super) page_tokens: usize,
}

impl PagedKv {
    pub(super) fn new(page_tokens: usize) -> Self {
        Self {
            pages: Vec::new(),
            len: 0,
            page_tokens,
        }
    }

    /// Inputs have shape (1, kv_heads, new_tokens, head_dim). Prepare replacement
    /// pages before publishing them so a failed allocation leaves the prefix intact.
    pub(super) fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<()> {
        let (rows, heads, length, dim) = key.dims4()?;
        if rows != 1 || length == 0 || key.dims() != value.dims() || key.dtype() != value.dtype() {
            candle_core::bail!("Invalid KV page append");
        }
        if let Some(first) = self.pages.first()
            && (first.key.dim(1)? != heads
                || first.key.dim(3)? != dim
                || first.key.dtype() != key.dtype())
        {
            candle_core::bail!("Incompatible KV page append");
        }
        let mut pages = self.pages.clone();
        let mut offset = 0;
        if let Some(tail) = pages.last() {
            let live = tail.key.dim(2)?;
            if live < self.page_tokens {
                let count = length.min(self.page_tokens - live);
                // Candle's fallback concatenation can return a transposed view
                // for strided rotary keys. The CUDA page contract requires
                // compact storage, including the replacement partial tail.
                let key = Tensor::cat(&[&tail.key, &key.narrow(2, 0, count)?], 2)?.contiguous()?;
                let value =
                    Tensor::cat(&[&tail.value, &value.narrow(2, 0, count)?], 2)?.contiguous()?;
                *pages.last_mut().unwrap() = Arc::new(KvPage { key, value });
                offset = count;
            }
        }
        while offset < length {
            let count = self.page_tokens.min(length - offset);
            // Force compact storage even for contiguous narrow views, which
            // would otherwise keep the backing storage of an entire cohort.
            let key = key.narrow(2, offset, count)?.force_contiguous()?;
            let value = value.narrow(2, offset, count)?.force_contiguous()?;
            pages.push(Arc::new(KvPage { key, value }));
            offset += count;
        }
        self.pages = pages;
        self.len += length;
        Ok(())
    }

    pub(super) fn resident_bytes(&self) -> usize {
        self.pages.iter().fold(0usize, |bytes, page| {
            bytes
                .saturating_add(
                    page.key
                        .elem_count()
                        .saturating_mul(page.key.dtype().size_in_bytes()),
                )
                .saturating_add(
                    page.value
                        .elem_count()
                        .saturating_mul(page.value.dtype().size_in_bytes()),
                )
        })
    }
}

/// Tensor implementation of attention over page tables. K/V history is never
/// concatenated or padded. Score pages share one global F32 softmax maximum and
/// denominator; exponentials are rounded to KV dtype before their value product.
/// The per-row/page tensor dispatches are a correctness fallback, not a fused
/// GPU attention kernel. Temporary scores scale with query length times logical
/// history length; cached keys and values remain in their original pages.
pub(super) fn attention(
    query: &Tensor,
    rows: &[PagedKv],
    groups: usize,
    mask: Option<&Tensor>,
    scale: f64,
    output_dtype: DType,
) -> Result<Tensor> {
    if query.dim(0)? != rows.len() || rows.is_empty() {
        candle_core::bail!("Attention page table row mismatch");
    }
    let mut outputs = Vec::with_capacity(rows.len());
    for (row, cache) in rows.iter().enumerate() {
        let q = query
            .narrow(0, row, 1)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let mut score_pages = Vec::with_capacity(cache.pages.len());
        let mut global_max: Option<Tensor> = None;
        let mut offset = 0;
        for page in &cache.pages {
            let key = repeat_interleave(&page.key, groups, 1)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let mut scores = (q.matmul(&key.transpose(2, 3)?)? * scale)?;
            if let Some(mask) = mask {
                scores = scores.broadcast_add(
                    &mask
                        .narrow(0, row, 1)?
                        .narrow(3, offset, page.key.dim(2)?)?
                        .to_dtype(DType::F32)?,
                )?;
            }
            let max = scores.max_keepdim(D::Minus1)?;
            global_max = Some(match global_max {
                None => max,
                Some(previous) => previous.maximum(&max)?,
            });
            offset += page.key.dim(2)?;
            score_pages.push(scores);
        }
        let max = global_max
            .ok_or_else(|| candle_core::Error::Msg("Empty attention page table".into()))?;
        let mut denominator: Option<Tensor> = None;
        let mut numerator: Option<Tensor> = None;
        for (scores, page) in score_pages.iter().zip(&cache.pages) {
            let exponentials = scores.broadcast_sub(&max)?.exp()?;
            let sum = exponentials.sum_keepdim(D::Minus1)?;
            denominator = Some(match denominator {
                None => sum,
                Some(previous) => (previous + sum)?,
            });
            let values = repeat_interleave(&page.value, groups, 1)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let product = exponentials
                .to_dtype(page.value.dtype())?
                .to_dtype(DType::F32)?
                .matmul(&values)?;
            numerator = Some(match numerator {
                None => product,
                Some(previous) => (previous + product)?,
            });
        }
        outputs.push(
            numerator
                .unwrap()
                .broadcast_div(&denominator.unwrap())?
                .to_dtype(output_dtype)?,
        );
    }
    Tensor::cat(&outputs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    fn tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), (1, 1, values.len(), 1), &Device::Cpu).unwrap()
    }
    #[test]
    fn immutable_pages_share_full_prefix_and_copy_partial_tail() -> Result<()> {
        let mut original = PagedKv::new(2);
        original.append(&tensor(&[1., 2., 3.]), &tensor(&[4., 5., 6.]))?;
        let mut fork = original.clone();
        fork.append(&tensor(&[8., 9.]), &tensor(&[10., 11.]))?;
        assert!(Arc::ptr_eq(&original.pages[0], &fork.pages[0]));
        assert!(!Arc::ptr_eq(&original.pages[1], &fork.pages[1]));
        assert_eq!(original.len, 3);
        assert_eq!(fork.len, 5);
        assert_eq!(original.pages[1].key.flatten_all()?.to_vec1::<f32>()?, [3.]);
        assert_eq!(fork.pages[1].key.flatten_all()?.to_vec1::<f32>()?, [3., 8.]);
        assert_eq!(fork.resident_bytes(), 5 * 2 * 4);
        Ok(())
    }
    #[test]
    fn strided_rotary_tail_append_publishes_compact_pages() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mut cache = PagedKv::new(16);
        let initial = Tensor::zeros((1, 2, 3, 4), DType::F32, &device)?;
        cache.append(&initial, &initial)?;
        let strided = Tensor::from_vec(
            (0..16).map(|n| n as f32).collect::<Vec<_>>(),
            (1, 2, 4, 2),
            &device,
        )?
        .transpose(2, 3)?
        .narrow(2, 0, 1)?;
        assert!(!strided.is_contiguous());
        cache.append(&strided, &strided.contiguous()?)?;
        assert!(cache.pages[0].key.is_contiguous());
        assert!(cache.pages[0].value.is_contiguous());
        assert_eq!(cache.pages[0].key.dim(2)?, 4);
        Ok(())
    }
    #[test]
    fn softmax_uses_global_maximum_across_pages() -> Result<()> {
        let mut cache = PagedKv::new(2);
        cache.append(
            &tensor(&[-10., 1., 20., 2., -5.]),
            &tensor(&[7., 8., 9., 10., 11.]),
        )?;
        let q = tensor(&[1.]);
        let actual = attention(&q, &[cache], 1, None, 1., DType::F32)?;
        let k = tensor(&[-10., 1., 20., 2., -5.]);
        let v = tensor(&[7., 8., 9., 10., 11.]);
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let probabilities = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let expected = probabilities.matmul(&v)?;
        assert!((&actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
        Ok(())
    }
    #[test]
    fn bf16_exponentials_use_dense_global_rounding_with_causal_suffix() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let key = Tensor::from_vec(
            (0..42).map(|i| (i as f32 * 0.31).sin()).collect::<Vec<_>>(),
            (1, 1, 7, 6),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let value = Tensor::from_vec(
            (0..42).map(|i| (i as f32 * 0.19).cos()).collect::<Vec<_>>(),
            (1, 1, 7, 6),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let query = Tensor::from_vec(
            (0..36).map(|i| (i as f32 * 0.27).cos()).collect::<Vec<_>>(),
            (1, 2, 3, 6),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let mask = Tensor::from_vec(
            (0..3)
                .flat_map(|row| {
                    (0..7).map(move |column| {
                        if column > 4 + row {
                            f32::NEG_INFINITY
                        } else {
                            0.
                        }
                    })
                })
                .collect::<Vec<_>>(),
            (1, 1, 3, 7),
            &device,
        )?;
        let k = repeat_interleave(&key, 2, 1)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let v = repeat_interleave(&value, 2, 1)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let scores = (query.to_dtype(DType::F32)?.matmul(&k.transpose(2, 3)?)? * 0.4)?
            .broadcast_add(&mask)?;
        let exps = scores
            .broadcast_sub(&scores.max_keepdim(D::Minus1)?)?
            .exp()?;
        let expected = exps
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?
            .matmul(&v)?
            .broadcast_div(&exps.sum_keepdim(D::Minus1)?)?;
        for page_tokens in [1, 2, 3, 7] {
            let mut cache = PagedKv::new(page_tokens);
            cache.append(&key, &value)?;
            let actual = attention(&query, &[cache], 2, Some(&mask), 0.4, DType::F32)?;
            assert!(
                (&actual - &expected)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?
                    <= 1e-6
            );
        }
        Ok(())
    }
}
