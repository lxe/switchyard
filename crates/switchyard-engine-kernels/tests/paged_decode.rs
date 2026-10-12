#[cfg(feature = "cuda")]
use candle_core::D;
use candle_core::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use switchyard_engine_kernels::PagedRow;
use switchyard_engine_kernels::paged_decode;

#[test]
fn cpu_uses_portable_attention() -> Result<()> {
    let query = Tensor::zeros((1, 2, 1, 16), DType::F32, &Device::Cpu)?;
    assert!(paged_decode(&query, &[], 1)?.is_none());
    Ok(())
}

#[cfg(feature = "cuda")]
fn deterministic(
    shape: (usize, usize, usize, usize),
    phase: f32,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let size = shape.0 * shape.1 * shape.2 * shape.3;
    let values = (0..size)
        .map(|i| ((i as f32 * 0.071) + phase).sin() * 0.6)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
}

#[cfg(feature = "cuda")]
fn repeat_heads(t: &Tensor, groups: usize) -> Result<Tensor> {
    let (batch, heads, len, dim) = t.dims4()?;
    t.unsqueeze(2)?
        .expand((batch, heads, groups, len, dim))?
        .reshape((batch, heads * groups, len, dim))?
        .contiguous()
}

#[cfg(feature = "cuda")]
fn parity(dtype: DType, dim: usize) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let lengths = [1, 17, 31, 65];
    let page_tokens = 16;
    let heads = 4;
    let groups = 2;
    let query = deterministic((lengths.len(), heads, 1, dim), 0.23, dtype, &device)?;
    let dense = lengths
        .iter()
        .map(|&len| {
            Ok((
                deterministic((1, heads / groups, len, dim), 0.15, dtype, &device)?,
                deterministic((1, heads / groups, len, dim), 1.2, dtype, &device)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let pages = dense
        .iter()
        .map(|(key, value)| {
            (0..key.dim(2)?)
                .step_by(page_tokens)
                .map(|offset| {
                    let live = page_tokens.min(key.dim(2)? - offset);
                    Ok((
                        key.narrow(2, offset, live)?.force_contiguous()?,
                        value.narrow(2, offset, live)?.force_contiguous()?,
                    ))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let rows = pages
        .iter()
        .zip(lengths)
        .map(|(pages, len)| PagedRow {
            pages: pages.iter().map(|(k, v)| (k, v)).collect(),
            len,
            page_tokens,
        })
        .collect::<Vec<_>>();
    let actual = paged_decode(&query, &rows, groups)?.expect("CUDA kernel selected");
    let expected = dense
        .iter()
        .enumerate()
        .map(|(row, (key, value))| {
            let q = query
                .narrow(0, row, 1)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let key = repeat_heads(key, groups)?.to_dtype(DType::F32)?;
            let value = repeat_heads(value, groups)?.to_dtype(DType::F32)?;
            let scores = (q.matmul(&key.transpose(2, 3)?)? * (1.0 / (dim as f64).sqrt()))?;
            let exponentials = scores
                .broadcast_sub(&scores.max_keepdim(D::Minus1)?)?
                .exp()?;
            exponentials
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?
                .matmul(&value)?
                .broadcast_div(&exponentials.sum_keepdim(D::Minus1)?)?
                .to_dtype(dtype)
        })
        .collect::<Result<Vec<_>>>()?;
    let expected = Tensor::cat(&expected, 0)?;
    let differences = (&actual.to_dtype(DType::F32)? - &expected.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_error = differences.iter().copied().fold(0.0f32, f32::max);
    let tolerance = if dtype == DType::F32 { 2e-5 } else { 0.003 };
    eprintln!("paged CUDA {dtype:?} dim={dim} max_error={max_error}");
    assert!(
        max_error <= tolerance,
        "paged attention error {max_error} exceeds {tolerance}"
    );
    // Reusing identical immutable pages must preserve the answer after original
    // page owners are dropped and allocator activity has occurred.
    drop(dense);
    let second = paged_decode(&query, &rows, groups)?.expect("CUDA kernel selected");
    assert_eq!(
        actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        second
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires exclusive CUDA test slot"]
fn cuda_compact_pages_match_dense_f32_and_bf16() -> Result<()> {
    for dtype in [DType::F32, DType::BF16] {
        for dim in [128, 256] {
            parity(dtype, dim)?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn dense_attention(query: &Tensor, key: &Tensor, value: &Tensor, groups: usize) -> Result<Tensor> {
    let dtype = query.dtype();
    let dim = query.dim(3)?;
    let key = repeat_heads(key, groups)?.to_dtype(DType::F32)?;
    let value = repeat_heads(value, groups)?.to_dtype(DType::F32)?;
    let scores = (query
        .to_dtype(DType::F32)?
        .contiguous()?
        .matmul(&key.transpose(2, 3)?)?
        * (1.0 / (dim as f64).sqrt()))?;
    let exponentials = scores
        .broadcast_sub(&scores.max_keepdim(D::Minus1)?)?
        .exp()?;
    exponentials
        .to_dtype(dtype)?
        .to_dtype(DType::F32)?
        .matmul(&value)?
        .broadcast_div(&exponentials.sum_keepdim(D::Minus1)?)?
        .to_dtype(dtype)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires exclusive CUDA test slot"]
fn cuda_paged_decode_microbenchmark() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let iterations = 100;
    let heads = 8;
    let groups = 4;
    let dim = 128;
    let page_tokens = 16;
    for batch in [1, 8] {
        for len in [128, 1024] {
            let query = deterministic((batch, heads, 1, dim), 0.2, DType::BF16, &device)?;
            let key = deterministic((batch, heads / groups, len, dim), 0.5, DType::BF16, &device)?;
            let value =
                deterministic((batch, heads / groups, len, dim), 1.5, DType::BF16, &device)?;
            let pages = (0..batch)
                .map(|row| {
                    (0..len)
                        .step_by(page_tokens)
                        .map(|offset| {
                            Ok((
                                key.narrow(0, row, 1)?
                                    .narrow(2, offset, page_tokens)?
                                    .force_contiguous()?,
                                value
                                    .narrow(0, row, 1)?
                                    .narrow(2, offset, page_tokens)?
                                    .force_contiguous()?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            let rows = pages
                .iter()
                .map(|pages| PagedRow {
                    pages: pages.iter().map(|(k, v)| (k, v)).collect(),
                    len,
                    page_tokens,
                })
                .collect::<Vec<_>>();
            for _ in 0..3 {
                let _ = paged_decode(&query, &rows, groups)?;
                let _ = dense_attention(&query, &key, &value, groups)?;
            }
            device.synchronize()?;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = paged_decode(&query, &rows, groups)?;
            }
            device.synchronize()?;
            let paged_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = dense_attention(&query, &key, &value, groups)?;
            }
            device.synchronize()?;
            let dense_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
            eprintln!(
                "paged_decode_benchmark batch={batch} len={len} heads={heads} dim={dim} dtype=BF16 iterations={iterations} paged_ms={paged_ms:.6} dense_ms={dense_ms:.6} ratio={:.3}",
                paged_ms / dense_ms
            );
        }
    }
    Ok(())
}
