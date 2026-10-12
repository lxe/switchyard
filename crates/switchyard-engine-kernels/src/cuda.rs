//! The only unsafe operation is launching the compiled kernel. Validation and
//! storage borrows establish its pointer, shape, dtype and lifetime invariants.
use super::PagedRow;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{
    DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use candle_core::cuda_backend::{CudaDType, WrapErr};
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape, Storage, Tensor,
};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/paged_decode.ptx"));
const MAX_HEAD_DIM: usize = 256;

struct Decode {
    pages: Vec<(Tensor, Tensor)>,
    offsets: Vec<u32>,
    lengths: Vec<u32>,
    widths: Vec<u32>,
    heads: usize,
    groups: usize,
    dim: usize,
    max_len: usize,
}

fn as_u32(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg("paged decode dimension exceeds u32".into()))
}

pub(super) fn attention(
    query: &Tensor,
    rows: &[PagedRow<'_>],
    groups: usize,
) -> Result<Option<Tensor>> {
    if !query.device().is_cuda() || !matches!(query.dtype(), DType::F32 | DType::BF16) {
        return Ok(None);
    }
    let (batch, heads, query_len, dim) = query.dims4()?;
    if query_len != 1 || dim == 0 || dim > MAX_HEAD_DIM {
        return Ok(None);
    }
    if batch == 0 || rows.len() != batch || groups == 0 || !heads.is_multiple_of(groups) {
        candle_core::bail!("invalid paged decode batch or grouped-query heads");
    }
    let mut op = Decode {
        pages: Vec::new(),
        offsets: Vec::new(),
        lengths: Vec::new(),
        widths: Vec::new(),
        heads,
        groups,
        dim,
        max_len: 0,
    };
    for (row_index, row) in rows.iter().enumerate() {
        if row.len == 0 || row.page_tokens == 0 || row.pages.is_empty() {
            candle_core::bail!("paged decode requires nonempty sequences and pages");
        }
        op.offsets.push(as_u32(op.pages.len())?);
        op.lengths.push(as_u32(row.len)?);
        op.widths.push(as_u32(row.page_tokens)?);
        op.max_len = op.max_len.max(row.len);
        let mut total = 0usize;
        for (index, (key, value)) in row.pages.iter().enumerate() {
            let (one, kv_heads, live, page_dim) = key.dims4()?;
            if one != 1
                || kv_heads != heads / groups
                || page_dim != dim
                || live == 0
                || live > row.page_tokens
                || (index + 1 != row.pages.len() && live != row.page_tokens)
                || value.dims() != key.dims()
                || key.dtype() != query.dtype()
                || value.dtype() != query.dtype()
                || !key.device().same_device(query.device())
                || !value.device().same_device(query.device())
                || !key.is_contiguous()
                || !value.is_contiguous()
            {
                candle_core::bail!(
                    "invalid compact KV page: row={row_index} page={index}/{} key_dims={:?} value_dims={:?} key_dtype={:?} value_dtype={:?} query_dtype={:?} key_strides={:?} value_strides={:?} key_contiguous={} value_contiguous={} key_same_device={} value_same_device={} expected_kv_heads={} query_heads={heads} groups={groups} head_dim={dim} page_tokens={} sequence_len={}",
                    row.pages.len(),
                    key.dims(),
                    value.dims(),
                    key.dtype(),
                    value.dtype(),
                    query.dtype(),
                    key.stride(),
                    value.stride(),
                    key.is_contiguous(),
                    value.is_contiguous(),
                    key.device().same_device(query.device()),
                    value.device().same_device(query.device()),
                    heads / groups,
                    row.page_tokens,
                    row.len,
                );
            }
            total = total
                .checked_add(live)
                .ok_or_else(|| candle_core::Error::Msg("KV length overflow".into()))?;
            op.pages.push(((*key).clone(), (*value).clone()));
        }
        if total != row.len {
            candle_core::bail!("paged decode sequence length differs from page contents");
        }
    }
    // Kernel scalar dimensions are signed ints; prevent indexing overflow too.
    let scores = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(op.max_len));
    let outputs = batch.checked_mul(heads).and_then(|n| n.checked_mul(dim));
    let key_index = (heads / groups)
        .checked_mul(op.max_len)
        .and_then(|n| n.checked_mul(dim));
    if key_index.is_none_or(|n| n > i32::MAX as usize)
        || scores.is_none_or(|n| n > i32::MAX as usize)
        || outputs.is_none_or(|n| n > i32::MAX as usize)
        || row_count_exceeds_grid(batch, heads)
        || op.pages.len() > i32::MAX as usize / 3
        || op.max_len > i32::MAX as usize
        || row_max_width(rows) > i32::MAX as usize
    {
        candle_core::bail!("paged decode exceeds kernel indexing limits");
    }
    Ok(Some(query.contiguous()?.apply_op1_no_bwd(&op)?))
}

fn row_count_exceeds_grid(batch: usize, heads: usize) -> bool {
    batch > i32::MAX as usize || heads > 65535
}
fn row_max_width(rows: &[PagedRow<'_>]) -> usize {
    rows.iter().map(|r| r.page_tokens).max().unwrap_or(0)
}

impl CustomOp1 for Decode {
    fn name(&self) -> &'static str {
        "switchyard-paged-decode"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged CUDA decode cannot execute on CPU")
    }
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        match storage.dtype() {
            DType::F32 => self.launch::<f32>(storage, layout, "paged_decode_f32"),
            DType::BF16 => self.launch::<half::bf16>(storage, layout, "paged_decode_bf16"),
            _ => candle_core::bail!("unsupported paged CUDA dtype"),
        }
    }
}

impl Decode {
    fn launch<T: CudaDType + DeviceRepr + ValidAsZeroBits>(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
        kernel: &str,
    ) -> Result<(CudaStorage, Shape)> {
        let device = storage.device();
        let stream = device.cuda_stream();
        // Hold read locks for every source allocation through launch. Each raw
        // pointer guard schedules a read-completion event when dropped below.
        // Cudarc then orders frees/reuse after those reads, without a GPU fence.
        let storages = self
            .pages
            .iter()
            .flat_map(|(k, v)| [k.storage_and_layout(), v.storage_and_layout()])
            .collect::<Vec<_>>();
        let mut pointer_guards = Vec::with_capacity(storages.len());
        let mut addresses = Vec::with_capacity(self.pages.len() * 3);
        for (index, pair) in storages.as_chunks::<2>().0.iter().enumerate() {
            for (storage, layout) in pair {
                let Storage::Cuda(storage) = &**storage else {
                    candle_core::bail!("KV page changed device");
                };
                let (begin, end) = layout
                    .contiguous_offsets()
                    .ok_or_else(|| candle_core::Error::Msg("KV page is not contiguous".into()))?;
                let slice = storage.as_cuda_slice::<T>()?;
                if end > slice.len() {
                    candle_core::bail!("KV page storage shorter than layout");
                }
                let (pointer, guard) = slice.device_ptr(&stream);
                addresses.push(pointer + (begin * std::mem::size_of::<T>()) as u64);
                pointer_guards.push(guard);
            }
            addresses.push(self.pages[index].0.dim(2)? as u64);
        }
        let pages = device.clone_htod(&addresses)?;
        let offsets = device.clone_htod(&self.offsets)?;
        let lengths = device.clone_htod(&self.lengths)?;
        let widths = device.clone_htod(&self.widths)?;
        let batch = self.lengths.len();
        let mut scores = device.alloc_zeros::<f32>(batch * self.heads * self.max_len)?;
        let mut output = device.alloc_zeros::<T>(batch * self.heads * self.dim)?;
        let (begin, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("query is not contiguous".into()))?;
        let source = storage.as_cuda_slice::<T>()?.slice(begin..end);
        let function = device.get_or_load_custom_func(kernel, "switchyard_paged_decode_v1", PTX)?;
        let config = LaunchConfig {
            grid_dim: (batch as u32, self.heads as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = function.builder();
        builder
            .arg(&source)
            .arg(&pages)
            .arg(&offsets)
            .arg(&lengths)
            .arg(&widths)
            .arg(&mut scores)
            .arg(&mut output);
        let heads = self.heads as i32;
        let groups = self.groups as i32;
        let dim = self.dim as i32;
        let max_len = self.max_len as i32;
        builder.arg(&heads).arg(&groups).arg(&dim).arg(&max_len);
        // SAFETY: The embedded kernel signature matches these typed arguments.
        // Shape/device/dtype/contiguity and signed indexing bounds are validated
        // above. Source locks and guards outlive launch; all launches, parameter
        // copies, output ownership and page read events use Candle's same stream.
        unsafe { builder.launch(config) }.w()?;
        drop(pointer_guards);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((batch, self.heads, 1, self.dim)),
        ))
    }
}
