//! Qwen3.5 image encoder/projector. Adapted from Candle Qwen3-VL vision implementation,
//! commit c68b24997319b8d76e5a4e775dab197c7f26322a (Apache-2.0 OR MIT).
//! CPU preprocessing uses the released Qwen3.5 channel/temporal/merge patch order.
use std::f64;

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Activation, Embedding, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, embedding,
    layer_norm, linear,
};

/// Released Qwen3.5 vision encoder and projector dimensions.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
    pub hidden_act: Activation,
    pub intermediate_size: usize,
    pub num_heads: usize,
    #[serde(alias = "in_chans")]
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

struct PatchEmbed {
    proj: Linear,
}
impl PatchEmbed {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let width = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size.pow(2);
        let weight = vb
            .get(
                (
                    cfg.hidden_size,
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                ),
                "proj.weight",
            )?
            .reshape((cfg.hidden_size, width))?;
        let bias = vb.get(cfg.hidden_size, "proj.bias")?;
        Ok(Self {
            proj: Linear::new(weight, Some(bias)),
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.proj.forward(xs)
    }
}

struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl VisionMlp {
    fn new(dim: usize, hidden_dim: usize, act: Activation, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(dim, hidden_dim, vb.pp("linear_fc1"))?,
            fc2: linear(hidden_dim, dim, vb.pp("linear_fc2"))?,
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs.apply(&self.act)?;
        self.fc2.forward(&xs)
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;

    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: linear(dim, dim * 3, vb.pp("qkv"))?,
            proj: linear(dim, dim, vb.pp("proj"))?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let hidden_states = self.qkv.forward(xs)?;
        let qkv = hidden_states
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;
        let mut q = qkv.i(0)?.squeeze(0)?;
        let mut k = qkv.i(1)?.squeeze(0)?;
        let mut v = qkv.i(2)?.squeeze(0)?;

        let cos = cos.to_dtype(DType::F32)?;
        let sin = sin.to_dtype(DType::F32)?;
        q = q.to_dtype(DType::F32)?;
        k = k.to_dtype(DType::F32)?;
        v = v.to_dtype(DType::F32)?;
        (q, k) = apply_rotary_pos_emb_vision(&q, &k, &cos, &sin)?;

        let mut outputs = Vec::new();
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            if end <= start {
                continue;
            }
            let len = end - start;
            let q_chunk = q.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let k_chunk = k.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;

            let mut chunk_out = {
                let q = q_chunk.unsqueeze(0)?;
                let k = k_chunk.unsqueeze(0)?;
                let v = v_chunk.unsqueeze(0)?;

                let attn_weights =
                    (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;

                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&v)?
            };
            chunk_out = chunk_out.squeeze(0)?.transpose(0, 1)?;

            chunk_out = chunk_out.reshape((len, self.num_heads * self.head_dim))?;
            outputs.push(chunk_out.to_dtype(xs.dtype())?);
        }
        let attn_output = Tensor::cat(&outputs, 0)?;
        self.proj.forward(&attn_output)
    }
}

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm1 = layer_norm(cfg.hidden_size, norm_cfg, vb.pp("norm1"))?;
        let norm2 = layer_norm(cfg.hidden_size, norm_cfg, vb.pp("norm2"))?;
        let attn = VisionAttention::new(cfg.hidden_size, cfg.num_heads, vb.pp("attn"))?;
        let mlp = VisionMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_act,
            vb.pp("mlp"),
        )?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.norm1.forward(xs)?;
        let attn_out = self.attn.forward(&normed, cu_seqlens, cos, sin)?;
        let xs_att = xs.add(&attn_out)?;
        let mlp_out = self.mlp.forward(&self.norm2.forward(&xs_att)?)?;
        xs_att.add(&mlp_out)
    }
}

struct PatchMerger {
    norm: LayerNorm,
    use_postshuffle_norm: bool,
    spatial_merge_unit: usize,
    merged_hidden_size: usize,
    fc1: Linear,
    fc2: Linear,
}

impl PatchMerger {
    fn new(cfg: &VisionConfig, use_postshuffle_norm: bool, vb: VarBuilder) -> Result<Self> {
        let merged_hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
        let norm_dim = if use_postshuffle_norm {
            merged_hidden_size
        } else {
            cfg.hidden_size
        };
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        Ok(Self {
            norm: layer_norm(norm_dim, norm_cfg, vb.pp("norm"))?,
            use_postshuffle_norm,
            spatial_merge_unit: cfg.spatial_merge_size.pow(2),
            merged_hidden_size,
            fc1: linear(merged_hidden_size, merged_hidden_size, vb.pp("linear_fc1"))?,
            fc2: linear(merged_hidden_size, cfg.out_hidden_size, vb.pp("linear_fc2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        if seq_len % self.spatial_merge_unit != 0 {
            candle_core::bail!(
                "Sequence length {} is not divisible by spatial merge unit {}",
                seq_len,
                self.spatial_merge_unit
            );
        }
        let grouped = seq_len / self.spatial_merge_unit;
        let norm_input = if self.use_postshuffle_norm {
            xs.reshape((grouped, self.merged_hidden_size))?
        } else {
            xs.clone()
        };
        let normed = self.norm.forward(&norm_input)?;
        let reshaped = if self.use_postshuffle_norm {
            normed
        } else {
            normed.reshape((grouped, self.merged_hidden_size))?
        };
        let xs = self.fc1.forward(&reshaped)?;
        let xs = xs.gelu_erf()?;
        self.fc2.forward(&xs)
    }
}

struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    const THETA: f32 = 10000.;

    fn new(dim: usize, device: &Device) -> Result<Self> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / Self::THETA.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?,
        })
    }

    fn make_embeds(&self, seqlen: usize) -> Result<Tensor> {
        let seq =
            Tensor::arange(0f32, seqlen as f32, self.inv_freq.device())?.unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq)
    }
}

pub struct VisionModel {
    patch_embed: PatchEmbed,
    pos_embed: Embedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
    num_grid_per_side: usize,
    hidden_size: usize,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        let patch_embed = PatchEmbed::new(cfg, vb.pp("patch_embed"))?;
        let pos_embed = embedding(
            cfg.num_position_embeddings,
            cfg.hidden_size,
            vb.pp("pos_embed"),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(VisionBlock::new(cfg, vb.pp(format!("blocks.{i}")))?);
        }

        let merger = PatchMerger::new(cfg, false, vb.pp("merger"))?;
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, vb.device())?;

        let num_grid_per_side = (cfg.num_position_embeddings as f64).sqrt().round() as usize;
        if num_grid_per_side * num_grid_per_side != cfg.num_position_embeddings {
            candle_core::bail!(
                "num_position_embeddings {} is not a perfect square",
                cfg.num_position_embeddings
            );
        }

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            merger,
            rotary_pos_emb,
            spatial_merge_size: cfg.spatial_merge_size,
            num_grid_per_side,
            hidden_size: cfg.hidden_size,
        })
    }

    fn linspace_points(&self, steps: usize) -> Vec<f32> {
        if steps == 1 {
            return vec![0.0];
        }
        let max_val = (self.num_grid_per_side - 1) as f32;
        let step = max_val / (steps.saturating_sub(1)) as f32;
        (0..steps).map(|i| i as f32 * step).collect()
    }

    fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.pos_embed.embeddings().device();
        let dtype = self.pos_embed.embeddings().dtype();
        let grid = grid_thw.to_vec2::<u32>()?;

        let mut idx_lists: [Vec<i64>; 4] = Default::default();
        let mut weight_lists: [Vec<f32>; 4] = Default::default();
        let mut hw_lengths = Vec::with_capacity(grid.len());

        for g in &grid {
            let h = g[1] as usize;
            let w = g[2] as usize;
            hw_lengths.push(h * w);

            let h_vals = self.linspace_points(h);
            let w_vals = self.linspace_points(w);

            let h_floor: Vec<usize> = h_vals.iter().map(|v| v.floor() as usize).collect();
            let w_floor: Vec<usize> = w_vals.iter().map(|v| v.floor() as usize).collect();
            let h_ceil: Vec<usize> = h_vals
                .iter()
                .map(|v| (v.ceil() as usize).min(self.num_grid_per_side - 1))
                .collect();
            let w_ceil: Vec<usize> = w_vals
                .iter()
                .map(|v| (v.ceil() as usize).min(self.num_grid_per_side - 1))
                .collect();
            let dh: Vec<f32> = h_vals
                .iter()
                .zip(&h_floor)
                .map(|(v, f)| v - *f as f32)
                .collect();
            let dw: Vec<f32> = w_vals
                .iter()
                .zip(&w_floor)
                .map(|(v, f)| v - *f as f32)
                .collect();

            for ((&hf, &hc), &dh_val) in h_floor.iter().zip(&h_ceil).zip(&dh) {
                for ((&wf, &wc), &dw_val) in w_floor.iter().zip(&w_ceil).zip(&dw) {
                    let base00 = (hf * self.num_grid_per_side + wf) as i64;
                    let base01 = (hf * self.num_grid_per_side + wc) as i64;
                    let base10 = (hc * self.num_grid_per_side + wf) as i64;
                    let base11 = (hc * self.num_grid_per_side + wc) as i64;

                    let w00 = (1.0 - dh_val) * (1.0 - dw_val);
                    let w01 = (1.0 - dh_val) * dw_val;
                    let w10 = dh_val * (1.0 - dw_val);
                    let w11 = dh_val * dw_val;

                    idx_lists[0].push(base00);
                    idx_lists[1].push(base01);
                    idx_lists[2].push(base10);
                    idx_lists[3].push(base11);

                    weight_lists[0].push(w00);
                    weight_lists[1].push(w01);
                    weight_lists[2].push(w10);
                    weight_lists[3].push(w11);
                }
            }
        }

        let idx_tensors = idx_lists
            .iter()
            .map(|idxs| Tensor::from_vec(idxs.clone(), (idxs.len(),), device))
            .collect::<Result<Vec<_>>>()?;
        let idx_tensor = Tensor::stack(&idx_tensors, 0)?;

        let weight_tensors = weight_lists
            .iter()
            .map(|weights| Tensor::from_vec(weights.clone(), (weights.len(),), device))
            .collect::<Result<Vec<_>>>()?;
        let weight_tensor = Tensor::stack(&weight_tensors, 0)?.to_dtype(dtype)?;

        let pos_embeds = self.pos_embed.forward(&idx_tensor)?;
        let pos_embeds = pos_embeds.broadcast_mul(&weight_tensor.unsqueeze(D::Minus1)?)?;
        let pos_embeds = pos_embeds.sum(0)?;

        let mut splits = Vec::with_capacity(hw_lengths.len());
        let mut start = 0;
        for len in hw_lengths {
            splits.push(pos_embeds.narrow(0, start, len)?);
            start += len;
        }

        let mut permuted = Vec::with_capacity(grid.len());
        for (pos_embed, g) in splits.into_iter().zip(&grid) {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let pos_embed = pos_embed.repeat((t, 1))?;
            let pos_embed = pos_embed.reshape((
                t,
                h / self.spatial_merge_size,
                self.spatial_merge_size,
                w / self.spatial_merge_size,
                self.spatial_merge_size,
                self.hidden_size,
            ))?;
            let pos_embed = pos_embed
                .permute((0, 1, 3, 2, 4, 5))?
                .reshape((t * h * w, self.hidden_size))?;
            permuted.push(pos_embed);
        }

        Tensor::cat(&permuted, 0)
    }

    fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.rotary_pos_emb.inv_freq.device();
        let grid = grid_thw.to_vec2::<u32>()?;
        let max_hw = grid
            .iter()
            .flat_map(|v| v[1..3].iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let freq_table = self.rotary_pos_emb.make_embeds(max_hw)?;

        let mut coords: Vec<(i64, i64)> = Vec::new();
        for g in &grid {
            let h = g[1] as usize;
            let w = g[2] as usize;
            let merged_h = h / self.spatial_merge_size;
            let merged_w = w / self.spatial_merge_size;

            let mut base_coords: Vec<(i64, i64)> = Vec::with_capacity(h * w);
            for br in 0..merged_h {
                for bc in 0..merged_w {
                    for ir in 0..self.spatial_merge_size {
                        for ic in 0..self.spatial_merge_size {
                            base_coords.push((
                                (br * self.spatial_merge_size + ir) as i64,
                                (bc * self.spatial_merge_size + ic) as i64,
                            ));
                        }
                    }
                }
            }

            for _ in 0..(g[0] as usize) {
                coords.extend(base_coords.iter().cloned());
            }
        }

        let total_tokens = coords.len();
        let mut rows = Vec::with_capacity(total_tokens);
        let mut cols = Vec::with_capacity(total_tokens);
        for &(r, c) in &coords {
            rows.push(r);
            cols.push(c);
        }
        let rows = Tensor::from_vec(rows, (total_tokens,), device)?;
        let cols = Tensor::from_vec(cols, (total_tokens,), device)?;
        let row_embeds = freq_table.index_select(&rows, 0)?;
        let col_embeds = freq_table.index_select(&cols, 0)?;
        Tensor::stack(&[row_embeds, col_embeds], D::Minus2)?
            .reshape((total_tokens, freq_table.dim(D::Minus1)? * 2))
    }

    fn build_cu_seqlens(&self, grid_thw: &Tensor) -> Result<Vec<usize>> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut cu = Vec::with_capacity(grid.iter().map(|v| v[0] as usize).sum::<usize>() + 1);
        cu.push(0usize);
        let mut acc = 0usize;
        for g in &grid {
            let area = (g[1] * g[2]) as usize;
            for _ in 0..(g[0] as usize) {
                acc += area;
                cu.push(acc);
            }
        }
        Ok(cu)
    }

    pub fn forward_patches(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let dtype = self.pos_embed.embeddings().dtype();
        let xs = self.patch_embed.forward(&xs.to_dtype(dtype)?)?;
        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?;
        let mut hidden_states = xs.add(&pos_embeds)?;

        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;

        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
        }

        let hidden_states = self.merger.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

impl VisionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.depth == 0
            || self.depth > 128
            || self.hidden_size == 0
            || self.hidden_size > 8192
            || self.out_hidden_size == 0
            || self.intermediate_size == 0
            || self.num_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_heads)
            || !(self.hidden_size / self.num_heads).is_multiple_of(4)
            || self.in_channels != 3
            || self.patch_size != 16
            || self.temporal_patch_size != 2
            || self.spatial_merge_size != 2
            || self.num_position_embeddings == 0
            || !self.deepstack_visual_indexes.is_empty()
        {
            candle_core::bail!(
                "Unsupported Qwen3.5 vision dimensions; images require RGB 16x16 patches, temporal repeat 2, spatial merge 2, no deepstack"
            );
        }
        Ok(())
    }
}

/// One decoded image in merged-block patch order; no request-owned GPU memory is retained by the model.
pub struct PreparedImage {
    pub pixels: Tensor,
    pub grid: [usize; 3],
}

/// Deterministic Qwen smart resize with a fixed pixel budget. The source dimensions
/// are validated separately before image decoding. Images retain their aspect ratio
/// to the nearest 32-pixel merge block. No video frames are admitted.
pub fn resize_grid(width: u32, height: u32, max_pixels: usize) -> Result<[usize; 3]> {
    const FACTOR: usize = 32;
    if width == 0 || height == 0 || !(FACTOR * FACTOR..=16_777_216).contains(&max_pixels) {
        candle_core::bail!("Invalid image dimensions or fixed pixel budget");
    }
    let w = width as f64;
    let h = height as f64;
    if w.max(h) / w.min(h) > 200. {
        candle_core::bail!("Image aspect ratio exceeds 200");
    }
    // Python round ties to even, which is also required for the processor geometry.
    let mut rw = ((w / FACTOR as f64).round_ties_even() as usize).max(1) * FACTOR;
    let mut rh = ((h / FACTOR as f64).round_ties_even() as usize).max(1) * FACTOR;
    if rw.checked_mul(rh).is_none_or(|n| n > max_pixels) {
        let beta = (h * w / max_pixels as f64).sqrt();
        rw = ((w / beta / FACTOR as f64).floor() as usize).max(1) * FACTOR;
        rh = ((h / beta / FACTOR as f64).floor() as usize).max(1) * FACTOR;
    } else {
        let min_pixels = 65_536usize.min(max_pixels);
        if rw * rh < min_pixels {
            let beta = (min_pixels as f64 / (h * w)).sqrt();
            rw = ((w * beta / FACTOR as f64).ceil() as usize).max(1) * FACTOR;
            rh = ((h * beta / FACTOR as f64).ceil() as usize).max(1) * FACTOR;
        }
    }
    if rw.checked_mul(rh).is_none_or(|n| n > max_pixels) {
        candle_core::bail!("Image aspect ratio cannot fit the fixed merge-block pixel budget");
    }
    Ok([1, rh / 16, rw / 16])
}

/// Decode a bounded PNG/JPEG, normalize to [-1,1], duplicate the static frame and
/// flatten patches in [block_h,block_w,inner_h,inner_w,channel,time,patch_h,patch_w].
pub fn prepare_image(encoded: &[u8], max_pixels: usize, device: &Device) -> Result<PreparedImage> {
    if encoded.is_empty() || encoded.len() > 8 << 20 {
        candle_core::bail!("Encoded image exceeds the 8 MiB bound");
    }
    let format =
        image::guess_format(encoded).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    if !matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg) {
        candle_core::bail!("Only PNG and JPEG images are supported");
    }
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(encoded), format);
    let limits = image::Limits::default();
    reader.limits(limits);
    let (width, height) = image::ImageReader::with_format(std::io::Cursor::new(encoded), format)
        .into_dimensions()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    if u64::from(width) * u64::from(height) > 16_777_216 {
        candle_core::bail!("Source image exceeds 16 megapixels");
    }
    let grid = resize_grid(width, height, max_pixels)?;
    let source = reader
        .decode()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        .to_rgb8();
    let resized = if width as usize == grid[2] * 16 && height as usize == grid[1] * 16 {
        source
    } else {
        image::imageops::resize(
            &source,
            (grid[2] * 16) as u32,
            (grid[1] * 16) as u32,
            image::imageops::FilterType::CatmullRom,
        )
    };
    let mut patches = Vec::with_capacity(grid[1] * grid[2] * 3 * 2 * 16 * 16);
    for bh in 0..grid[1] / 2 {
        for bw in 0..grid[2] / 2 {
            for ih in 0..2 {
                for iw in 0..2 {
                    for c in 0..3 {
                        for _t in 0..2 {
                            for ph in 0..16 {
                                for pw in 0..16 {
                                    let pixel = resized
                                        .get_pixel(
                                            ((bw * 2 + iw) * 16 + pw) as u32,
                                            ((bh * 2 + ih) * 16 + ph) as u32,
                                        )
                                        .0[c];
                                    patches.push((pixel as f32 / 255. - 0.5) / 0.5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(PreparedImage {
        pixels: Tensor::from_vec(patches, (grid[1] * grid[2], 3 * 2 * 16 * 16), device)?,
        grid,
    })
}

impl VisionModel {
    pub fn forward(&self, image: &PreparedImage) -> Result<Tensor> {
        if image.grid[0] != 1
            || image.grid[1] == 0
            || image.grid[2] == 0
            || !image.grid[1].is_multiple_of(2)
            || !image.grid[2].is_multiple_of(2)
            || image.grid[1] * image.grid[2] > 65_536
        {
            candle_core::bail!("Invalid bounded image patch grid");
        }
        let grid = Tensor::from_vec(
            image.grid.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            (1, 3),
            image.pixels.device(),
        )?;
        self.forward_patches(&image.pixels, &grid)
    }
}

impl VisionModel {
    /// Read a released llama.cpp Qwen3-VL merger mmproj. This file contains the
    /// complete vision encoder and merger, not just the final linear projector.
    /// Packed projector tensors are expanded once at startup; text weights retain
    /// their separately packed execution path.
    pub fn from_gguf(
        cfg: &VisionConfig,
        path: &std::path::Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        use candle_core::quantized::{ggml_file::qtensor_from_ggml, gguf_file::Value};
        use std::io::{Read, Seek, SeekFrom};
        use switchyard_quantized::gguf::{Index, IndexLimits};
        let mut file = std::fs::File::open(path)?;
        let index = Index::read(&mut file, IndexLimits::default())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let meta = index.metadata();
        let string = |key: &str, want: &str| -> Result<()> {
            if !matches!(meta.get(key),Some(Value::String(value)) if value==want) {
                candle_core::bail!("mmproj {key} must be {want}");
            }
            Ok(())
        };
        string("general.architecture", "clip")?;
        string("clip.projector_type", "qwen3vl_merger")?;
        if !matches!(meta.get("clip.has_vision_encoder"), Some(Value::Bool(true))) {
            candle_core::bail!("mmproj lacks its vision encoder");
        }
        for (key, want) in [
            ("clip.vision.projection_dim", cfg.out_hidden_size),
            ("clip.vision.patch_size", cfg.patch_size),
            ("clip.vision.embedding_length", cfg.hidden_size),
            ("clip.vision.feed_forward_length", cfg.intermediate_size),
            ("clip.vision.block_count", cfg.depth),
            ("clip.vision.attention.head_count", cfg.num_heads),
            ("clip.vision.spatial_merge_size", cfg.spatial_merge_size),
        ] {
            if !matches!(meta.get(key),Some(Value::U32(value)) if *value as usize==want) {
                candle_core::bail!("mmproj metadata {key} disagrees with model configuration");
            }
        }
        let mut used = 0u64;
        let mut load = |name: &str| -> Result<Tensor> {
            let desc = index
                .tensor(name)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let shape = desc.backend_shape();
            let elements = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| candle_core::Error::Msg("mmproj shape overflow".into()))?;
            used = used
                .checked_add(elements as u64 * 4)
                .filter(|&n| n <= 2 << 30)
                .ok_or_else(|| {
                    candle_core::Error::Msg(
                        "mmproj exceeds fixed 2 GiB expanded weight budget".into(),
                    )
                })?;
            let mut raw = vec![0u8; desc.bytes() as usize];
            file.seek(SeekFrom::Start(desc.file_offset()))?;
            file.read_exact(&mut raw)?;
            qtensor_from_ggml(desc.encoding().backend(), &raw, shape, device)?
                .dequantize(device)?
                .to_dtype(dtype)
        };
        let mut tensors = std::collections::HashMap::new();
        let mut insert = |native: &str, gguf: &str| -> Result<()> {
            tensors.insert(native.to_owned(), load(gguf)?);
            Ok(())
        };
        for i in 0..cfg.depth {
            for (native, gguf) in [
                ("attn.qkv", "attn_qkv"),
                ("attn.proj", "attn_out"),
                ("mlp.linear_fc1", "ffn_up"),
                ("mlp.linear_fc2", "ffn_down"),
                ("norm1", "ln1"),
                ("norm2", "ln2"),
            ] {
                for suffix in ["weight", "bias"] {
                    insert(
                        &format!("blocks.{i}.{native}.{suffix}"),
                        &format!("v.blk.{i}.{gguf}.{suffix}"),
                    )?;
                }
            }
        }
        for (native, gguf) in [
            ("merger.norm", "v.post_ln"),
            ("merger.linear_fc1", "mm.0"),
            ("merger.linear_fc2", "mm.2"),
        ] {
            for suffix in ["weight", "bias"] {
                insert(&format!("{native}.{suffix}"), &format!("{gguf}.{suffix}"))?;
            }
        }
        insert("pos_embed.weight", "v.position_embd.weight")?;
        insert("patch_embed.proj.bias", "v.patch_embd.bias")?;
        let first = load("v.patch_embd.weight")?;
        let second = load("v.patch_embd.weight.1")?;
        let patch = Tensor::stack(&[first, second], 2)?;
        tensors.insert("patch_embed.proj.weight".into(), patch);
        Self::new(cfg, VarBuilder::from_tensors(tensors, dtype, device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resize_obeys_patch_merging_and_bounds() {
        assert_eq!(resize_grid(256, 256, 65_536).unwrap(), [1, 16, 16]);
        assert_eq!(resize_grid(512, 256, 65_536).unwrap(), [1, 10, 22]);
        assert_eq!(resize_grid(16, 16, 65_536).unwrap(), [1, 16, 16]);
        assert!(resize_grid(0, 128, 65_536).is_err());
        assert!(resize_grid(128, 128, 512).is_err());
        assert!(resize_grid(40_000, 1, 65_536).is_err());
    }
    #[test]
    fn processor_patch_order_normalization_and_temporal_repeat() {
        let image = image::RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([(x * 8) as u8, (y * 8) as u8, 255])
        });
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let prepared = prepare_image(encoded.get_ref(), 1024, &Device::Cpu).unwrap();
        assert_eq!(prepared.grid, [1, 2, 2]);
        let values = prepared.pixels.to_vec2::<f32>().unwrap();
        assert_eq!(values.len(), 4);
        assert_eq!(values[0][0], -1.);
        assert_eq!(values[0][256], values[0][0]);
        assert_eq!(values[0][1024], 1.);
        assert_eq!(values[1][0], (128. / 255. - 0.5) / 0.5);
        assert_eq!(values[2][512], (128. / 255. - 0.5) / 0.5);
        assert!(prepare_image(&[], 1024, &Device::Cpu).is_err());
    }
}
