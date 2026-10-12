// Adapted from Candle PR #3396, commit e9e4929f3e31b6183cc17e7be0596929c6725257.
// Copyright Candle contributors, Apache-2.0 OR MIT; see THIRD_PARTY.md.
use super::projection::{Embedding, Linear, ModelWeights, linear_b, linear_no_bias};
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Conv1d, Conv1dConfig, VarBuilder, conv1d_no_bias};
use std::sync::Arc;
mod paged;
mod paged_cuda;
mod state;
pub use state::SequenceState;

#[derive(Clone, Copy)]
enum Positions<'a> {
    Shared(usize),
    Rows(&'a Tensor),
    Multimodal(&'a Tensor),
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<super::vision::VisionConfig>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub vision_start_token_id: Option<u32>,
    #[serde(default)]
    pub vision_end_token_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub attention_bias: bool,
    pub hidden_act: Activation,

    // Qwen 3.5 specific
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_section: Option<[usize; 3]>,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let c = &self.text_config;
        if c.num_attention_heads == 0
            || c.num_key_value_heads == 0
            || c.hidden_size == 0
            || c.vocab_size == 0
            || c.intermediate_size == 0
            || c.num_hidden_layers == 0
            || c.num_hidden_layers != c.layer_types.len()
            || c.layer_types
                .iter()
                .any(|x| !matches!(x.as_str(), "linear_attention" | "full_attention"))
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
            || c.linear_num_key_heads == 0
            || c.linear_num_value_heads == 0
            || !c
                .linear_num_value_heads
                .is_multiple_of(c.linear_num_key_heads)
            || c.linear_key_head_dim == 0
            || c.linear_value_head_dim == 0
            || c.linear_conv_kernel_dim < 2
            || c.max_position_embeddings == 0
            || c.rms_norm_eps <= 0.
            || !c.rms_norm_eps.is_finite()
            || c.rope_parameters.rope_theta <= 0.
            || !c.rope_parameters.rope_theta.is_finite()
        {
            candle_core::bail!("Invalid dense Qwen3.5 model dimensions or layer types");
        }
        let rotary = self.head_dim() as f64 * c.rope_parameters.partial_rotary_factor;
        if rotary < 2.
            || rotary > self.head_dim() as f64
            || rotary.fract() != 0.
            || !(rotary as usize).is_multiple_of(2)
        {
            candle_core::bail!("Unsupported partial rotary dimension");
        }
        Ok(())
    }
    pub fn head_dim(&self) -> usize {
        self.text_config
            .head_dim
            .unwrap_or(self.text_config.hidden_size / self.text_config.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f64 {
        self.text_config.rope_parameters.rope_theta
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen3_5RmsNorm {
    pub(crate) fn new(dim: usize, eps: f64, vb: ModelWeights) -> Result<Self> {
        let weight = vb.norm_weight(dim)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.normalize(&x.to_dtype(DType::F32)?, x.dtype())
    }

    fn add(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        // Fused residual normalization uses the unrounded F32 sum. The
        // separately retained residual is rounded to the model's dtype.
        let sum = (x.to_dtype(DType::F32)? + residual.to_dtype(DType::F32)?)?;
        Ok((self.normalize(&sum, x.dtype())?, sum.to_dtype(x.dtype())?))
    }

    fn normalize(&self, hidden_states: &Tensor, x_dtype: DType) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&hidden_states.contiguous()?, &self.weight, self.eps as f32)?
            .to_dtype(x_dtype)
    }
}

fn l2_norm(x: &Tensor) -> Result<Tensor> {
    let inv_norm = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?;
    x.broadcast_mul(&inv_norm)
}

fn repeat_interleave(x: &Tensor, n: usize, dim: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let mut dims = x.dims().to_vec();
    dims.insert(dim + 1, n);
    let expanded = x.unsqueeze(dim + 1)?.broadcast_as(dims.as_slice())?;
    let mut final_dims = x.dims().to_vec();
    final_dims[dim] *= n;
    expanded.reshape(final_dims.as_slice())
}

struct RecurrentInput<'a> {
    state: Option<&'a Tensor>,
    prefill: bool,
}

#[derive(Debug, Clone)]
pub struct Qwen3_5GatedDeltaNet {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_dim: usize,

    conv1d: Conv1d,
    dt_bias_f32: Tensor,
    neg_a_f32: Tensor,

    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,

    norm_weight: Tensor,
    norm_eps: f64,

    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl Qwen3_5GatedDeltaNet {
    pub(crate) fn new(cfg: &Config, vb: ModelWeights) -> Result<Self> {
        let hidden_size = cfg.text_config.hidden_size;
        let num_v_heads = cfg.text_config.linear_num_value_heads;
        let num_k_heads = cfg.text_config.linear_num_key_heads;
        let head_k_dim = cfg.text_config.linear_key_head_dim;
        let head_v_dim = cfg.text_config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;

        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = cfg.text_config.linear_conv_kernel_dim;

        let conv1d_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: conv_dim,
            ..Default::default()
        };
        let conv1d = conv1d_no_bias(
            conv_dim,
            conv_dim,
            conv_kernel_size,
            conv1d_cfg,
            vb.pp("conv1d").builder(),
        )?;

        let dt_bias = vb.get(num_v_heads, "dt_bias")?;
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
        let neg_a_f32 = vb.negative_a(num_v_heads)?;

        let in_proj_qkv = linear_no_bias(hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_a"))?;
        let out_proj = linear_no_bias(value_dim, hidden_size, vb.pp("out_proj"))?;

        let norm_weight = vb
            .clone()
            .set_dtype(DType::F32)
            .get(head_v_dim, "norm.weight")?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv_dim,
            conv1d,
            dt_bias_f32,
            neg_a_f32,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            norm_weight,
            norm_eps: cfg.text_config.rms_norm_eps,
            conv_state: None,
            recurrent_state: None,
        })
    }

    fn rms_norm_gated(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let variance = hidden_states.sqr()?.mean_keepdim(D::Minus1)?;
        let hidden_states =
            hidden_states.broadcast_mul(&(&variance + self.norm_eps)?.sqrt()?.recip()?)?;
        let hidden_states = hidden_states.broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?;
        let gate_silu = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        hidden_states
            .broadcast_mul(&gate_silu)?
            .to_dtype(input_dtype)
    }

    fn torch_recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial: RecurrentInput<'_>,
    ) -> Result<(Tensor, Tensor)> {
        let mut query_normalized = l2_norm(&query.to_dtype(DType::F32)?)?;
        let mut key_normalized = l2_norm(&key.to_dtype(DType::F32)?)?;
        // Native vLLM prefill stores normalized Q/K in the activation dtype;
        // its recurrent decode kernel keeps this normalization in F32.
        if initial.prefill {
            query_normalized = query_normalized
                .to_dtype(query.dtype())?
                .to_dtype(DType::F32)?;
            key_normalized = key_normalized.to_dtype(key.dtype())?.to_dtype(DType::F32)?;
        }
        let query = query_normalized;
        let key = key_normalized;

        let query = query.transpose(1, 2)?; // (batch, heads, seq, dim)
        let key = key.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;

        let (batch_size, num_heads, seq_len, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(3)?;

        let scale = 1.0 / (query.dim(3)? as f64).sqrt();
        let query = (query * scale)?;

        let mut core_attn_out_vec = Vec::with_capacity(seq_len);

        // We MUST use F32 for the recurrent state to avoid numerical drift
        let mut last_recurrent_state = match initial.state {
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                DType::F32,
                query.device(),
            )?,
        };

        // Move loop-invariant conversions outside and prepare dimensions
        let query_f32 = query.to_dtype(DType::F32)?;
        let key_f32 = key.to_dtype(DType::F32)?;
        let value_f32 = value.to_dtype(DType::F32)?;
        let g_exp = g.to_dtype(DType::F32)?.exp()?.unsqueeze(3)?; // (B, H, S, 1)
        let beta_f32 = beta.to_dtype(DType::F32)?.unsqueeze(3)?; // (B, H, S, 1)

        for i in 0..seq_len {
            let q_t = query_f32.narrow(2, i, 1)?.contiguous()?; // (B, H, 1, K)
            let k_t = key_f32.narrow(2, i, 1)?.contiguous()?; // (B, H, 1, K)
            let v_t = value_f32.narrow(2, i, 1)?; // (B, H, 1, V)
            let g_t = g_exp.narrow(2, i, 1)?; // (B, H, 1, 1)
            let beta_t = beta_f32.narrow(2, i, 1)?; // (B, H, 1, 1)

            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_t)?;

            // k_t.matmul(&state) is equivalent to broadcast_mul + sum(2)
            let kv_mem = k_t.matmul(&last_recurrent_state)?; // (B, H, 1, V)
            let delta = (&v_t - &kv_mem)?.broadcast_mul(&beta_t)?; // (B, H, 1, V)

            // k_t.T.matmul(&delta) is equivalent to outer product
            let delta_k = k_t.transpose(2, 3)?.matmul(&delta)?; // (B, H, K, 1) x (B, H, 1, V) = (B, H, K, V)
            last_recurrent_state = (&last_recurrent_state + &delta_k)?;

            let out_t = q_t.matmul(&last_recurrent_state)?; // (B, H, 1, V)
            core_attn_out_vec.push(out_t);
        }

        let core_attn_out = Tensor::cat(&core_attn_out_vec, 2)?; // (B, H, S, V)
        let core_attn_out = core_attn_out.transpose(1, 2)?; // (B, S, H, V)
        Ok((core_attn_out, last_recurrent_state))
    }

    pub fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let prefill = self.recurrent_state.is_none() || hidden_states.dim(1)? > 1;
        self.forward_internal(hidden_states, prefill)
    }

    fn forward_internal(&mut self, hidden_states: &Tensor, prefill: bool) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let hidden_states = hidden_states.contiguous()?;
        let initial_dtype = hidden_states.dtype();

        let mixed_qkv = self.in_proj_qkv.forward(&hidden_states)?.transpose(1, 2)?;

        let z = self.in_proj_z.forward(&hidden_states)?;
        let z = z.reshape((batch_size, seq_len, (), self.head_v_dim))?;

        let b = self.in_proj_b.forward(&hidden_states)?;
        let a = self.in_proj_a.forward(&hidden_states)?;

        let use_precomputed_states = self.conv_state.is_some();

        let mixed_qkv = if use_precomputed_states {
            let conv_state = self.conv_state.as_mut().unwrap();
            let conv_state_data = Tensor::cat(&[&*conv_state, &mixed_qkv], 2)?;
            *conv_state = conv_state_data.narrow(2, seq_len, self.conv_kernel_size - 1)?;
            let out = depthwise_conv(&conv_state_data, self.conv1d.weight())?;
            candle_nn::ops::silu(&out)?.to_dtype(initial_dtype)?
        } else {
            let pad = self.conv_kernel_size - 1;
            let padding = Tensor::zeros(
                (batch_size, self.conv_dim, pad),
                mixed_qkv.dtype(),
                mixed_qkv.device(),
            )?;
            let padded_qkv = Tensor::cat(&[&padding, &mixed_qkv], 2)?;
            self.conv_state = Some(padded_qkv.narrow(2, seq_len, pad)?);
            let out = depthwise_conv(&padded_qkv, self.conv1d.weight())?;
            candle_nn::ops::silu(&out)?.to_dtype(initial_dtype)?
        };

        let mixed_qkv = mixed_qkv.transpose(1, 2)?; // (batch, seq, conv_dim)

        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, (), self.head_v_dim))?;

        let beta = candle_nn::ops::sigmoid(&b.to_dtype(DType::F32)?)?;
        let g = {
            let a_f32 = a.to_dtype(DType::F32)?;
            let a_plus_dt = a_f32.broadcast_add(&self.dt_bias_f32)?;
            let softplus =
                (a_plus_dt.relu()? + ((a_plus_dt.abs()? * -1.0)?.exp()? + 1.0)?.log()?)?;
            self.neg_a_f32.broadcast_mul(&softplus)?
        };

        let repeat_n = self.num_v_heads / self.num_k_heads;
        let q = repeat_interleave(&q, repeat_n, 2)?;
        let k = repeat_interleave(&k, repeat_n, 2)?;

        let initial_state = if use_precomputed_states {
            self.recurrent_state.as_ref()
        } else {
            None
        };

        let (core_attn_out, new_state) = self.torch_recurrent_gated_delta_rule(
            &q,
            &k,
            &v,
            &g,
            &beta,
            RecurrentInput {
                state: initial_state,
                prefill,
            },
        )?;
        self.recurrent_state = Some(new_state);

        let core_attn_out = core_attn_out.to_dtype(initial_dtype)?;
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = self.rms_norm_gated(&core_attn_out, &z_flat)?;
        let core_attn_out = core_attn_out.reshape((batch_size, seq_len, ()))?;

        self.out_proj.forward(&core_attn_out)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5TextRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    mrope_section: Option<[usize; 3]>,
}

impl Qwen3_5TextRotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = (cfg.head_dim() as f64 * cfg.text_config.rope_parameters.partial_rotary_factor)
            as usize;
        let max_seq_len = cfg.text_config.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta().powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            mrope_section: cfg.text_config.rope_parameters.mrope_section,
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        self.apply_positions(q, k, Positions::Shared(seqlen_offset))
    }

    fn apply_positions(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: Positions<'_>,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let (cos, sin) = match positions {
            Positions::Shared(offset) => (
                self.cos.narrow(0, offset, seq_len)?,
                self.sin.narrow(0, offset, seq_len)?,
            ),
            Positions::Rows(ids) => {
                if seq_len != 1 {
                    candle_core::bail!("Per-row positions require single-token decode");
                }
                (
                    self.cos.index_select(ids, 0)?.unsqueeze(1)?.contiguous()?,
                    self.sin.index_select(ids, 0)?.unsqueeze(1)?.contiguous()?,
                )
            }
            Positions::Multimodal(ids) => {
                let (axes, batch, length) = ids.dims3()?;
                if axes != 3 || batch != 1 || length != seq_len {
                    candle_core::bail!("Image positions require [3,1,sequence] and batch one");
                }
                let section = self.mrope_section.ok_or_else(|| {
                    candle_core::Error::Msg("Image model lacks MRoPE sections".into())
                })?;
                let width = self.cos.dim(1)?;
                if section.iter().sum::<usize>() != width {
                    candle_core::bail!("MRoPE sections do not cover rotary frequencies");
                }
                let select = |table: &Tensor| -> Result<Tensor> {
                    let indices = ids.flatten_all()?;
                    let tables = table
                        .index_select(&indices, 0)?
                        .reshape((3, seq_len, width))?;
                    let mut cols = Vec::with_capacity(width);
                    for j in 0..width {
                        let axis = if j % 3 == 1 && j < section[1] * 3 {
                            1
                        } else if j % 3 == 2 && j < section[2] * 3 {
                            2
                        } else {
                            0
                        };
                        cols.push(tables.narrow(0, axis, 1)?.squeeze(0)?.narrow(1, j, 1)?);
                    }
                    Tensor::cat(&cols, 1)
                };
                (select(&self.cos)?, select(&self.sin)?)
            }
        };
        let rotary_dim = cos.dim(D::Minus1)? * 2;
        let apply = |x: &Tensor| -> Result<Tensor> {
            let rotated = candle_nn::rotary_emb::rope(
                &x.narrow(3, 0, rotary_dim)?.contiguous()?,
                &cos,
                &sin,
            )?;
            Tensor::cat(
                &[&rotated, &x.narrow(3, rotary_dim, x.dim(3)? - rotary_dim)?],
                3,
            )
        };
        let q_embed = apply(q)?;
        let k_embed = apply(k)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3_5RmsNorm,
    k_norm: Qwen3_5RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
    kv_cache: Option<Vec<paged::PagedKv>>,
    kv_page_tokens: usize,
}

impl Qwen3_5Attention {
    pub(crate) fn new(
        cfg: &Config,
        rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
        vb: ModelWeights,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.text_config.num_attention_heads;
        let num_kv_heads = cfg.text_config.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            cfg.text_config.hidden_size,
            num_heads * head_dim * 2,
            cfg.text_config.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.text_config.hidden_size,
            num_kv_heads * head_dim,
            cfg.text_config.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.text_config.hidden_size,
            num_kv_heads * head_dim,
            cfg.text_config.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.text_config.hidden_size,
            cfg.text_config.attention_bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache: None,
            kv_page_tokens: 16,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.forward_internal(
            hidden_states,
            attention_mask,
            Positions::Shared(seqlen_offset),
            None,
        )
    }

    fn forward_internal(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        positions: Positions<'_>,
        mut captured: Option<&mut super::LayerTrace>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = hidden_states.dims3()?;

        let q_proj_out = self.q_proj.forward(hidden_states)?;
        if let Some(c) = captured.as_mut() {
            c.push(("q_proj".into(), q_proj_out.clone()));
        }
        let q_proj_out = q_proj_out.reshape((b_sz, q_len, self.num_heads, self.head_dim * 2))?;

        let query_states = q_proj_out.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_proj_out.narrow(D::Minus1, self.head_dim, self.head_dim)?;

        let query_states = self.q_norm.forward(&query_states)?;
        if let Some(c) = captured.as_mut() {
            c.push(("q_norm".into(), query_states.clone()));
        }
        let query_states = query_states.transpose(1, 2)?;

        let key_states = self.k_proj.forward(hidden_states)?;
        if let Some(c) = captured.as_mut() {
            c.push(("k_proj".into(), key_states.clone()));
        }
        let key_states = key_states.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let key_states = self.k_norm.forward(&key_states)?;
        if let Some(c) = captured.as_mut() {
            c.push(("k_norm".into(), key_states.clone()));
        }
        let key_states = key_states.transpose(1, 2)?;

        let value_states = self.v_proj.forward(hidden_states)?;
        if let Some(c) = captured.as_mut() {
            c.push(("v_proj".into(), value_states.clone()));
        }
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_positions(&query_states, &key_states, positions)?;
        if let Some(c) = captured.as_mut() {
            c.push(("q_rope".into(), query_states.clone()));
            c.push(("k_rope".into(), key_states.clone()));
        }

        let fresh = self.kv_cache.is_none();
        let mut pages = self.kv_cache.clone().unwrap_or_else(|| {
            (0..b_sz)
                .map(|_| paged::PagedKv::new(self.kv_page_tokens))
                .collect()
        });
        if pages.len() != b_sz {
            candle_core::bail!("Attention state row mismatch");
        }
        for (row, cache) in pages.iter_mut().enumerate() {
            cache.append(
                &key_states.narrow(0, row, 1)?,
                &value_states.narrow(0, row, 1)?,
            )?;
        }
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = if fresh {
            // Keep batched initial prefill on the established dense path. Only
            // newly projected keys and values are materialized here.
            let keys = repeat_interleave(&key_states, self.num_kv_groups, 1)?.contiguous()?;
            let values = repeat_interleave(&value_states, self.num_kv_groups, 1)?.contiguous()?;
            let scores = (query_states
                .to_dtype(DType::F32)?
                .contiguous()?
                .matmul(&keys.to_dtype(DType::F32)?.transpose(2, 3)?)?
                * scale)?;
            let scores = match attention_mask {
                None => scores,
                Some(mask) => scores.broadcast_add(&mask.to_dtype(DType::F32)?)?,
            };
            let exponentials = scores
                .broadcast_sub(&scores.max_keepdim(D::Minus1)?)?
                .exp()?;
            let denominator = exponentials.sum_keepdim(D::Minus1)?;
            exponentials
                .to_dtype(values.dtype())?
                .to_dtype(DType::F32)?
                .matmul(&values.to_dtype(DType::F32)?)?
                .broadcast_div(&denominator)?
                .to_dtype(hidden_states.dtype())?
        } else if q_len == 1 && attention_mask.is_none() {
            match paged_cuda::attention(&query_states, &pages, self.num_kv_groups)? {
                Some(output) => output.to_dtype(hidden_states.dtype())?,
                None => paged::attention(
                    &query_states,
                    &pages,
                    self.num_kv_groups,
                    None,
                    scale,
                    hidden_states.dtype(),
                )?,
            }
        } else {
            paged::attention(
                &query_states,
                &pages,
                self.num_kv_groups,
                attention_mask,
                scale,
                hidden_states.dtype(),
            )?
        };
        self.kv_cache = Some(pages);

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?;

        let attn_output = attn_output
            .to_dtype(DType::F32)?
            .broadcast_mul(&candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?
            .to_dtype(hidden_states.dtype())?;

        let attn_output = attn_output.reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        if let Some(c) = captured.as_mut() {
            c.push(("o_input".into(), attn_output.clone()));
        }
        self.o_proj.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3_5MLP {
    pub(crate) fn new(cfg: &Config, vb: ModelWeights) -> Result<Self> {
        let hidden_sz = cfg.text_config.hidden_size;
        let intermediate_sz = cfg.text_config.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.text_config.hidden_act,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Match the fused activation's single rounding after SiLU * up.
        let lhs = xs
            .apply(&self.gate_proj)?
            .to_dtype(DType::F32)?
            .apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?.to_dtype(DType::F32)?;
        (lhs * rhs)?.to_dtype(xs.dtype())?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
enum TokenMixer {
    FullAttention(Qwen3_5Attention),
    LinearAttention(Qwen3_5GatedDeltaNet),
}

#[derive(Debug, Clone)]
struct Qwen3_5DecoderLayer {
    token_mixer: TokenMixer,
    mlp: Qwen3_5MLP,
    input_layernorm: Qwen3_5RmsNorm,
    post_attention_layernorm: Qwen3_5RmsNorm,
}

impl Qwen3_5DecoderLayer {
    fn new(
        cfg: &Config,
        layer_idx: usize,
        rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
        vb: ModelWeights,
    ) -> Result<Self> {
        let layer_type = cfg
            .text_config
            .layer_types
            .get(layer_idx)
            .cloned()
            .unwrap_or_else(|| "full_attention".to_string());

        let token_mixer = if layer_type == "linear_attention" {
            TokenMixer::LinearAttention(Qwen3_5GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?)
        } else {
            TokenMixer::FullAttention(Qwen3_5Attention::new(cfg, rotary_emb, vb.pp("self_attn"))?)
        };

        let mlp = Qwen3_5MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            token_mixer,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        positions: Positions<'_>,
        captured: Option<&mut super::LayerTrace>,
        prefill: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (norm_hidden_states, residual) = match residual {
            Some(residual) => self.input_layernorm.add(hidden_states, residual)?,
            None => (
                self.input_layernorm.forward(hidden_states)?,
                hidden_states.clone(),
            ),
        };

        let mixed_states = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => {
                attn.forward_internal(&norm_hidden_states, attention_mask, positions, captured)?
            }
            TokenMixer::LinearAttention(attn) => {
                attn.forward_internal(&norm_hidden_states, prefill)?
            }
        };

        let (norm_hidden_states, residual) = self
            .post_attention_layernorm
            .add(&mixed_states, &residual)?;
        let mlp_states = self.mlp.forward(&norm_hidden_states)?;
        Ok((mlp_states, residual))
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.kv_cache = None,
            TokenMixer::LinearAttention(attn) => {
                attn.conv_state = None;
                attn.recurrent_state = None;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RmsNorm,
    device: Device,
    dtype: DType,
    lora_applied: bool,
    state_owner: Arc<()>,
    state_positions: Vec<usize>,
    max_positions: usize,
}

impl Model {
    pub(crate) fn new(cfg: &Config, vb: ModelWeights) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = vb_m
            .pp("embed_tokens")
            .embedding(cfg.text_config.vocab_size, cfg.text_config.hidden_size)?;
        let rotary_emb = Arc::new(Qwen3_5TextRotaryEmbedding::new(
            vb.dtype(),
            cfg,
            vb_m.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.text_config.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.text_config.num_hidden_layers {
            let layer =
                Qwen3_5DecoderLayer::new(cfg, layer_idx, rotary_emb.clone(), vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let norm = Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb_m.pp("norm"),
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            lora_applied: false,
            state_owner: Arc::new(()),
            state_positions: Vec::new(),
            max_positions: cfg.text_config.max_position_embeddings,
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), mask.dtype(), &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_internal(input_ids, seqlen_offset, None)
    }

    pub fn prefill(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_mode(input_ids, seqlen_offset, None, true)
    }

    pub(crate) fn forward_internal(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        captured: Option<&mut super::LayerTrace>,
    ) -> Result<Tensor> {
        let prefill = seqlen_offset == 0 || input_ids.dim(1)? > 1;
        self.forward_mode(input_ids, seqlen_offset, captured, prefill)
    }

    fn forward_mode(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        captured: Option<&mut super::LayerTrace>,
        prefill: bool,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let end = seqlen_offset
            .checked_add(seq_len)
            .filter(|&end| end <= self.max_positions)
            .ok_or_else(|| {
                candle_core::Error::Msg("Sequence exceeds prepared context limit".into())
            })?;
        if b_size == 0 || seq_len == 0 {
            candle_core::bail!("Empty sequence cohort");
        }
        if (seqlen_offset == 0 && !self.state_positions.is_empty())
            || (seqlen_offset != 0 && self.state_positions != vec![seqlen_offset; b_size])
        {
            candle_core::bail!("Forward position does not match owned sequence state");
        }
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        };

        let output = self.forward_with_positions(
            input_ids,
            attention_mask.as_ref(),
            Positions::Shared(seqlen_offset),
            captured,
            prefill,
        )?;
        self.state_positions = vec![end; b_size];
        Ok(output)
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        positions: Positions<'_>,
        captured: Option<&mut super::LayerTrace>,
        prefill: bool,
    ) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        self.forward_embeddings(xs, attention_mask, positions, captured, prefill)
    }

    fn forward_embeddings(
        &mut self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        positions: Positions<'_>,
        mut captured: Option<&mut super::LayerTrace>,
        prefill: bool,
    ) -> Result<Tensor> {
        if let Some(captured) = captured.as_mut() {
            captured.push(("embedding".into(), xs.clone()));
        }
        let mut residual = None;
        let mut attention_captured = false;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let tap = if !attention_captured
                && matches!(layer.token_mixer, TokenMixer::FullAttention(_))
            {
                attention_captured = true;
                captured.as_deref_mut()
            } else {
                None
            };
            let (hidden, next_residual) = layer.forward(
                &xs,
                residual.as_ref(),
                attention_mask,
                positions,
                tap,
                prefill,
            )?;
            xs = hidden;
            residual = Some(next_residual);
            if let Some(captured) = captured.as_mut() {
                captured.push((
                    format!("layer_{index}"),
                    (&xs + residual.as_ref().unwrap())?,
                ));
            }
        }
        match residual {
            Some(residual) => Ok(self.norm.add(&xs, &residual)?.0),
            None => self.norm.forward(&xs),
        }
    }

    /// Choose a fixed page size before any session state is created.
    pub fn set_kv_page_tokens(&mut self, tokens: usize) -> Result<()> {
        if tokens == 0 || tokens > 4096 || !self.state_positions.is_empty() {
            candle_core::bail!("KV page size must be 1..=4096 on an empty model");
        }
        for layer in &mut self.layers {
            if let TokenMixer::FullAttention(attention) = &mut layer.token_mixer {
                attention.kv_page_tokens = tokens;
            }
        }
        Ok(())
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
        self.state_positions.clear();
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    pub base_model: Model,
    pub lm_head: Linear,
}

impl ModelForCausalLM {
    /// Creates an adapter-specific graph while sharing every immutable base
    /// tensor. Sequence state and prefix checkpoints must remain adapter-local.
    pub fn with_lora(&self, adapter: &super::lora::Adapter) -> crate::Result<Self> {
        if self.base_model.lora_applied || !matches!(self.lm_head, Linear::Dense(_)) {
            return Err(crate::Error::Config(
                "LoRA requires a dense unadapted safetensors base model".into(),
            ));
        }
        let mut model = self.clone();
        model.clear_kv_cache();
        model.base_model.state_owner = Arc::new(());
        model.base_model.lora_applied = true;
        let mut consumed = std::collections::BTreeSet::new();
        for (index, layer) in model.base_model.layers.iter_mut().enumerate() {
            let prefix = format!("model.language_model.layers.{index}");
            for (name, projection) in [
                ("gate_proj", &mut layer.mlp.gate_proj),
                ("up_proj", &mut layer.mlp.up_proj),
                ("down_proj", &mut layer.mlp.down_proj),
            ] {
                projection.attach_lora(&format!("{prefix}.mlp.{name}"), adapter, &mut consumed)?;
            }
            match &mut layer.token_mixer {
                TokenMixer::FullAttention(attention) => {
                    for (name, projection) in [
                        ("q_proj", &mut attention.q_proj),
                        ("k_proj", &mut attention.k_proj),
                        ("v_proj", &mut attention.v_proj),
                        ("o_proj", &mut attention.o_proj),
                    ] {
                        projection.attach_lora(
                            &format!("{prefix}.self_attn.{name}"),
                            adapter,
                            &mut consumed,
                        )?;
                    }
                }
                TokenMixer::LinearAttention(attention) => {
                    for (name, projection) in [
                        ("in_proj_qkv", &mut attention.in_proj_qkv),
                        ("in_proj_z", &mut attention.in_proj_z),
                        ("in_proj_b", &mut attention.in_proj_b),
                        ("in_proj_a", &mut attention.in_proj_a),
                        ("out_proj", &mut attention.out_proj),
                    ] {
                        projection.attach_lora(
                            &format!("{prefix}.linear_attn.{name}"),
                            adapter,
                            &mut consumed,
                        )?;
                    }
                }
            }
        }
        model
            .lm_head
            .attach_lora("lm_head", adapter, &mut consumed)?;
        adapter.check_consumed(&consumed)?;
        Ok(model)
    }

    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_weights(cfg, ModelWeights::dense(vb))
    }

    pub(crate) fn new_with_weights(cfg: &Config, vb: ModelWeights) -> Result<Self> {
        cfg.validate()?;
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        } else {
            base_model.embed_tokens.tied_output()
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    /// Prompt continuation retains prefill normalization even for one token.
    pub fn prefill(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        self.base_model
            .prefill(input_ids, seqlen_offset)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    /// Image spans replace only their already expanded placeholder embeddings.
    /// Images use request-owned state and bypass text-prefix lookup entirely.
    pub fn prefill_images(
        &mut self,
        input_ids: &Tensor,
        images: &[ImageSpan],
    ) -> Result<(Tensor, i64)> {
        let (batch, length) = input_ids.dims2()?;
        if batch != 1
            || length == 0
            || !self.base_model.state_positions.is_empty()
            || length > self.base_model.max_positions
        {
            candle_core::bail!("Image prefill requires one bounded empty sequence");
        }
        let (position_ids, delta) = image_positions(length, images, input_ids.device())?;
        let base = self.base_model.embed_tokens.forward(input_ids)?;
        let mut chunks = Vec::new();
        let mut cursor = 0;
        for image in images {
            let count = image.embeddings.dim(0)?;
            if image.embeddings.dim(1)? != base.dim(2)? {
                candle_core::bail!("Image projector hidden width mismatch");
            }
            if image.offset > cursor {
                chunks.push(base.narrow(1, cursor, image.offset - cursor)?);
            }
            chunks.push(image.embeddings.to_dtype(base.dtype())?.unsqueeze(0)?);
            cursor = image.offset + count;
        }
        if cursor < length {
            chunks.push(base.narrow(1, cursor, length - cursor)?);
        }
        let embeddings = Tensor::cat(&chunks, 1)?;
        let mask = if length > 1 {
            Some(
                self.base_model
                    .prepare_causal_attention_mask(1, length, 0)?,
            )
        } else {
            None
        };
        let hidden = self.base_model.forward_embeddings(
            embeddings,
            mask.as_ref(),
            Positions::Multimodal(&position_ids),
            None,
            true,
        )?;
        self.base_model.state_positions = vec![length];
        Ok((
            hidden.narrow(1, length - 1, 1)?.apply(&self.lm_head)?,
            delta,
        ))
    }

    /// Physical sequence length remains independent of image rotary coordinates.
    pub fn forward_with_rope_offset(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
        delta: i64,
    ) -> Result<Tensor> {
        let (batch, length) = input_ids.dims2()?;
        if batch != 1
            || length != 1
            || self.base_model.state_positions != vec![offset]
            || offset >= self.base_model.max_positions
        {
            candle_core::bail!("Image decode position does not match owned state");
        }
        let rotary = (offset as i64)
            .checked_add(delta)
            .filter(|&x| x >= 0 && x < (self.base_model.max_positions as i64))
            .ok_or_else(|| {
                candle_core::Error::Msg("Image rotary position exceeds prepared bound".into())
            })? as usize;
        let output = self.base_model.forward_with_positions(
            input_ids,
            None,
            Positions::Shared(rotary),
            None,
            false,
        )?;
        self.base_model.state_positions = vec![offset + 1];
        output.apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache();
    }

    pub fn decode_rows(&mut self, input_ids: &Tensor, positions: &[usize]) -> Result<Tensor> {
        self.base_model
            .decode_rows(input_ids, positions)?
            .apply(&self.lm_head)
    }
}

/// A depthwise short convolution across all channels in four vector operations.
/// Candle grouped conv1d launches once per channel; this model has 6,144 channels.
fn depthwise_conv(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (_, channels, length) = input.dims3()?;
    let kernel = weight.dim(2)?;
    let out_len = length - kernel + 1;
    let x = input.to_dtype(DType::F32)?;
    let w = weight.to_dtype(DType::F32)?.reshape((channels, kernel))?;
    let mut output = Tensor::zeros(
        (input.dim(0)?, channels, out_len),
        DType::F32,
        input.device(),
    )?;
    for tap in 0..kernel {
        output = (output
            + x.narrow(2, tap, out_len)?
                .broadcast_mul(&w.narrow(1, tap, 1)?.reshape((1, channels, 1))?)?)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_norm_uses_sum_before_bf16_rounding() -> Result<()> {
        let dev = Device::Cpu;
        let norm = Qwen3_5RmsNorm::new(
            2,
            1e-6,
            ModelWeights::dense(VarBuilder::zeros(DType::BF16, &dev)),
        )?;
        let input = Tensor::new(&[[0.25f32, 1.]], &dev)?.to_dtype(DType::BF16)?;
        let residual = Tensor::new(&[[-0.00390625f32, 0.00390625]], &dev)?.to_dtype(DType::BF16)?;
        let (normalized, retained) = norm.add(&input, &residual)?;
        assert_eq!(
            retained.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            [[0.24609375, 1.]]
        );
        assert_eq!(
            normalized.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            [[0.3359375, 1.375]]
        );
        assert_eq!(
            norm.forward(&retained)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?,
            [[173. / 512., 1.375]]
        );
        Ok(())
    }
}

/// A contiguous image-token region after the vision start marker.
#[derive(Clone)]
pub struct ImageSpan {
    pub offset: usize,
    pub grid: [usize; 3],
    pub embeddings: Tensor,
}

/// Released Qwen3.5 spatial positions: each merged image token has `[t,h,w]`
/// coordinates starting at the previous text position. Following text starts
/// after max(grid_h,grid_w)/2 rather than after the image token count.
pub fn image_positions(
    length: usize,
    images: &[ImageSpan],
    device: &Device,
) -> Result<(Tensor, i64)> {
    if images.is_empty() || images.len() > 4 {
        candle_core::bail!("Image spans outside 1..=4");
    }
    let mut positions = [Vec::<u32>::new(), Vec::new(), Vec::new()];
    let mut cursor = 0;
    let mut current = 0usize;
    for image in images {
        let [t, h, w] = image.grid;
        let count = image.embeddings.dim(0)?;
        if t != 1
            || h == 0
            || w == 0
            || !h.is_multiple_of(2)
            || !w.is_multiple_of(2)
            || h.checked_mul(w).is_none_or(|n| n / 4 != count)
            || image.offset < cursor
            || image
                .offset
                .checked_add(count)
                .is_none_or(|end| end > length)
        {
            candle_core::bail!("Invalid image span or projector token count");
        }
        for _ in cursor..image.offset {
            for axis in &mut positions {
                axis.push(current as u32);
            }
            current += 1;
        }
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                positions[0].push(current as u32);
                positions[1].push((current + y) as u32);
                positions[2].push((current + x) as u32);
            }
        }
        current += h.max(w) / 2;
        cursor = image.offset + count;
    }
    for _ in cursor..length {
        for axis in &mut positions {
            axis.push(current as u32);
        }
        current += 1;
    }
    let maximum = positions
        .iter()
        .flat_map(|x| x.iter())
        .copied()
        .max()
        .unwrap_or(0) as i64;
    Ok((
        Tensor::from_vec(
            positions.into_iter().flatten().collect::<Vec<_>>(),
            (3, 1, length),
            device,
        )?,
        maximum + 1 - length as i64,
    ))
}

#[cfg(test)]
mod vision_position_tests {
    use super::*;
    #[test]
    fn spatial_positions_and_decode_delta() {
        let image = ImageSpan {
            offset: 2,
            grid: [1, 4, 6],
            embeddings: Tensor::zeros((6, 4), DType::F32, &Device::Cpu).unwrap(),
        };
        let (p, delta) = image_positions(10, &[image], &Device::Cpu).unwrap();
        let axes = p.squeeze(1).unwrap().to_vec2::<u32>().unwrap();
        assert_eq!(axes[0], vec![0, 1, 2, 2, 2, 2, 2, 2, 5, 6]);
        assert_eq!(axes[1], vec![0, 1, 2, 2, 2, 3, 3, 3, 5, 6]);
        assert_eq!(axes[2], vec![0, 1, 2, 3, 4, 2, 3, 4, 5, 6]);
        assert_eq!(delta, -3);
    }
    #[test]
    fn spatial_rotary_matches_text_rotary_when_axes_are_equal() {
        let table = Tensor::arange(0f32, 64. * 32., &Device::Cpu)
            .unwrap()
            .reshape((64, 32))
            .unwrap();
        let rope = Qwen3_5TextRotaryEmbedding {
            cos: table.cos().unwrap(),
            sin: table.sin().unwrap(),
            mrope_section: Some([11, 11, 10]),
        };
        let query = Tensor::arange(0f32, 2. * 5. * 80., &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 5, 80))
            .unwrap();
        let axis = Tensor::arange(0u32, 5, &Device::Cpu).unwrap();
        let positions = Tensor::stack(&[&axis, &axis, &axis], 0)
            .unwrap()
            .unsqueeze(1)
            .unwrap();
        let (text, _) = rope.apply(&query, &query, 0).unwrap();
        let (spatial, _) = rope
            .apply_positions(&query, &query, Positions::Multimodal(&positions))
            .unwrap();
        assert_eq!(
            text.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            spatial.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn invalid_or_overlapping_spans_fail() {
        let mut image = ImageSpan {
            offset: 2,
            grid: [1, 4, 4],
            embeddings: Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap(),
        };
        assert!(image_positions(5, &[image.clone()], &Device::Cpu).is_err());
        assert!(image_positions(12, &[image.clone(), image.clone()], &Device::Cpu).is_err());
        image.grid = [1, 3, 4];
        assert!(image_positions(12, &[image], &Device::Cpu).is_err());
    }
}
