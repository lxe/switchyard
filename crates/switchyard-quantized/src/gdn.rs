//! Qwen GGUF recurrent layout requirements, not a model implementation.
use crate::{
    Error, Result,
    gguf::{Encoding, Index},
};
use candle_core::quantized::gguf_file::Value;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct HeadOrder {
    key_heads: usize,
    value_heads: usize,
    head_width: usize,
    channels: usize,
}
impl HeadOrder {
    pub fn new(key_heads: usize, value_heads: usize, head_width: usize) -> Result<Self> {
        if key_heads == 0
            || value_heads == 0
            || head_width == 0
            || !value_heads.is_multiple_of(key_heads)
        {
            return Err(Error::Invalid(
                "value heads must be a nonzero multiple of key heads".into(),
            ));
        }
        let channels = value_heads
            .checked_mul(head_width)
            .ok_or_else(|| Error::Invalid("head channel overflow".into()))?;
        Ok(Self {
            key_heads,
            value_heads,
            head_width,
            channels,
        })
    }
    /// HF groups V heads under each K head; llama.cpp tiles K heads for broadcast.
    /// This maps one V channel without constructing a full permutation tensor.
    pub fn grouped_to_tiled(self, channel: usize) -> Result<usize> {
        if channel >= self.channels {
            return Err(Error::Invalid("V channel out of bounds".into()));
        }
        let head = channel / self.head_width;
        let ratio = self.value_heads / self.key_heads;
        Ok(
            ((head % ratio) * self.key_heads + head / ratio) * self.head_width
                + channel % self.head_width,
        )
    }
    pub fn tiled_to_grouped(self, channel: usize) -> Result<usize> {
        if channel >= self.channels {
            return Err(Error::Invalid("V channel out of bounds".into()));
        }
        let head = channel / self.head_width;
        let ratio = self.value_heads / self.key_heads;
        Ok(
            ((head % self.key_heads) * ratio + head / self.key_heads) * self.head_width
                + channel % self.head_width,
        )
    }
}

#[derive(Debug, Serialize)]
pub struct Qwen35Requirements {
    pub hidden_size: usize,
    pub block_count: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub head_width: usize,
    pub recurrent_layers: Vec<usize>,
    pub head_order: HeadOrder,
    pub native_serving_admitted: bool,
    pub required_conversions: Vec<&'static str>,
}
impl Qwen35Requirements {
    /// Validates recurrent tensor names and shapes for the inspected official
    /// llama.cpp layout. This does not authorize native model execution.
    pub fn inspect(index: &Index) -> Result<Self> {
        index.require_architecture("qwen35")?;
        let positive = |key: &str| -> Result<usize> {
            let value = index
                .metadata()
                .get(key)
                .ok_or_else(|| Error::Invalid(format!("missing metadata {key}")))?;
            let number = match value {
                Value::U32(n) => u64::from(*n),
                Value::U64(n) => *n,
                _ => return Err(Error::Invalid(format!("{key} must be unsigned integer"))),
            };
            usize::try_from(number)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| {
                    Error::Invalid(format!("{key} must be positive and fit address space"))
                })
        };
        let hidden_size = positive("qwen35.embedding_length")?;
        let block_count = positive("qwen35.block_count")?;
        if block_count > 4096 {
            return Err(Error::Limit("Qwen block count".into()));
        }
        if index.metadata().contains_key("qwen35.nextn_predict_layers") {
            return Err(Error::Invalid(
                "MTP/NextN blocks require a separate architecture contract".into(),
            ));
        }
        let key_heads = positive("qwen35.ssm.group_count")?;
        let value_heads = positive("qwen35.ssm.time_step_rank")?;
        let head_width = positive("qwen35.ssm.state_size")?;
        let value_dim = positive("qwen35.ssm.inner_size")?;
        let conv_kernel = positive("qwen35.ssm.conv_kernel")?;
        let head_order = HeadOrder::new(key_heads, value_heads, head_width)?;
        if value_dim != head_order.channels {
            return Err(Error::Invalid("GGUF contract requires equal K/V head widths; inner_size must equal value_heads * state_size".into()));
        }
        let key_dim = key_heads
            .checked_mul(head_width)
            .ok_or_else(|| Error::Invalid("key dimension overflow".into()))?;
        let conv_dim = key_dim
            .checked_mul(2)
            .and_then(|n| n.checked_add(value_dim))
            .ok_or_else(|| Error::Invalid("conv dimension overflow".into()))?;
        let recurrent = match index.metadata().get("qwen35.attention.recurrent_layers") {
            Some(Value::Array(flags)) if flags.len() == block_count => flags
                .iter()
                .map(|flag| match flag {
                    Value::Bool(b) => Ok(*b),
                    _ => Err(Error::Invalid(
                        "recurrent_layers must be boolean array".into(),
                    )),
                })
                .collect::<Result<Vec<_>>>()?,
            Some(_) => {
                return Err(Error::Invalid(
                    "recurrent_layers length must equal block_count".into(),
                ));
            }
            None => {
                let interval = positive("qwen35.full_attention_interval")?;
                (0..block_count)
                    .map(|layer| !(layer + 1).is_multiple_of(interval))
                    .collect()
            }
        };
        let recurrent_layers: Vec<_> = recurrent
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| r.then_some(i))
            .collect();
        if recurrent_layers.is_empty() {
            return Err(Error::Invalid(
                "qwen35 contract has no recurrent layers".into(),
            ));
        }
        for &layer in &recurrent_layers {
            for (suffix, dimensions, dense) in [
                ("attn_qkv.weight", vec![hidden_size, conv_dim], false),
                ("attn_gate.weight", vec![hidden_size, value_dim], false),
                ("ssm_beta.weight", vec![hidden_size, value_heads], false),
                ("ssm_alpha.weight", vec![hidden_size, value_heads], false),
                ("ssm_conv1d.weight", vec![conv_kernel, conv_dim], true),
                ("ssm_dt.bias", vec![value_heads], true),
                ("ssm_a", vec![value_heads], true),
                ("ssm_norm.weight", vec![head_width], true),
                ("ssm_out.weight", vec![value_dim, hidden_size], false),
            ] {
                let tensor = index.tensor(&format!("blk.{layer}.{suffix}"))?;
                if tensor.dimensions() != dimensions {
                    return Err(Error::Invalid(format!(
                        "{}: GDN shape mismatch",
                        tensor.name()
                    )));
                }
                if dense
                    && !matches!(
                        tensor.encoding(),
                        Encoding::F32 | Encoding::F16 | Encoding::Bf16
                    )
                {
                    return Err(Error::Invalid(format!(
                        "{}: recurrent vector/conv/norm must retain dense representation",
                        tensor.name()
                    )));
                }
            }
        }
        Ok(Self {
            hidden_size,
            block_count,
            key_heads,
            value_heads,
            head_width,
            recurrent_layers,
            head_order,
            native_serving_admitted: false,
            required_conversions: vec![
                "Undo or explicitly execute tiled V order for QKV V rows, z/b/a rows, dt/A vectors, V conv channels and output columns",
                "Consume ssm_a as negative exp(A_log), without applying exp a second time",
                "GGUF non-GDN RMS norm weights already include the HF +1 adjustment",
                "Bind released checkpoint revision, tokenizer, output gate activation and tied/untied output weights; architecture string alone is insufficient",
                "Validate full-attention/MRoPE layouts and numerical single/batch model gate before serving",
            ],
        })
    }
}
