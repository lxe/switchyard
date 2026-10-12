//! llama.cpp Qwen3.5 GGUF layout adapter, sharing native model arithmetic.
//! The converter changes V head order, folds RMS +1 and stores negative exp(A).
//! Restore those semantics on activations/small tensors; never expand projection weights.
use super::{
    projection::{Linear, ModelWeights},
    qwen35,
};
use crate::{Error, Result, telemetry::Telemetry};
use candle_core::{
    DType, Device, Shape, Tensor,
    quantized::{QMatMul, ggml_file::qtensor_from_ggml, gguf_file::Value},
};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::{Arc, Mutex},
};
use switchyard_quantized::{
    gdn::{HeadOrder, Qwen35Requirements},
    gguf::{Index, IndexLimits},
};

pub(crate) struct Source {
    file: Mutex<File>,
    index: Index,
    key_heads: usize,
    value_heads: usize,
    head_width: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Permutation {
    Qkv,
    Value,
    Heads,
    Output,
}
impl Source {
    pub fn open(
        path: &Path,
        max_bytes: u64,
        cfg: &qwen35::Config,
        telemetry: &Telemetry,
    ) -> Result<Arc<Self>> {
        let mut file = File::open(path)?;
        let bytes = file.metadata()?.len();
        if bytes > max_bytes {
            return Err(Error::Config(
                "GGUF exceeds explicit packed weight byte budget".into(),
            ));
        }
        telemetry.event(
            "weights_read_started",
            serde_json::json!({"format":"gguf","files":1,"bytes":bytes}),
        );
        let index = Index::read(&mut file, IndexLimits::default()).map_err(convert)?;
        let req = Qwen35Requirements::inspect(&index).map_err(convert)?;
        let c = &cfg.text_config;
        if req.hidden_size != c.hidden_size
            || req.block_count != c.num_hidden_layers
            || req.key_heads != c.linear_num_key_heads
            || req.value_heads != c.linear_num_value_heads
            || req.head_width != c.linear_key_head_dim
            || req.head_width != c.linear_value_head_dim
            || req.recurrent_layers
                != c.layer_types
                    .iter()
                    .enumerate()
                    .filter_map(|(i, t)| (t == "linear_attention").then_some(i))
                    .collect::<Vec<_>>()
        {
            return Err(Error::Config(
                "GGUF metadata does not match pinned HF Qwen3.5 text config".into(),
            ));
        }
        let integer = |key: &str, expected: usize| -> Result<()> {
            let actual = match index.metadata().get(key) {
                Some(Value::U32(v)) => *v as usize,
                Some(Value::U64(v)) => {
                    usize::try_from(*v).map_err(|_| Error::Config(format!("Invalid {key}")))?
                }
                _ => return Err(Error::Config(format!("Missing GGUF metadata {key}"))),
            };
            if actual != expected {
                return Err(Error::Config(format!("GGUF {key} differs from HF config")));
            }
            Ok(())
        };
        integer("qwen35.feed_forward_length", c.intermediate_size)?;
        integer("qwen35.attention.head_count", c.num_attention_heads)?;
        integer("qwen35.attention.head_count_kv", c.num_key_value_heads)?;
        integer("qwen35.attention.key_length", cfg.head_dim())?;
        integer("qwen35.ssm.conv_kernel", c.linear_conv_kernel_dim)?;
        integer(
            "qwen35.rope.dimension_count",
            (cfg.head_dim() as f64 * c.rope_parameters.partial_rotary_factor) as usize,
        )?;
        let float = |key: &str, expected: f64| -> Result<()> {
            let actual = match index.metadata().get(key) {
                Some(Value::F32(v)) => f64::from(*v),
                Some(Value::F64(v)) => *v,
                _ => return Err(Error::Config(format!("Missing GGUF metadata {key}"))),
            };
            if !actual.is_finite() || (actual - expected).abs() > expected.abs() * 1e-6 {
                return Err(Error::Config(format!("GGUF {key} differs from HF config")));
            }
            Ok(())
        };
        float("qwen35.attention.layer_norm_rms_epsilon", c.rms_norm_eps)?;
        float("qwen35.rope.freq_base", c.rope_parameters.rope_theta)?;
        if c.hidden_act != candle_nn::Activation::Silu {
            return Err(Error::Config(
                "Qwen3.5 GGUF requires the published SiLU feed-forward activation".into(),
            ));
        }
        if c.attention_bias {
            return Err(Error::Config(
                "GGUF biased Qwen3.5 projections unsupported".into(),
            ));
        }
        // Hadamard-folded variants require their own activation transform contract.
        if index.metadata().keys().any(|k| k.contains("hadamard")) {
            return Err(Error::Config("Hadamard-folded GGUF is unsupported".into()));
        }
        let emb = index.tensor("token_embd.weight").map_err(convert)?;
        if emb.dimensions() != [c.hidden_size, c.vocab_size] {
            return Err(Error::Config("GGUF embedding dimensions mismatch".into()));
        }
        telemetry.event("weights_indexed", serde_json::json!({"format":"gguf","bytes":bytes,"tensors":index.tensors().count(),"projection_storage":"packed","embedding_storage":"packed"}));
        Ok(Arc::new(Self {
            file: Mutex::new(file),
            index,
            key_heads: req.key_heads,
            value_heads: req.value_heads,
            head_width: req.head_width,
        }))
    }
    pub fn eos_token(&self) -> Result<u32> {
        let eos = match self.index.metadata().get("tokenizer.ggml.eos_token_id") {
            Some(Value::U32(value)) => *value,
            Some(Value::U64(value)) => u32::try_from(*value)
                .map_err(|_| Error::Config("GGUF EOS token exceeds u32".into()))?,
            _ => {
                return Err(Error::Config(
                    "GGUF requires tokenizer.ggml.eos_token_id".into(),
                ));
            }
        };
        let vocabulary = self
            .index
            .tensor("token_embd.weight")
            .map_err(convert)?
            .dimensions()[1];
        if eos as usize >= vocabulary {
            return Err(Error::Config("GGUF EOS token exceeds vocabulary".into()));
        }
        Ok(eos)
    }
    pub fn builder(self: &Arc<Self>, device: &Device) -> ModelWeights<'static> {
        ModelWeights::gguf(
            VarBuilder::from_backend(Box::new(Backend(self.clone())), DType::F32, device.clone()),
            self.clone(),
        )
    }
    fn read(
        &self,
        name: &str,
        device: &Device,
    ) -> candle_core::Result<candle_core::quantized::QTensor> {
        let desc = self.index.tensor(name).map_err(tensor_error)?;
        let mut file = self
            .file
            .lock()
            .map_err(|_| candle_core::Error::Msg("GGUF reader poisoned".into()))?;
        file.seek(SeekFrom::Start(desc.file_offset()))?;
        let mut packed = vec![
            0u8;
            usize::try_from(desc.bytes()).map_err(|_| candle_core::Error::Msg(
                "GGUF tensor address space".into()
            ))?
        ];
        file.read_exact(&mut packed)?;
        qtensor_from_ggml(
            desc.encoding().backend(),
            &packed,
            desc.backend_shape(),
            device,
        )
    }
    pub fn projection_name(
        &self,
        path: &str,
    ) -> candle_core::Result<(String, Option<Permutation>)> {
        let (name, _) = map_name(&format!("{path}.weight"))?;
        let permutation = if path.ends_with("linear_attn.in_proj_qkv") {
            Some(Permutation::Qkv)
        } else if path.ends_with("linear_attn.in_proj_z") {
            Some(Permutation::Value)
        } else if path.ends_with("linear_attn.in_proj_a") || path.ends_with("linear_attn.in_proj_b")
        {
            Some(Permutation::Heads)
        } else if path.ends_with("linear_attn.out_proj") {
            Some(Permutation::Output)
        } else {
            None
        };
        Ok((name, permutation))
    }
    pub fn projection(
        &self,
        name: &str,
        input: usize,
        output: usize,
        permutation: Option<Permutation>,
        device: &Device,
    ) -> candle_core::Result<Linear> {
        let desc = self.index.tensor(name).map_err(tensor_error)?;
        if desc.dimensions() != [input, output] {
            candle_core::bail!("{name}: GGUF projection dimensions mismatch");
        }
        let weight = self.read(name, device)?;
        // The explicit QTensor variant prevents CANDLE_DEQUANTIZE_ALL from
        // expanding quantized matrices. Native F32/F16 tensors remain dense.
        let operator = if desc.encoding().quantized() {
            QMatMul::QTensor(Arc::new(weight))
        } else {
            QMatMul::Tensor(weight.dequantize(device)?)
        };
        let order = permutation.map(|p| self.order(p, device)).transpose()?;
        let (input_order, output_order) = if matches!(permutation, Some(Permutation::Output)) {
            (order, None)
        } else {
            (None, order)
        };
        Ok(Linear::Packed {
            operator,
            input_order,
            output_order,
        })
    }
    fn order(&self, p: Permutation, device: &Device) -> candle_core::Result<Tensor> {
        let width = if matches!(p, Permutation::Heads) {
            1
        } else {
            self.head_width
        };
        let heads =
            HeadOrder::new(self.key_heads, self.value_heads, width).map_err(tensor_error)?;
        let vdim = self.value_heads * width;
        let qk = if matches!(p, Permutation::Qkv) {
            2 * self.key_heads * self.head_width
        } else {
            0
        };
        let mut indices = (0..qk).map(|i| i as u32).collect::<Vec<_>>();
        for i in 0..vdim {
            let source = if matches!(p, Permutation::Output) {
                heads.tiled_to_grouped(i)
            } else {
                heads.grouped_to_tiled(i)
            }
            .map_err(tensor_error)?;
            indices.push(
                u32::try_from(qk + source).map_err(|_| {
                    candle_core::Error::Msg("GGUF permutation index overflow".into())
                })?,
            );
        }
        let length = indices.len();
        Tensor::from_vec(indices, length, device)
    }
    fn dense(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let (mapped, transform) = map_name(name)?;
        let mut tensor = self.read(&mapped, device)?.dequantize(device)?;
        match transform {
            DenseTransform::Conv => {
                let order = self.order(Permutation::Qkv, device)?;
                tensor = tensor.index_select(&order, 0)?.unsqueeze(1)?;
            }
            DenseTransform::Heads => {
                let order = self.order(Permutation::Heads, device)?;
                tensor = tensor.index_select(&order, 0)?;
            }
            DenseTransform::None => {}
        }
        tensor.to_dtype(dtype)
    }
}
fn convert(error: switchyard_quantized::Error) -> Error {
    Error::Config(error.to_string())
}
fn tensor_error(error: switchyard_quantized::Error) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}
#[derive(Clone, Copy)]
enum DenseTransform {
    None,
    Conv,
    Heads,
}
fn map_name(name: &str) -> candle_core::Result<(String, DenseTransform)> {
    match name {
        "model.language_model.embed_tokens.weight" => {
            return Ok(("token_embd.weight".into(), DenseTransform::None));
        }
        "model.language_model.norm.weight" => {
            return Ok(("output_norm.weight".into(), DenseTransform::None));
        }
        "lm_head.weight" => return Ok(("output.weight".into(), DenseTransform::None)),
        _ => {}
    }
    let path = name
        .strip_prefix("model.language_model.layers.")
        .ok_or_else(|| candle_core::Error::Msg(format!("Unsupported GGUF auxiliary {name}")))?;
    let (layer, suffix) = path
        .split_once('.')
        .ok_or_else(|| candle_core::Error::Msg(format!("Invalid GGUF layer name {name}")))?;
    let _: usize = layer
        .parse()
        .map_err(|_| candle_core::Error::Msg("Invalid GGUF layer index".into()))?;
    let (suffix, transform) = match suffix {
        "input_layernorm.weight" => ("attn_norm.weight", DenseTransform::None),
        "post_attention_layernorm.weight" => ("post_attention_norm.weight", DenseTransform::None),
        "self_attn.q_norm.weight" => ("attn_q_norm.weight", DenseTransform::None),
        "self_attn.k_norm.weight" => ("attn_k_norm.weight", DenseTransform::None),
        "self_attn.q_proj.weight" => ("attn_q.weight", DenseTransform::None),
        "self_attn.k_proj.weight" => ("attn_k.weight", DenseTransform::None),
        "self_attn.v_proj.weight" => ("attn_v.weight", DenseTransform::None),
        "self_attn.o_proj.weight" => ("attn_output.weight", DenseTransform::None),
        "mlp.gate_proj.weight" => ("ffn_gate.weight", DenseTransform::None),
        "mlp.up_proj.weight" => ("ffn_up.weight", DenseTransform::None),
        "mlp.down_proj.weight" => ("ffn_down.weight", DenseTransform::None),
        "linear_attn.in_proj_qkv.weight" => ("attn_qkv.weight", DenseTransform::None),
        "linear_attn.in_proj_z.weight" => ("attn_gate.weight", DenseTransform::None),
        "linear_attn.in_proj_b.weight" => ("ssm_beta.weight", DenseTransform::None),
        "linear_attn.in_proj_a.weight" => ("ssm_alpha.weight", DenseTransform::None),
        "linear_attn.out_proj.weight" => ("ssm_out.weight", DenseTransform::None),
        "linear_attn.conv1d.weight" => ("ssm_conv1d.weight", DenseTransform::Conv),
        "linear_attn.dt_bias" => ("ssm_dt.bias", DenseTransform::Heads),
        "linear_attn.A_log" => ("ssm_a", DenseTransform::Heads),
        "linear_attn.norm.weight" => ("ssm_norm.weight", DenseTransform::None),
        _ => candle_core::bail!("Unsupported Qwen3.5 GGUF tensor {name}"),
    };
    Ok((format!("blk.{layer}.{suffix}"), transform))
}
struct Backend(Arc<Source>);
impl SimpleBackend for Backend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hints: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!("GGUF {name}: expected {shape:?}, got {:?}", tensor.shape());
        }
        Ok(tensor)
    }
    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        if name.ends_with(".weight")
            && !name.contains("norm.weight")
            && !name.ends_with("embed_tokens.weight")
            && !name.ends_with("conv1d.weight")
        {
            candle_core::bail!("GGUF projection {name} must execute packed");
        }
        self.0.dense(name, dtype, device)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        map_name(name)
            .ok()
            .is_some_and(|(mapped, _)| self.0.index.tensor(&mapped).is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MixedModelSession;
    use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
    use std::collections::BTreeMap;

    #[test]
    fn layout_mapping_rejects_unrelated_names() {
        assert_eq!(
            map_name("model.language_model.layers.17.linear_attn.A_log")
                .unwrap()
                .0,
            "blk.17.ssm_a"
        );
        assert_eq!(
            map_name("model.language_model.layers.3.self_attn.q_proj.weight")
                .unwrap()
                .0,
            "blk.3.attn_q.weight"
        );
        assert!(map_name("model.visual.weight").is_err());
        assert!(map_name("model.language_model.layers.bad.mlp.gate_proj.weight").is_err());
    }
    #[test]
    fn complete_packed_model_prefill_decode_and_mixed_rows() -> Result<()> {
        // Independent GGUF fixture has a 2:1 V/K ratio, folded norms and negative A.
        // Both recurrent and attention layers execute packed Q4_0 matrices.
        let cfg: qwen35::Config = serde_json::from_value(serde_json::json!({"text_config":{
            "vocab_size":32,"hidden_size":32,"intermediate_size":32,"num_hidden_layers":2,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,
            "max_position_embeddings":64,"rms_norm_eps":1e-6,
            "rope_parameters":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
            "attention_bias":false,"hidden_act":"silu","layer_types":["linear_attention","full_attention"],
            "linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,
            "linear_value_head_dim":16,"linear_conv_kernel_dim":4
        }}))?;
        let mut tensors = BTreeMap::<String, QTensor>::new();
        let mut add = |name: &str,
                       shape: &[usize],
                       dense: bool,
                       value: Option<f32>|
         -> candle_core::Result<()> {
            let count: usize = shape.iter().product();
            let seed = tensors.len();
            let vals = (0..count)
                .map(|i| value.unwrap_or(((i + seed * 7) as f32 * 0.031).sin() * 0.03))
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(vals, shape, &Device::Cpu)?;
            let quant = QTensor::quantize(
                &tensor,
                if dense {
                    GgmlDType::F32
                } else {
                    GgmlDType::Q4_0
                },
            )?;
            tensors.insert(name.into(), quant);
            Ok(())
        };
        add("token_embd.weight", &[32, 32], false, None)?;
        add("output_norm.weight", &[32], true, Some(1.0))?;
        for layer in 0..2 {
            for suffix in ["attn_norm.weight", "post_attention_norm.weight"] {
                add(&format!("blk.{layer}.{suffix}"), &[32], true, Some(1.0))?;
            }
            for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                add(&format!("blk.{layer}.{suffix}"), &[32, 32], false, None)?;
            }
        }
        for (suffix, shape, dense, value) in [
            ("attn_qkv.weight", vec![128, 32], false, None),
            ("attn_gate.weight", vec![64, 32], false, None),
            ("ssm_beta.weight", vec![4, 32], false, None),
            ("ssm_alpha.weight", vec![4, 32], false, None),
            ("ssm_conv1d.weight", vec![128, 4], true, None),
            ("ssm_dt.bias", vec![4], true, Some(-1.0)),
            ("ssm_a", vec![4], true, Some(-0.5)),
            ("ssm_norm.weight", vec![16], true, Some(1.0)),
            ("ssm_out.weight", vec![32, 64], false, None),
        ] {
            add(&format!("blk.0.{suffix}"), &shape, dense, value)?;
        }
        for (suffix, shape, dense) in [
            ("attn_q.weight", vec![64, 32], false),
            ("attn_k.weight", vec![16, 32], false),
            ("attn_v.weight", vec![16, 32], false),
            ("attn_output.weight", vec![32, 32], false),
            ("attn_q_norm.weight", vec![16], true),
            ("attn_k_norm.weight", vec![16], true),
        ] {
            add(
                &format!("blk.1.{suffix}"),
                &shape,
                dense,
                if dense { Some(1.0) } else { None },
            )?;
        }
        let mut metadata = vec![
            ("general.architecture", Value::String("qwen35".into())),
            ("tokenizer.ggml.eos_token_id", Value::U32(2)),
        ];
        for (k, v) in [
            ("block_count", 2),
            ("embedding_length", 32),
            ("feed_forward_length", 32),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 16),
            ("ssm.group_count", 2),
            ("ssm.time_step_rank", 4),
            ("ssm.state_size", 16),
            ("ssm.inner_size", 64),
            ("ssm.conv_kernel", 4),
            ("full_attention_interval", 2),
            ("rope.dimension_count", 16),
        ] {
            metadata.push((
                Box::leak(format!("qwen35.{k}").into_boxed_str()),
                Value::U32(v),
            ));
        }
        metadata.push(("qwen35.rope.freq_base", Value::F32(10000.0)));
        metadata.push(("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)));
        let mut bytes = std::io::Cursor::new(Vec::new());
        gguf_file::write(
            &mut bytes,
            &metadata.iter().map(|(n, v)| (*n, v)).collect::<Vec<_>>(),
            &tensors
                .iter()
                .map(|(n, t)| (n.as_str(), t))
                .collect::<Vec<_>>(),
        )?;
        let mut bytes = bytes.into_inner();
        bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
        let path = std::env::temp_dir().join(format!(
            "switchyard-gguf-fixture-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes)?;
        let source = Source::open(&path, 1 << 20, &cfg, &Telemetry::default())?;
        std::fs::remove_file(&path)?;
        assert_eq!(source.eos_token()?, 2);
        let neg = source
            .builder(&Device::Cpu)
            .pp("model.language_model.layers.0.linear_attn")
            .negative_a(4)?
            .to_vec1::<f32>()?;
        assert_eq!(neg, [-0.5; 4]);
        let norm = source
            .builder(&Device::Cpu)
            .pp("model.language_model.norm")
            .norm_weight(32)?
            .to_vec1::<f32>()?;
        assert_eq!(norm, [1.; 32]);
        assert!(
            source
                .builder(&Device::Cpu)
                .pp("model.language_model.layers.0.mlp.gate_proj")
                .get((32, 32), "weight")
                .is_err()
        );
        let order = source
            .order(Permutation::Heads, &Device::Cpu)?
            .to_vec1::<u32>()?;
        assert_eq!(order, [0, 2, 1, 3]);
        let mut one =
            qwen35::ModelForCausalLM::new_with_weights(&cfg, source.builder(&Device::Cpu))?;
        let mut batch = one.clone();
        let logits = one.forward(&Tensor::new(&[[1u32, 2, 3]], &Device::Cpu)?, 0)?;
        let doubled = batch.forward(&Tensor::new(&[[1u32, 2, 3], [1, 2, 3]], &Device::Cpu)?, 0)?;
        assert_eq!(logits.dims(), [1, 1, 32]);
        let difference = (logits.broadcast_sub(&doubled)?.abs()?.max_all()?).to_scalar::<f32>()?;
        assert!(difference < 1e-5, "batch difference {difference}");
        let decode = one.forward(&Tensor::new(&[[4u32]], &Device::Cpu)?, 3)?;
        assert!(
            decode
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        );
        let mut mixed = super::super::QwenMixedSession::new(batch, 2)?;
        mixed.prefill(&[0], &Tensor::new(&[[1u32, 2, 3]], &Device::Cpu)?)?;
        mixed.prefill(&[1], &Tensor::new(&[[1u32, 2]], &Device::Cpu)?)?;
        let out = mixed.decode(
            &[0, 1],
            &Tensor::new(&[[4u32], [3]], &Device::Cpu)?,
            &[3, 2],
        )?;
        assert_eq!(out.dims(), [2, 32]);
        Ok(())
    }
}
