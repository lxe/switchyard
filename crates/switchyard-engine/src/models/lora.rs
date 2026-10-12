//! Runtime PEFT LoRA artifacts. Base tensors are never modified or merged.
//!
//! Scaling follows PEFT's Linear adapter: alpha/r, or alpha/sqrt(r) for rsLoRA.
//! See <https://huggingface.co/docs/peft/en/package_reference/lora>.
use crate::{Error, Result};
use candle_core::{DType, Device, Tensor, safetensors::BufferedSafetensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::Path,
};

#[derive(Debug, Deserialize)]
struct Config {
    peft_type: String,
    base_model_name_or_path: String,
    r: usize,
    lora_alpha: f64,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    target_modules: serde_json::Value,
    #[serde(flatten)]
    options: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub(crate) struct Pair {
    pub a: Tensor,
    pub b: Tensor,
    pub scale: f64,
}

/// CPU-owned validated adapter. Transfer happens on the model owner thread.
#[derive(Debug)]
pub struct Adapter {
    pairs: BTreeMap<String, Pair>,
    base_model: String,
    revision: String,
    bytes: u64,
    rank: usize,
}

impl Adapter {
    /// Read the standard PEFT config and single safetensors artifact within an
    /// aggregate file and expanded-F32 tensor budget. An empty expected_base
    /// leaves alias validation to the caller; otherwise equality is required.
    pub fn read(root: &Path, max_bytes: u64, max_rank: usize, expected_base: &str) -> Result<Self> {
        if max_bytes == 0 || max_rank == 0 {
            return Err(Error::Config(
                "LoRA byte and rank bounds must be positive".into(),
            ));
        }
        let config_bytes = read_bounded(&root.join("adapter_config.json"), max_bytes.min(1 << 20))?;
        let cfg: Config = serde_json::from_slice(&config_bytes)?;
        validate_config(&cfg, max_rank)?;
        if !expected_base.is_empty() && expected_base != cfg.base_model_name_or_path {
            return Err(Error::Config(
                "LoRA base model does not match the configured model".into(),
            ));
        }
        let target_regex = cfg
            .target_modules
            .as_str()
            .filter(|s| *s != "all-linear")
            .map(|pattern| regex::Regex::new(&format!("\\A(?:{pattern})\\z")))
            .transpose()
            .map_err(|e| Error::Config(format!("Invalid LoRA target regex: {e}")))?;
        let remaining = max_bytes
            .checked_sub(config_bytes.len() as u64)
            .ok_or_else(|| Error::Config("LoRA artifact exceeds byte budget".into()))?;
        let tensor_bytes = read_bounded(&root.join("adapter_model.safetensors"), remaining)?;
        // Inline JSON transport can change whitespace/key order without changing
        // the adapter. Keep those representations in the same revision namespace.
        let canonical_config =
            serde_json::to_vec(&serde_json::from_slice::<serde_json::Value>(&config_bytes)?)?;
        let mut digest = Sha256::new();
        digest.update((canonical_config.len() as u64).to_le_bytes());
        digest.update(&canonical_config);
        digest.update((tensor_bytes.len() as u64).to_le_bytes());
        digest.update(&tensor_bytes);
        let revision = format!("{:x}", digest.finalize());
        let tensors = BufferedSafetensors::new(tensor_bytes)?;
        // Validate every name, pair and shape before any expanded tensor load.
        let mut names: BTreeMap<String, [Option<String>; 2]> = BTreeMap::new();
        let mut bytes = 0u64;
        for (name, view) in tensors.tensors() {
            let (module, slot) = tensor_name(&name)?;
            if !declared_target(&cfg.target_modules, &module, target_regex.as_ref()) {
                return Err(Error::Config(format!(
                    "LoRA tensor target is absent from target_modules: {module}"
                )));
            }
            let shape = view.shape();
            if shape.len() != 2 || shape.contains(&0) {
                return Err(Error::Config(format!(
                    "LoRA matrix must be nonempty rank two: {name}"
                )));
            }
            let rank = shape[if slot == 0 { 0 } else { 1 }];
            if rank != cfg.r || rank > max_rank {
                return Err(Error::Config(format!(
                    "LoRA rank disagrees with config or exceeds bound: {name}"
                )));
            }
            if !matches!(
                DType::try_from(view.dtype()),
                Ok(DType::F32 | DType::F16 | DType::BF16)
            ) {
                return Err(Error::Config(format!(
                    "LoRA matrix must have floating-point storage: {name}"
                )));
            }
            let count = shape[0]
                .checked_mul(shape[1])
                .and_then(|n| n.checked_mul(4))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| Error::Config("LoRA byte count overflow".into()))?;
            bytes = bytes
                .checked_add(count)
                .filter(|&n| n <= max_bytes)
                .ok_or_else(|| Error::Config("Expanded LoRA matrices exceed byte budget".into()))?;
            let pair = names.entry(module).or_default();
            if pair[slot].replace(name).is_some() {
                return Err(Error::Config("Duplicate normalized LoRA matrix".into()));
            }
        }
        if names.is_empty() {
            return Err(Error::Config(
                "LoRA artifact contains no matrix pairs".into(),
            ));
        }
        let scale = cfg.lora_alpha
            / if cfg.use_rslora {
                (cfg.r as f64).sqrt()
            } else {
                cfg.r as f64
            };
        let mut pairs = BTreeMap::new();
        for (module, [a, b]) in names {
            let (Some(a), Some(b)) = (a, b) else {
                return Err(Error::Config(format!("Incomplete LoRA A/B pair: {module}")));
            };
            let a = tensors.load(&a, &Device::Cpu)?.to_dtype(DType::F32)?;
            let b = tensors.load(&b, &Device::Cpu)?.to_dtype(DType::F32)?;
            if a.dims()[0] != b.dims()[1] {
                return Err(Error::Config(format!("LoRA A/B rank mismatch: {module}")));
            }
            for tensor in [&a, &b] {
                if tensor
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .any(|v| !v.is_finite())
                {
                    return Err(Error::Config(format!("Nonfinite LoRA weights: {module}")));
                }
            }
            pairs.insert(module, Pair { a, b, scale });
        }
        Ok(Self {
            pairs,
            base_model: cfg.base_model_name_or_path,
            revision,
            bytes,
            rank: cfg.r,
        })
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
    pub fn base_model_name(&self) -> &str {
        &self.base_model
    }
    /// Native LoRA keeps F32 matrices even when the shared base uses BF16.
    /// The argument describes base precision; it does not change branch storage.
    pub fn allocated_bytes(&self, _dtype: DType) -> Result<u64> {
        Ok(self.bytes)
    }
    pub(crate) fn pair(&self, name: &str) -> Option<&Pair> {
        self.pairs.get(name)
    }
    pub(crate) fn check_consumed(&self, consumed: &BTreeSet<String>) -> Result<()> {
        if let Some(name) = self.pairs.keys().find(|name| !consumed.contains(*name)) {
            return Err(Error::Config(format!(
                "LoRA target does not exist in this model: {name}"
            )));
        }
        Ok(())
    }
}

fn read_bounded(path: &Path, bound: u64) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.len() > bound {
        return Err(Error::Config(
            "LoRA artifact must be a bounded regular file".into(),
        ));
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(bound.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > bound {
        return Err(Error::Config(
            "LoRA artifact grew beyond byte budget".into(),
        ));
    }
    Ok(bytes)
}

fn validate_config(cfg: &Config, max_rank: usize) -> Result<()> {
    if cfg.peft_type != "LORA"
        || cfg.r == 0
        || cfg.r > max_rank
        || !cfg.lora_alpha.is_finite()
        || cfg.lora_alpha <= 0.0
        || cfg.base_model_name_or_path.is_empty()
    {
        return Err(Error::Config(
            "Invalid LoRA type, rank, alpha or base model".into(),
        ));
    }
    if !matches!(&cfg.target_modules, serde_json::Value::Array(v) if !v.is_empty() && v.iter().all(|x| x.as_str().is_some_and(|s| !s.is_empty())))
        && !cfg.target_modules.as_str().is_some_and(|s| !s.is_empty())
    {
        return Err(Error::Config(
            "LoRA target_modules requires a name list or regex".into(),
        ));
    }
    for key in [
        "use_dora",
        "fan_in_fan_out",
        "lora_bias",
        "use_qalora",
        "use_bdlora",
        "ensure_weight_tying",
    ] {
        if cfg
            .options
            .get(key)
            .is_some_and(|v| v != &serde_json::Value::Bool(false) && !v.is_null())
        {
            return Err(Error::Config(format!("Unsupported LoRA option: {key}")));
        }
    }
    if cfg
        .options
        .get("bias")
        .is_some_and(|v| v.as_str() != Some("none"))
    {
        return Err(Error::Config("LoRA bias updates are unsupported".into()));
    }
    for key in [
        "modules_to_save",
        "rank_pattern",
        "alpha_pattern",
        "layer_replication",
        "trainable_token_indices",
        "target_parameters",
        "alora_invocation_tokens",
        "arrow_config",
        "corda_config",
        "kasa_config",
        "monteclora_config",
        "velora_config",
        "loftq_config",
    ] {
        if cfg.options.get(key).is_some_and(|v| {
            !v.is_null()
                && !(v.as_array().is_some_and(Vec::is_empty)
                    || v.as_object().is_some_and(serde_json::Map::is_empty))
        }) {
            return Err(Error::Config(format!("Unsupported LoRA option: {key}")));
        }
    }
    // Initializers such as PiSSA mutate the base unless explicitly converted
    // back to ordinary LoRA. This loader accepts portable ordinary adapters.
    if cfg
        .options
        .get("init_lora_weights")
        .is_some_and(|v| !v.is_null() && !v.is_boolean() && v.as_str() != Some("gaussian"))
    {
        return Err(Error::Config(
            "LoRA initializer requires unsupported base transformation".into(),
        ));
    }
    if cfg
        .options
        .get("task_type")
        .is_some_and(|v| !v.is_null() && v.as_str() != Some("CAUSAL_LM"))
    {
        return Err(Error::Config("LoRA task must be CAUSAL_LM".into()));
    }
    if let Some(pattern) = cfg.target_modules.as_str().filter(|s| *s != "all-linear") {
        regex::Regex::new(&format!("\\A(?:{pattern})\\z"))
            .map_err(|e| Error::Config(format!("Invalid LoRA target regex: {e}")))?;
    }
    Ok(())
}
fn declared_target(
    targets: &serde_json::Value,
    module: &str,
    regex: Option<&regex::Regex>,
) -> bool {
    targets.as_str() == Some("all-linear")
        || regex.is_some_and(|regex| regex.is_match(module))
        || targets.as_array().is_some_and(|targets| {
            targets.iter().any(|target| {
                target.as_str().is_some_and(|t| {
                    module == t || module.strip_suffix(t).is_some_and(|p| p.ends_with('.'))
                })
            })
        })
}
fn tensor_name(name: &str) -> Result<(String, usize)> {
    let (module, slot) = [
        (".lora_A.weight", 0),
        (".lora_B.weight", 1),
        (".lora_A.default.weight", 0),
        (".lora_B.default.weight", 1),
    ]
    .into_iter()
    .find_map(|(suffix, slot)| name.strip_suffix(suffix).map(|module| (module, slot)))
    .ok_or_else(|| Error::Config(format!("Unsupported LoRA tensor: {name}")))?;
    let module = module.strip_prefix("base_model.model.").unwrap_or(module);
    let module = if let Some(tail) = module.strip_prefix("language_model.") {
        format!("model.language_model.{tail}")
    } else {
        module.to_owned()
    };
    let supported = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ];
    if module != "lm_head"
        && !(module.starts_with("model.language_model.layers.")
            && module
                .rsplit('.')
                .next()
                .is_some_and(|n| supported.contains(&n)))
    {
        return Err(Error::Config(format!("Unsupported LoRA target: {module}")));
    }
    Ok((module, slot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::projection::Linear;
    use candle_core::Module;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new(config: serde_json::Value, tensors: BTreeMap<String, Tensor>) -> Self {
            static ID: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "switchyard-lora-{}-{}",
                std::process::id(),
                ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            std::fs::write(
                path.join("adapter_config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            candle_core::safetensors::save(
                &tensors
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>(),
                path.join("adapter_model.safetensors"),
            )
            .unwrap();
            Self(path)
        }
        fn read(&self) -> Result<Adapter> {
            Adapter::read(&self.0, 1 << 20, 16, "test-base")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn config() -> serde_json::Value {
        serde_json::json!({"peft_type":"LORA","base_model_name_or_path":"test-base","r":2,"lora_alpha":3.,"target_modules":["q_proj"],"bias":"none"})
    }
    fn tensors() -> BTreeMap<String, Tensor> {
        [
            ("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight".into(), Tensor::new(&[[0.1f32, 0.3, -0.2], [0.4, -0.1, 0.5]], &Device::Cpu).unwrap()),
            ("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight".into(), Tensor::new(&[[0.3f32, -0.7], [-0.5, 0.9]], &Device::Cpu).unwrap()),
        ].into_iter().collect()
    }
    #[test]
    fn config_transport_formatting_does_not_change_revision() {
        let fixture = Fixture::new(config(), tensors());
        let compact = fixture.read().unwrap();
        std::fs::write(
            fixture.0.join("adapter_config.json"),
            serde_json::to_vec_pretty(&config()).unwrap(),
        )
        .unwrap();
        assert_eq!(compact.revision(), fixture.read().unwrap().revision());
    }
    #[test]
    fn lora_matches_independent_scalar_math_and_keeps_base_shared() {
        let fixture = Fixture::new(config(), tensors());
        let adapter = fixture.read().unwrap();
        let weight = Tensor::new(&[[0.2f32, -0.4, 0.3], [0.6, 0.7, -0.8]], &Device::Cpu).unwrap();
        let bias = Tensor::new(&[0.25f32, -0.75], &Device::Cpu).unwrap();
        let base = candle_nn::Linear::new(weight.clone(), Some(bias.clone()));
        let mut projection = Linear::Dense(base.clone());
        let mut consumed = BTreeSet::new();
        projection
            .attach_lora(
                "model.language_model.layers.0.self_attn.q_proj",
                &adapter,
                &mut consumed,
            )
            .unwrap();
        adapter.check_consumed(&consumed).unwrap();
        let Linear::Lora { base: shared, .. } = &projection else {
            panic!("Missing overlay");
        };
        assert_eq!(shared.weight().id(), weight.id());
        assert_eq!(shared.bias().unwrap().id(), bias.id());
        let rows = [
            [0.8f32, -0.3, 0.6],
            [0.2, 0.1, -0.9],
            [0.7, 0.4, 0.2],
            [0.3, -0.8, 0.5],
        ];
        let input = Tensor::new(&rows, &Device::Cpu)
            .unwrap()
            .reshape((2, 2, 3))
            .unwrap();
        let actual = projection
            .forward(&input)
            .unwrap()
            .reshape((4, 2))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let a = [[0.1f32, 0.3, -0.2], [0.4, -0.1, 0.5]];
        let b = [[0.3f32, -0.7], [-0.5, 0.9]];
        let w = [[0.2f32, -0.4, 0.3], [0.6, 0.7, -0.8]];
        for (row, x) in rows.iter().enumerate() {
            for output in 0..2 {
                let base = (0..3).map(|i| w[output][i] * x[i]).sum::<f32>() + [0.25, -0.75][output];
                let update = (0..2)
                    .map(|r| b[output][r] * (0..3).map(|i| a[r][i] * x[i]).sum::<f32>())
                    .sum::<f32>();
                assert!((actual[row][output] - (base + 1.5 * update)).abs() < 1e-6);
            }
        }
        assert_eq!(weight.to_vec2::<f32>().unwrap(), w);
        assert_eq!(adapter.bytes(), 40);
        assert_eq!(adapter.allocated_bytes(DType::BF16).unwrap(), 40);
        assert_eq!(adapter.rank(), 2);
        assert_eq!(adapter.revision(), fixture.read().unwrap().revision());
    }
    #[test]
    fn regex_and_rslora_scaling_are_supported() {
        let mut cfg = config();
        cfg["target_modules"] = serde_json::json!(".*language_model.*\\.self_attn\\.q_proj");
        cfg["use_rslora"] = serde_json::json!(true);
        let fixture = Fixture::new(cfg, tensors());
        let adapter = fixture.read().unwrap();
        assert!((adapter.pairs.values().next().unwrap().scale - 3. / 2f64.sqrt()).abs() < 1e-12);
    }
    #[test]
    fn bf16_base_uses_f32_adapter_branch_then_casts_the_sum() {
        let adapter = Fixture::new(config(), tensors()).read().unwrap();
        let weight = Tensor::new(
            &[[0.211f32, -0.402, 0.303], [0.601, 0.702, -0.803]],
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let base = candle_nn::Linear::new(weight, None);
        let mut projection = Linear::Dense(base.clone());
        projection
            .attach_lora(
                "model.language_model.layers.0.self_attn.q_proj",
                &adapter,
                &mut BTreeSet::new(),
            )
            .unwrap();
        let input = Tensor::new(
            &[[0.811f32, -0.321, 0.617], [0.235, 0.123, -0.917]],
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let xs = input
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let base_values = base
            .forward(&input)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let a = [[0.1f32, 0.3, -0.2], [0.4, -0.1, 0.5]];
        let b = [[0.3f32, -0.7], [-0.5, 0.9]];
        let expected = (0..2)
            .flat_map(|row| (0..2).map(move |output| (row, output)))
            .map(|(row, output)| {
                base_values[row][output]
                    + 1.5
                        * (0..2)
                            .map(|r| {
                                b[output][r] * (0..3).map(|i| a[r][i] * xs[row][i]).sum::<f32>()
                            })
                            .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let expected = Tensor::from_vec(expected, (2, 2), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let result = projection.forward(&input).unwrap();
        assert_eq!(result.dtype(), DType::BF16);
        assert_eq!(
            result
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            expected.to_vec2::<f32>().unwrap()
        );
        let Linear::Lora { a, b, .. } = projection else {
            panic!("Missing overlay");
        };
        assert_eq!(a.weight().dtype(), DType::F32);
        assert_eq!(b.weight().dtype(), DType::F32);
    }

    #[test]
    fn every_hybrid_model_projection_matches_offline_scalar_merge_and_base_isolation() {
        use crate::models::{MixedModelSession, QwenMixedSession, qwen35::ModelForCausalLM};
        use candle_nn::{VarBuilder, VarMap};
        let cfg: crate::models::qwen35::Config =
            serde_json::from_value(serde_json::json!({"text_config":{
                "vocab_size":32,"hidden_size":16,"intermediate_size":24,
                "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":8,"max_position_embeddings":16,"rms_norm_eps":1e-6,
                "rope_parameters":{"rope_theta":10000.,"partial_rotary_factor":0.5},
                "attention_bias":false,"hidden_act":"silu",
                "layer_types":["linear_attention","full_attention"],
                "linear_num_key_heads":2,"linear_num_value_heads":4,
                "linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":4
            }}))
            .unwrap();
        let vars = VarMap::new();
        ModelForCausalLM::new(
            &cfg,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let mut base_tensors = vars
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, tensor)| {
                let values = (0..tensor.elem_count())
                    .map(|i| (i as f32 * 0.17).sin() * 0.03)
                    .collect::<Vec<_>>();
                (
                    name.clone(),
                    Tensor::from_vec(values, tensor.shape(), &Device::Cpu).unwrap(),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        base_tensors.insert(
            "lm_head.weight".into(),
            base_tensors["model.language_model.embed_tokens.weight"].clone(),
        );
        let base = ModelForCausalLM::new(
            &cfg,
            VarBuilder::from_tensors(base_tensors.clone(), DType::F32, &Device::Cpu),
        )
        .unwrap();
        let mut merged_tensors = base_tensors.clone();
        let mut adapter_tensors = BTreeMap::new();
        let mut target_count = 0;
        for (name, tensor) in &base_tensors {
            let Some(module) = name.strip_suffix(".weight") else {
                continue;
            };
            if tensor_name(&format!("{module}.lora_A.weight")).is_err() {
                continue;
            }
            let (output, input) = tensor.dims2().unwrap();
            let a = (0..2 * input)
                .map(|i| ((i + 3) as f32 * 0.13).cos() * 0.02)
                .collect::<Vec<_>>();
            let b = (0..output * 2)
                .map(|i| ((i + 7) as f32 * 0.31).sin() * 0.02)
                .collect::<Vec<_>>();
            adapter_tensors.insert(
                format!("base_model.model.{module}.lora_A.weight"),
                Tensor::from_vec(a.clone(), (2, input), &Device::Cpu).unwrap(),
            );
            adapter_tensors.insert(
                format!("base_model.model.{module}.lora_B.weight"),
                Tensor::from_vec(b.clone(), (output, 2), &Device::Cpu).unwrap(),
            );
            // Independent, test-only scalar construction of W + scale*BA.
            let mut values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for o in 0..output {
                for i in 0..input {
                    values[o * input + i] +=
                        1.5 * (0..2).map(|r| b[o * 2 + r] * a[r * input + i]).sum::<f32>();
                }
            }
            merged_tensors.insert(
                name.clone(),
                Tensor::from_vec(values, (output, input), &Device::Cpu).unwrap(),
            );
            target_count += 1;
        }
        assert_eq!(target_count, 16);
        let mut config = config();
        config["target_modules"] = serde_json::json!("all-linear");
        let adapter = Fixture::new(config, adapter_tensors).read().unwrap();
        let adapted = base.with_lora(&adapter).unwrap();
        assert!(adapted.with_lora(&adapter).is_err());
        let merged = ModelForCausalLM::new(
            &cfg,
            VarBuilder::from_tensors(merged_tensors, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let (
            Linear::Dense(original_head),
            Linear::Lora {
                base: adapted_head, ..
            },
        ) = (&base.lm_head, &adapted.lm_head)
        else {
            panic!("Missing output overlay");
        };
        assert_eq!(original_head.weight().id(), adapted_head.weight().id());
        let mut base_before = base.clone();
        let tokens = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &Device::Cpu).unwrap();
        let original = base_before.forward(&tokens, 0).unwrap();
        let base_states = base_before.base_model.sequence_states().unwrap();
        assert!(
            adapted
                .clone()
                .base_model
                .restore_sequences(&base_states.iter().collect::<Vec<_>>())
                .is_err()
        );
        let mut actual = adapted.clone();
        let mut expected = merged.clone();
        close(
            &actual.forward(&tokens, 0).unwrap(),
            &expected.forward(&tokens, 0).unwrap(),
        );
        let next = Tensor::new(&[[7u32], [8]], &Device::Cpu).unwrap();
        close(
            &actual.forward(&next, 3).unwrap(),
            &expected.forward(&next, 3).unwrap(),
        );
        let mut base_after = base.clone();
        close(&original, &base_after.forward(&tokens, 0).unwrap());
        assert!(
            (original - adapted.clone().forward(&tokens, 0).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                > 1e-7
        );
        // A populated base cache cannot satisfy an adapter's first request;
        // separate adapters also receive independent cache retention.
        use crate::models::{ModelFactory, PrefixCache, QwenFactory};
        let cache_config = crate::config::CacheConfig {
            page_tokens: 2,
            checkpoint_tokens: 2,
            max_bytes: 1 << 20,
            max_entries: 8,
        };
        let telemetry = crate::telemetry::Telemetry::default();
        let factory = QwenFactory {
            weights: base.clone(),
            device: Device::Cpu,
            vocab: 32,
            eos: 31,
            cache: PrefixCache::new(cache_config.clone(), telemetry.clone()).unwrap(),
            vision: None,
            cache_config,
            telemetry,
        };
        factory.session().forward(&tokens, 0).unwrap();
        assert!(factory.prefix_cache_stats().unwrap().unwrap().entries > 0);
        let first = factory.with_lora(&adapter).unwrap();
        let second = factory.with_lora(&adapter).unwrap();
        assert_eq!(first.prefix_cache_stats().unwrap().unwrap().entries, 0);
        close(
            &first.session().forward(&tokens, 0).unwrap(),
            &merged
                .clone()
                .forward(&tokens, 0)
                .unwrap()
                .squeeze(1)
                .unwrap(),
        );
        assert_eq!(first.prefix_cache_stats().unwrap().unwrap().hit_rows, 0);
        assert_eq!(second.prefix_cache_stats().unwrap().unwrap().entries, 0);
        first.session().forward(&tokens, 0).unwrap();
        assert!(first.prefix_cache_stats().unwrap().unwrap().hit_rows > 0);
        let mut actual = QwenMixedSession::new(adapted, 2).unwrap();
        let mut expected = QwenMixedSession::new(merged, 2).unwrap();
        for (row, prompt) in [(0, vec![1u32, 2]), (1, vec![4u32, 5, 6])] {
            let tokens = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &Device::Cpu).unwrap();
            close(
                &actual.prefill(&[row], &tokens).unwrap(),
                &expected.prefill(&[row], &tokens).unwrap(),
            );
        }
        close(
            &actual.decode(&[0, 1], &next, &[2, 3]).unwrap(),
            &expected.decode(&[0, 1], &next, &[2, 3]).unwrap(),
        );
    }
    fn close(a: &Tensor, b: &Tensor) {
        let error = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 2e-5, "Maximum model logit error: {error}");
    }
    #[test]
    fn malformed_or_unsupported_artifacts_fail_explicitly() {
        for (key, value) in [
            ("use_dora", serde_json::json!(true)),
            ("bias", serde_json::json!("all")),
            ("modules_to_save", serde_json::json!(["lm_head"])),
            ("rank_pattern", serde_json::json!({"q_proj":4})),
            ("target_modules", serde_json::json!("[")),
            ("fan_in_fan_out", serde_json::json!(true)),
            ("r", serde_json::json!(17)),
        ] {
            let mut cfg = config();
            cfg[key] = value;
            assert!(Fixture::new(cfg, tensors()).read().is_err(), "{key}");
        }
        let fixture = Fixture::new(config(), tensors());
        assert!(Adapter::read(&fixture.0, 20, 16, "test-base").is_err());
        assert!(Adapter::read(&fixture.0, 1 << 20, 1, "test-base").is_err());
        assert!(Adapter::read(&fixture.0, 1 << 20, 16, "wrong-base").is_err());
        let mut incomplete = tensors();
        incomplete.pop_last();
        assert!(Fixture::new(config(), incomplete).read().is_err());
        let mut duplicate = tensors();
        duplicate.insert(
            "model.language_model.layers.0.self_attn.q_proj.lora_A.weight".into(),
            Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(Fixture::new(config(), duplicate).read().is_err());
        let mut unsupported = tensors();
        unsupported.insert(
            "model.visual.foo.lora_A.weight".into(),
            Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(Fixture::new(config(), unsupported).read().is_err());
        let mut nonfinite = tensors();
        *nonfinite.values_mut().next().unwrap() =
            Tensor::new(&[[f32::NAN, 0., 0.], [0., 0., 0.]], &Device::Cpu).unwrap();
        assert!(Fixture::new(config(), nonfinite).read().is_err());
    }
    #[test]
    fn stale_targets_and_wrong_projection_shapes_do_not_partially_modify_base() {
        let adapter = Fixture::new(config(), tensors()).read().unwrap();
        assert!(adapter.check_consumed(&BTreeSet::new()).is_err());
        let weight = Tensor::zeros((4, 3), DType::F32, &Device::Cpu).unwrap();
        let mut projection = Linear::Dense(candle_nn::Linear::new(weight.clone(), None));
        let mut consumed = BTreeSet::new();
        assert!(
            projection
                .attach_lora(
                    "model.language_model.layers.0.self_attn.q_proj",
                    &adapter,
                    &mut consumed
                )
                .is_err()
        );
        let Linear::Dense(base) = projection else {
            panic!("Base unexpectedly modified");
        };
        assert_eq!(base.weight().id(), weight.id());
        assert!(consumed.is_empty());
    }
    #[test]
    #[ignore = "Requires existing local Pocket Model training artifacts"]
    fn actual_pocket_model_artifacts_load() {
        for (path, base) in [
            (
                "/home/lxe/pm-dogfood-data/v9/run1-adapter/peft",
                "Qwen/Qwen3.5-0.8B",
            ),
            (
                "/home/lxe/pm-dogfood-data/v9/agent/run2-peft/peft",
                "Qwen/Qwen3.5-2B",
            ),
        ] {
            let adapter = Adapter::read(Path::new(path), 128 << 20, 16, base).unwrap();
            assert_eq!(adapter.pairs.len(), 150);
            assert_eq!(adapter.rank(), 16);
            eprintln!(
                "{base}: {} bytes, revision {}",
                adapter.bytes(),
                adapter.revision()
            );
        }
    }
}
