//! Safe, name-addressed native tensor loading, independent of model execution.
//! Buffers have an explicit aggregate byte limit. No serve-time content hashing.
use crate::{Error, Result, telemetry::Telemetry};
use candle_core::{DType, Device, Shape, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};
use std::{collections::HashMap, path::Path};

pub struct Weights {
    shards: Vec<BufferedSafetensors>,
    names: HashMap<String, usize>,
}
impl Weights {
    pub fn read(root: &Path, max_bytes: u64, telemetry: &Telemetry) -> Result<Self> {
        let mut files = std::fs::read_dir(root)?
            .map(|e| e.map(|x| x.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        files.retain(|p| p.extension().is_some_and(|x| x == "safetensors"));
        files.sort();
        if files.is_empty() {
            return Err(Error::Config("No safetensors weights found".into()));
        }
        let mut total = 0u64;
        for file in &files {
            total = total
                .checked_add(file.metadata()?.len())
                .ok_or_else(|| Error::Config("Weight byte count overflow".into()))?;
        }
        if total > max_bytes {
            return Err(Error::Config(
                "Weights exceed explicit host buffer budget".into(),
            ));
        }
        telemetry.event(
            "weights_read_started",
            serde_json::json!({"files": files.len(),"bytes":total}),
        );
        let mut shards = Vec::new();
        let mut names = HashMap::new();
        for file in files {
            let shard = BufferedSafetensors::new(std::fs::read(file)?)?;
            for (name, _) in shard.tensors() {
                if names.insert(name.clone(), shards.len()).is_some() {
                    return Err(Error::Config(format!("Duplicate weight: {name}")));
                }
            }
            shards.push(shard);
        }
        telemetry.event(
            "weights_read_finished",
            serde_json::json!({"bytes":total,"tensors":names.len()}),
        );
        Ok(Self { shards, names })
    }
    fn tensor(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let index = self
            .names
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing weight {name}")))?;
        self.shards[*index].load(name, device)?.to_dtype(dtype)
    }
    pub fn builder(self, dtype: DType, device: &Device) -> VarBuilder<'static> {
        VarBuilder::from_backend(Box::new(self), dtype, device.clone())
    }
    /// Bake a text-only native tensor file. Runtime requires no HF name mapping
    /// or conversion; vision tensors are deliberately outside this adapter.
    pub fn prepare(&self, source: &Path, output: &Path) -> Result<()> {
        if output.exists() {
            return Err(Error::Config("Preparation output must be fresh".into()));
        }
        std::fs::create_dir(output)?;
        let mut tensors = HashMap::new();
        for (name, &index) in &self.names {
            if name.starts_with("model.language_model.") || name == "lm_head.weight" {
                tensors.insert(name.clone(), self.shards[index].load(name, &Device::Cpu)?);
            }
        }
        if tensors.is_empty() {
            return Err(Error::Config("No Qwen3.5 language tensors found".into()));
        }
        candle_core::safetensors::save(&tensors, output.join("model.safetensors"))?;
        for name in [
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "LICENSE",
        ] {
            let path = source.join(name);
            if path.exists() {
                std::fs::copy(path, output.join(name))?;
            }
        }
        std::fs::write(
            output.join("switchyard-engine.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "format":"switchyard-native-text-v1","model_type":"qwen3_5","tensors":tensors.len(),
                "layout":"dense_c","precision":"source","gpu_snapshot":false
            }))?,
        )?;
        Ok(())
    }
}
impl SimpleBackend for Weights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.tensor(name, dtype, device)?;
        if tensor.shape() != &shape {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("Weight {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            });
        }
        Ok(tensor)
    }
    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        self.tensor(name, dtype, device)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.names.contains_key(name)
    }
}
