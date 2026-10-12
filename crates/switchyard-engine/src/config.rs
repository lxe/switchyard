use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model: String,
    pub model_root: PathBuf,
    #[serde(default, skip_serializing_if = "WeightFormat::is_default")]
    pub weights: WeightFormat,
    pub device: DeviceConfig,
    pub dtype: Precision,
    pub max_context: usize,
    pub max_weight_bytes: u64,
    pub max_batch_size: usize,
    pub queue_capacity: usize,
    pub batch_wait_ms: u64,
    #[serde(default, skip_serializing_if = "CacheConfig::is_default")]
    pub cache: CacheConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<crate::multimodal::ServingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraConfig>,
}
/// Dynamic PEFT adapter limits. Base weights remain shared; each adapter owns
/// its low-rank tensors and a separately bounded prefix cache.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoraConfig {
    pub root: PathBuf,
    #[serde(default)]
    pub base_model_aliases: Vec<String>,
    pub max_loaded: usize,
    pub max_rank: usize,
    pub max_upload_bytes: u64,
    pub max_resident_bytes: u64,
    pub transfer_timeout_s: u64,
}
impl LoraConfig {
    pub fn validate(&self) -> Result<()> {
        if self.root.as_os_str().is_empty()
            || !(1..=64).contains(&self.max_loaded)
            || !(1..=256).contains(&self.max_rank)
            || self.max_upload_bytes == 0
            || self.max_upload_bytes > 1024 * 1024 * 1024
            || self.max_resident_bytes == 0
            || !(1..=600).contains(&self.transfer_timeout_s)
            || self.base_model_aliases.iter().any(String::is_empty)
        {
            return Err(Error::Config("Invalid fixed LoRA admission bounds".into()));
        }
        Ok(())
    }
}
/// Model storage contract; GGUF uses the companion HF config and tokenizer in model_root.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WeightFormat {
    #[default]
    Safetensors,
    Gguf {
        file: PathBuf,
    },
}
impl WeightFormat {
    fn is_default(&self) -> bool {
        matches!(self, Self::Safetensors)
    }
    fn validate(&self) -> Result<()> {
        if let Self::Gguf { file } = self
            && (file.as_os_str().is_empty()
                || file
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_))))
        {
            return Err(Error::Config(
                "GGUF file must be a relative path within model_root".into(),
            ));
        }

        Ok(())
    }
}
/// Explicit GPU state retention bounds. Zero bytes and entries disable prefix
/// retention; attention still uses bounded, immutable pages for active rows.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheConfig {
    pub page_tokens: usize,
    pub checkpoint_tokens: usize,
    pub max_bytes: usize,
    pub max_entries: usize,
}
impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            page_tokens: 16,
            checkpoint_tokens: 32,
            max_bytes: 0,
            max_entries: 0,
        }
    }
}
impl CacheConfig {
    fn is_default(&self) -> bool {
        self == &Self::default()
    }
    pub fn enabled(&self) -> bool {
        self.max_entries > 0 && self.max_bytes > 0
    }
    pub fn validate(&self) -> Result<()> {
        if !(1..=4096).contains(&self.page_tokens)
            || !(1..=4096).contains(&self.checkpoint_tokens)
            || self.max_entries > 4096
            || (self.max_bytes == 0) != (self.max_entries == 0)
        {
            return Err(Error::Config(
                "Invalid page, checkpoint or prefix retention bounds".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeviceConfig {
    Cpu,
    Cuda { ordinal: usize },
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    Bf16,
    F32,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        self.cache.validate()?;
        if let Some(lora) = &self.lora {
            lora.validate()?;
        }
        if let Some(vision) = &self.vision {
            vision.validate()?;
        }
        self.weights.validate()?;
        if matches!(self.weights, WeightFormat::Gguf { .. })
            && !matches!(self.dtype, Precision::F32)
        {
            return Err(Error::Config(
                "Packed GGUF projections require F32 activations".into(),
            ));
        }
        if matches!(self.weights, WeightFormat::Gguf { .. })
            && self.vision.as_ref().is_some_and(|v| v.mmproj.is_none())
        {
            return Err(Error::Config(
                "GGUF image serving requires a separate mmproj artifact".into(),
            ));
        }
        if self.model.is_empty()
            || self.max_context == 0
            || self.max_weight_bytes == 0
            || self.max_batch_size == 0
            || self.queue_capacity < self.max_batch_size
            || self.batch_wait_ms > 1000
        {
            return Err(Error::Config(
                "Invalid explicit context, batch or queue bounds".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_image_profiles_validate_fixed_bounds() {
        let profiles = [
            include_str!("../../../examples/native-qwen35-0.8b-vision.json"),
            include_str!("../../../examples/native-qwen35-2b-vision.json"),
            include_str!("../../../examples/native-qwen35-4b-vision.json"),
            include_str!("../../../examples/native-qwen35-0.8b-gguf-vision.json"),
            include_str!("../../../examples/native-qwen35-2b-gguf-vision.json"),
            include_str!("../../../examples/native-qwen35-4b-gguf-vision.json"),
        ];
        for source in profiles {
            let config: Config = serde_json::from_str(source).unwrap();
            config.validate().unwrap();
            assert_eq!(config.max_context, 512);
            assert_eq!(config.vision.as_ref().unwrap().max_pixels, 65_536);
            let roundtrip: Config =
                serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
            roundtrip.validate().unwrap();
        }
    }
    #[test]
    fn gguf_paths_and_activation_contract_are_explicit() {
        let mut config: Config = serde_json::from_str(include_str!(
            "../../../examples/native-qwen35-0.8b-gguf.json"
        ))
        .unwrap();
        assert!(config.validate().is_ok());
        for file in ["../model.gguf", "/model.gguf", ""] {
            config.weights = WeightFormat::Gguf { file: file.into() };
            assert!(config.validate().is_err());
        }
        config.weights = WeightFormat::Gguf {
            file: "model.gguf".into(),
        };
        config.dtype = Precision::Bf16;
        assert!(config.validate().is_err());
        config.weights = WeightFormat::Safetensors;
        assert!(config.validate().is_ok());
        let value = serde_json::to_value(config).unwrap();
        assert!(!value.as_object().unwrap().contains_key("weights"));
    }
}
