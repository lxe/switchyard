use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model: String,
    pub model_root: PathBuf,
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
        if let Some(vision) = &self.vision {
            vision.validate()?;
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
