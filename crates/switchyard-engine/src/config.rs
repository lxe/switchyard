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
