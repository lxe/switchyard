//! vLLM adapter identity and private plugin bundle, separate from transport.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Component, Path},
};
use switchyard_artifacts::{Error, Result};

const ADAPTER: &str = include_str!("../python/switchyard_extent_loader.py");
const RUNNER: &str = include_str!("../python/serve.py");
const STARTUP: &str = include_str!("../python/switchyard_vllm_startup.py");
const HOOKS: &str = include_str!("../python/switchyard_vllm_hooks.py");
const SERVING_PROFILE: &str = include_str!("../python/switchyard_serving_profile.py");
const CHUNKS: &str = include_str!("../python/switchyard_chunk_loader.py");
const FINAL_STATE: &str = include_str!("../python/switchyard_final_state.py");
pub(crate) const CPU_TREE: &str = include_str!("../python/switchyard_cpu_tree.py");
fn python_program() -> String {
    "venv/bin/python".into()
}
fn pinned_bytes() -> usize {
    16 * 1024 * 1024
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VllmLoaderConfig {
    #[serde(default = "python_program")]
    pub python_program: String,
    #[serde(default)]
    pub cuda_stage: bool,
    #[serde(default = "pinned_bytes")]
    pub pinned_bytes: usize,
    #[serde(default)]
    pub buffers: switchyard_artifacts::buffer_limits::BufferLimits,
}
impl Default for VllmLoaderConfig {
    fn default() -> Self {
        Self {
            python_program: python_program(),
            cuda_stage: false,
            pinned_bytes: pinned_bytes(),
            buffers: Default::default(),
        }
    }
}
impl VllmLoaderConfig {
    pub fn validate(&self) -> Result<()> {
        self.buffers.validate()?;
        if self.python_program.is_empty()
            || self.python_program.contains(['\\', '\0'])
            || !Path::new(&self.python_program)
                .components()
                .all(|p| matches!(p, Component::Normal(_)))
            || self.pinned_bytes == 0
            || self.pinned_bytes > 64 * 1024 * 1024
        {
            return Err(Error::Invalid("vLLM loader configuration".into()));
        }
        Ok(())
    }
    /// Small control-plane fingerprint; never hashes runtime or model content at startup.
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(
                "switchyard-vllm-loader-v4",
                CHUNKS,
                ADAPTER,
                RUNNER,
                STARTUP,
                HOOKS,
                SERVING_PROFILE,
                FINAL_STATE,
                CPU_TREE,
                self
            ))?)
        ))
    }
}
/// Create a caller-owned sidecar without modifying the immutable runtime artifact.
pub fn write_vllm_bundle(directory: &Path) -> Result<()> {
    fs::create_dir(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
    }
    fs::write(directory.join("switchyard_extent_loader.py"), ADAPTER)?;
    fs::write(directory.join("switchyard_chunk_loader.py"), CHUNKS)?;
    fs::write(directory.join("serve.py"), RUNNER)?;
    fs::write(directory.join("switchyard_vllm_startup.py"), STARTUP)?;
    fs::write(directory.join("switchyard_vllm_hooks.py"), HOOKS)?;
    fs::write(
        directory.join("switchyard_serving_profile.py"),
        SERVING_PROFILE,
    )?;
    fs::write(directory.join("switchyard_final_state.py"), FINAL_STATE)?;
    fs::write(directory.join("switchyard_cpu_tree.py"), CPU_TREE)?;
    let dist = directory.join("switchyard_extent_loader-0.0.0.dist-info");
    fs::create_dir(&dist)?;
    fs::write(
        dist.join("METADATA"),
        "Metadata-Version: 2.1\nName: switchyard-extent-loader\nVersion: 0.0.0\n",
    )?;
    fs::write(
        dist.join("entry_points.txt"),
        "[vllm.general_plugins]\nswitchyard_extents = switchyard_extent_loader:register\n",
    )?;
    Ok(())
}
