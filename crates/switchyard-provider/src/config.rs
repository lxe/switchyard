//! Provider and storage settings. Model, engine, and experiment configuration belongs to the runtime.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// RunPod-specific capacity settings, retained by the RunPod adapter.
/// Credential fields identify environment variables rather than secret values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputeConfig {
    pub api_url: String,
    pub key_env: String,
    pub cloud: String,
    pub gpu_types: Vec<String>,
    pub data_centers: Vec<String>,
    pub min_cuda: String,
    /// Placement floor for system RAM, expressed in the RunPod API's GB units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_ram_per_gpu_gb: Option<u64>,
    /// Placement floor for host vCPUs per requested GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_vcpu_count_per_gpu: Option<u64>,
    pub disk_gb: u64,
    pub disk_usd_per_gb_month: f64,
    pub billing_hours_per_month: f64,
}
impl ComputeConfig {
    pub fn validate(&self) -> Result<()> {
        validate_endpoint(&self.api_url)?;
        if !valid_env_name(&self.key_env)
            || !matches!(self.cloud.as_str(), "SECURE" | "COMMUNITY")
            || self.gpu_types.is_empty()
            || self.gpu_types.iter().any(|value| !valid_text(value))
            || self.data_centers.iter().any(|value| !valid_text(value))
            || !valid_text(&self.min_cuda)
            || self.min_ram_per_gpu_gb == Some(0)
            || self.min_vcpu_count_per_gpu == Some(0)
            || self.disk_gb == 0
            || self.billing_hours_per_month <= 0.0
            || !self.billing_hours_per_month.is_finite()
            || self.disk_usd_per_gb_month < 0.0
            || !self.disk_usd_per_gb_month.is_finite()
            || !(self.disk_gb as f64 * self.disk_usd_per_gb_month / self.billing_hours_per_month)
                .is_finite()
        {
            return Err(Error::Config("invalid RunPod capacity settings".into()));
        }
        Ok(())
    }
}

/// S3-compatible storage access settings. URL expiration is independent of runtime startup policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageConfig {
    pub endpoint: String,
    pub region: String,
    pub bucket: String,
    pub access_key_env: String,
    pub secret_key_env: String,
    pub url_lifetime_s: u64,
}
impl StorageConfig {
    pub fn validate(&self) -> Result<()> {
        validate_endpoint(&self.endpoint)?;
        if !valid_text(&self.region)
            || !valid_component(&self.bucket)
            || !valid_env_name(&self.access_key_env)
            || !valid_env_name(&self.secret_key_env)
            || !(60..=86400).contains(&self.url_lifetime_s)
        {
            return Err(Error::Config("invalid object storage settings".into()));
        }
        Ok(())
    }
}

/// Reject credentials in URLs and cleartext access outside a local test server.
pub fn validate_endpoint(value: &str) -> Result<()> {
    let url =
        reqwest::Url::parse(value).map_err(|_| Error::Config("invalid endpoint URL".into()))?;
    let local = url.host_str().is_some_and(|host| {
        host == "localhost"
            || host
                .trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    });
    if url.host_str().is_none()
        || !(url.scheme() == "https" || url.scheme() == "http" && local)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(Error::Config("endpoint URLs must use HTTPS without credentials or query, or address a local HTTP test server".into()));
    }
    Ok(())
}

/// Validate a portable identifier or path component; dot traversal components are excluded.
pub fn valid_component(value: &str) -> bool {
    !matches!(value, "" | "." | "..")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

/// Validate a slash-separated namespace made of portable components.
pub fn valid_key_prefix(value: &str) -> bool {
    !value.is_empty() && value.split('/').all(valid_component)
}

/// Require nonempty trimmed text without control characters.
pub fn valid_text(value: &str) -> bool {
    !value.is_empty() && value.trim() == value && !value.chars().any(char::is_control)
}

/// Validate an environment variable name without reading its value.
pub fn valid_env_name(value: &str) -> bool {
    value
        .bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}
