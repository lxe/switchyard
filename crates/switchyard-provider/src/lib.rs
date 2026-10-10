//! Cloud provisioning and object-storage adapters for live inference experiments.
pub mod config;
pub mod runpod;
pub mod storage;
pub use config::*;
pub use runpod::RunPod;
use std::collections::BTreeMap;
use std::future::Future;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("configuration: {0}")]
    Config(String),
    #[error("provider returned HTTP {status} for {operation}")]
    Http { status: u16, operation: String },
    #[error("transport failed for {0}")]
    Transport(String),
    #[error("invalid provider response: {0}")]
    Response(String),
    #[error("no matching capacity is available")]
    NoCapacity,
    #[error(
        "pod creation outcome is unknown for launch {name}; reconcile inventory before retrying"
    )]
    UnknownLaunch { name: String },
}
pub type Result<T> = std::result::Result<T, Error>;

/// A provider supplies quotes and lifecycle operations; it does not make routing decisions.
/// Creation must reconcile uncertain outcomes before retrying a paid mutation.
pub trait Provider: Send + Sync {
    /// Return the provider's externally reachable HTTP endpoint for this pod and port.
    fn endpoint_url(&self, pod: &Pod, port: u16) -> String;
    fn offers(&self) -> impl Future<Output = Result<Vec<Offering>>> + Send;
    fn create(&self, launch: Launch) -> impl Future<Output = Result<Pod>> + Send;
    fn get(&self, id: &str) -> impl Future<Output = Result<Option<Pod>>> + Send;
    fn list(&self) -> impl Future<Output = Result<Vec<Pod>>> + Send;
    fn terminate(&self, id: &str) -> impl Future<Output = Result<()>> + Send;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Offering {
    pub gpu: String,
    pub gpu_memory_gb: u64,
    pub data_center: String,
    pub availability: String,
    pub gpu_usd_per_hour: f64,
    pub disk_usd_per_hour: f64,
    pub quoted_at_unix_s: u64,
}
impl Offering {
    pub fn total_usd_per_hour(&self) -> f64 {
        self.gpu_usd_per_hour + self.disk_usd_per_hour
    }
}

/// Sanitized pod metadata. Environment, start commands, and credentials are never retained.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Pod {
    pub id: String,
    pub name: String,
    pub status: String,
    /// Provider-reported hourly quote. This is not a settled billing charge.
    /// Providers can include storage differently; billing reconciliation remains separate.
    pub cost: f64,
    #[serde(rename = "dataCenterId", default)]
    pub data_center: Option<String>,
}

/// Launch credentials are intentionally excluded from Debug and persisted receipts.
pub struct Launch {
    pub name: String,
    pub offering: Offering,
    pub container: ContainerSpec,
}

/// A generic container launch, independent of model serving and hydration.
/// Environment values and command arguments may carry secrets, so Debug is omitted.
#[derive(Clone)]
pub struct ContainerSpec {
    pub image: String,
    pub disk_gb: u64,
    /// An empty vector retains the image's entrypoint.
    pub entrypoint: Vec<String>,
    /// An empty vector retains the image's command.
    pub command: Vec<String>,
    pub environment: BTreeMap<String, String>,
    pub ports: Vec<Port>,
}

/// One container port. HTTP ports receive a provider proxy; other ports use TCP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Port {
    pub number: u16,
    pub http: bool,
}

impl ContainerSpec {
    pub fn validate(&self) -> Result<()> {
        if self.image.is_empty()
            || self.image.trim() != self.image
            || self.image.chars().any(char::is_control)
            || self.disk_gb == 0
            || self
                .entrypoint
                .iter()
                .chain(&self.command)
                .any(|v| v.contains('\0'))
            || self
                .environment
                .iter()
                .any(|(name, value)| !config::valid_env_name(name) || value.contains('\0'))
        {
            return Err(Error::Config("invalid container specification".into()));
        }
        let mut ports = std::collections::BTreeSet::new();
        if self
            .ports
            .iter()
            .any(|port| port.number == 0 || !ports.insert((port.number, port.http)))
        {
            return Err(Error::Config(
                "container ports must be nonzero and unique".into(),
            ));
        }
        Ok(())
    }
}

pub fn unix_s() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
