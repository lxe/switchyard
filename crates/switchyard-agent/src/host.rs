//! Multi-endpoint host tickets; immutable runtime/files may be shared, process state may not.
use crate::{Error, Result, Ticket};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use switchyard_core::topology::HostPlan;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostTicket {
    pub schema_version: u32,
    /// Shared immutable runtime/components and endpoint-private cache storage.
    /// None preserves /opt/switchyard; model destinations belong to endpoint tickets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_root: Option<String>,
    pub plan: HostPlan,
    pub endpoints: Vec<HostedEndpoint>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostedEndpoint {
    pub id: String,
    pub port: u16,
    pub ticket: Ticket,
}
impl HostTicket {
    pub fn validate(&self) -> Result<()> {
        crate::startup::StartupPolicy {
            model_root: self.storage_root.clone(),
            ..Default::default()
        }
        .validate()?;
        self.plan.validate().map_err(Error::Manifest)?;
        if self.schema_version != 1 || self.endpoints.len() != self.plan.endpoints.len() {
            return Err(Error::Manifest(
                "invalid host ticket version or endpoint bindings".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        let mut ports = BTreeSet::new();
        let mut sources = BTreeMap::new();
        let mut runtime_identity = None;
        let mut transfer_policy = None;
        for endpoint in &self.endpoints {
            let ticket = &endpoint.ticket;
            let runtime = ticket.runtime.as_ref().ok_or_else(|| {
                Error::Manifest("hosted engines require a hydrated runtime".into())
            })?;
            let plan = self
                .plan
                .endpoints
                .iter()
                .find(|e| e.id == endpoint.id)
                .ok_or_else(|| Error::Manifest("endpoint is absent from host plan".into()))?;
            if !ids.insert(&endpoint.id)
                || !ports.insert(endpoint.port)
                || !(8100..=8131).contains(&endpoint.port)
                || ticket.source.extent().is_some()
                || ticket.lifetime_s == 0
            {
                return Err(Error::Manifest(
                    "invalid hosted engine port, lifetime, source, or export".into(),
                ));
            }
            ticket.startup.validate()?;
            let model_directory = ticket.startup.model_directory();
            let storage =
                std::path::Path::new(self.storage_root.as_deref().unwrap_or("/opt/switchyard"));
            for name in ["runtime", "components", "cache", "artifacts", ".host-seeds"] {
                let reserved = storage.join(name);
                if model_directory.starts_with(&reserved) || reserved.starts_with(&model_directory)
                {
                    return Err(Error::Manifest(
                        "host model destination overlaps runtime storage".into(),
                    ));
                }
            }
            if sources.keys().any(|other: &std::path::PathBuf| {
                other != &model_directory
                    && (other.starts_with(&model_directory) || model_directory.starts_with(other))
            }) {
                return Err(Error::Manifest(
                    "nested host model destinations are unsupported".into(),
                ));
            }
            // The managed native launch contract identifies the device and storage explicitly.
            let args = &ticket.engine_args;
            if args.len() != 5
                || args[0] != "--config-json"
                || args[2] != "serve"
                || args[3] != "--bind"
                || args[4] != format!("127.0.0.1:{}", endpoint.port)
            {
                return Err(Error::Manifest(
                    "hosted serving requires managed native arguments".into(),
                ));
            }
            let config: serde_json::Value = serde_json::from_str(&args[1])
                .map_err(|_| Error::Manifest("invalid native configuration".into()))?;
            if config["model"].as_str() != Some(&ticket.model)
                || config["model_root"].as_str()
                    != Some(ticket.startup.model_directory().to_string_lossy().as_ref())
                || config["device"]["ordinal"].as_u64() != Some(0)
                || config["device"]["kind"].as_str() != Some("cuda")
                || config["max_weight_bytes"]
                    .as_u64()
                    .is_none_or(|n| n == 0 || n > plan.gpu_memory_bytes)
            {
                return Err(Error::Manifest(
                    "native model, storage, device, or reservation mismatch".into(),
                ));
            }
            let source = model_identity(ticket);
            if sources
                .insert(ticket.startup.model_directory(), source.clone())
                .is_some_and(|old| old != source)
            {
                return Err(Error::Manifest(
                    "different weight artifacts share a destination".into(),
                ));
            }
            let identity = (
                runtime.artifact.manifest.id.clone(),
                runtime
                    .components
                    .iter()
                    .map(|p| (p.name.clone(), p.artifact.manifest.id.clone()))
                    .collect::<Vec<_>>(),
            );
            if runtime_identity
                .replace(identity.clone())
                .is_some_and(|old| old != identity)
            {
                return Err(Error::Manifest(
                    "hosted engines require the same immutable runtime and components".into(),
                ));
            }
            let policy = (
                ticket.concurrency,
                ticket.stall_s,
                ticket.lifetime_s,
                ticket.startup.mode,
            );
            if transfer_policy
                .replace(policy)
                .is_some_and(|old| old != policy)
            {
                return Err(Error::Manifest(
                    "hosted engines require one shared transfer budget and lifetime".into(),
                ));
            }
            if !(1..=64).contains(&ticket.concurrency) {
                return Err(Error::Manifest("invalid host transfer budget".into()));
            }
        }
        Ok(())
    }
}
pub(crate) fn model_identity(ticket: &Ticket) -> String {
    ticket.source.artifact_identity().to_owned()
}
