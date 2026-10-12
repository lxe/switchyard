//! Physical allocations and endpoint residency, independent of provider and engine.
use crate::Profile;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FleetTopology {
    pub host_plans: Vec<HostPlan>,
}
pub static EMPTY_TOPOLOGY: FleetTopology = FleetTopology {
    host_plans: Vec::new(),
};

/// One indivisible allocation. Profiles describe endpoints; the host pays rent once.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostPlan {
    pub id: String,
    pub gpu: String,
    pub gpu_memory_bytes: u64,
    /// Device memory excluded from deployment admission (driver/workspace margin).
    pub reserved_gpu_memory_bytes: u64,
    /// Conservative multiplier for shared-device service; must be measured for precision.
    pub contention_factor: f64,
    pub endpoints: Vec<EndpointPlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointPlan {
    pub id: String,
    /// Exact profile identity. Repeating a profile creates separate engine replicas.
    pub profile: String,
    /// Fixed worst-case reservation: weights, KV/prefix state, adapters, and workspace.
    pub gpu_memory_bytes: u64,
    /// Reserved fraction of solo-profile capacity, including when siblings are idle.
    /// This is a planning model, not a CUDA execution-isolation guarantee.
    pub compute_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointPlacement {
    /// Physical allocation identity, shared by all resident endpoints.
    pub host: usize,
    pub plan: String,
    pub endpoint: String,
    pub gpu_memory_bytes: u64,
    pub capacity_fraction: f64,
}

impl HostPlan {
    pub fn validate(&self) -> Result<(), String> {
        if !component(&self.id)
            || self.gpu.trim().is_empty()
            || self.endpoints.is_empty()
            || self.endpoints.len() > 32
            || self.gpu_memory_bytes == 0
            || !self.contention_factor.is_finite()
            || self.contention_factor <= 0.0
            || self.contention_factor > 1.0
        {
            return Err("invalid host plan identity, capacity, or endpoint count".into());
        }
        let mut ids = BTreeSet::new();
        let mut memory = self.reserved_gpu_memory_bytes;
        let mut compute = 0.0;
        for endpoint in &self.endpoints {
            if !component(&endpoint.id)
                || !ids.insert(&endpoint.id)
                || endpoint.profile.is_empty()
                || endpoint.gpu_memory_bytes == 0
                || !endpoint.compute_fraction.is_finite()
                || endpoint.compute_fraction <= 0.0
            {
                return Err("invalid or duplicate endpoint reservation".into());
            }
            memory = memory
                .checked_add(endpoint.gpu_memory_bytes)
                .ok_or("host GPU memory reservation overflow")?;
            compute += endpoint.compute_fraction;
        }
        if memory > self.gpu_memory_bytes || compute > 1.0 + 1e-9 {
            return Err("host plan overcommits GPU memory or compute".into());
        }
        Ok(())
    }
    pub fn placement(&self, host: usize, endpoint: &EndpointPlan) -> EndpointPlacement {
        EndpointPlacement {
            host,
            plan: self.id.clone(),
            endpoint: endpoint.id.clone(),
            gpu_memory_bytes: endpoint.gpu_memory_bytes,
            capacity_fraction: endpoint.compute_fraction * self.contention_factor,
        }
    }
}
impl FleetTopology {
    pub fn plan_for(&self, profile: &str) -> Option<&HostPlan> {
        self.host_plans
            .iter()
            .find(|p| p.endpoints.iter().any(|e| e.profile == profile))
    }
    pub fn validate(&self, profiles: &[Profile]) -> Result<(), String> {
        self.validate_structure()?;
        for plan in &self.host_plans {
            let mut quote = None;
            for endpoint in &plan.endpoints {
                let profile = profiles
                    .iter()
                    .find(|p| p.id == endpoint.profile)
                    .ok_or_else(|| format!("unknown endpoint profile {}", endpoint.profile))?;
                if profile.gpu != plan.gpu {
                    return Err("host GPU and endpoint profile GPU differ".into());
                }
                let current = (profile.usd_per_hour, profile.minimum_bill_s);
                if quote.is_some_and(|q| q != current) {
                    return Err("resident profiles must describe the same rental quote".into());
                }
                quote = Some(current);
            }
        }
        Ok(())
    }
    pub fn validate_structure(&self) -> Result<(), String> {
        let mut ids = BTreeSet::new();
        let mut profiles = BTreeSet::new();
        if self.host_plans.len() > 128 {
            return Err("too many host plans".into());
        }
        for plan in &self.host_plans {
            plan.validate()?;
            if !ids.insert(&plan.id) {
                return Err("duplicate host plan".into());
            }
            let local: BTreeSet<_> = plan.endpoints.iter().map(|e| &e.profile).collect();
            for profile in local {
                if !profiles.insert(profile) {
                    return Err("a profile belongs to more than one host plan; use distinct placement profiles".into());
                }
            }
        }
        Ok(())
    }
}
pub fn component(value: &str) -> bool {
    !matches!(value, "" | "." | "..")
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}
