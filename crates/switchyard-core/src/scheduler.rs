//! Pluggable scheduling contract. No trace output lengths cross this boundary.
use crate::config::*;
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};

/// Scheduler-visible request, deliberately lacking actual or remaining output.
#[derive(Debug, Clone)]
pub struct PlannedRequest {
    /// False for requests already bound to a provider endpoint by another router.
    pub routing_allowed: bool,
    pub id: usize,
    pub job: usize,
    pub model: String,
    pub workload: Workload,
    pub class: String,
    pub arrival_s: f64,
    pub input_tokens: u64,
    pub planned_output: u64,
    pub max_output: u64,
    pub image_megapixels: f64,
    pub upload_bytes: u64,
    pub cached_fraction: f64,
    pub adapter: Option<String>,
    pub latency_target_s: f64,
    pub depends_on: Vec<usize>,
}
impl From<&Request> for PlannedRequest {
    fn from(r: &Request) -> Self {
        Self {
            routing_allowed: true,
            id: r.id,
            job: r.job,
            model: r.model.clone(),
            workload: r.workload,
            class: r.class.clone(),
            arrival_s: r.arrival_s,
            input_tokens: r.input_tokens,
            planned_output: r.planned_output,
            max_output: r.max_output,
            image_megapixels: r.image_megapixels,
            upload_bytes: r.upload_bytes,
            cached_fraction: r.cached_fraction,
            adapter: r.adapter.clone(),
            latency_target_s: r.latency_target_s,
            depends_on: r.depends_on.clone(),
        }
    }
}
impl InferenceInput for PlannedRequest {
    fn model(&self) -> &str {
        &self.model
    }
    fn workload(&self) -> Workload {
        self.workload
    }
    fn input_tokens(&self) -> u64 {
        self.input_tokens
    }
    fn max_output(&self) -> u64 {
        self.max_output
    }
    fn cached_fraction(&self) -> f64 {
        self.cached_fraction
    }
    fn image_megapixels(&self) -> f64 {
        self.image_megapixels
    }
    fn adapter(&self) -> Option<&str> {
        self.adapter.as_deref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    Future,
    Queued,
    Active,
    Delivering,
    Done,
    Rejected,
}
#[derive(Debug, Clone)]
pub struct ObservedRequest {
    pub state: RequestStatus,
    pub completed_s: Option<f64>,
}
#[derive(Debug, Clone)]
pub struct ObservedActive {
    pub id: usize,
    pub prefill: f64,
    pub generated: f64,
    pub available_s: f64,
}
#[derive(Debug, Clone)]
pub struct ObservedNode {
    pub placement: Option<crate::topology::EndpointPlacement>,
    pub profile: usize,
    pub warm: bool,
    pub allocated_s: f64,
    pub ready_s: f64,
    pub ended_s: Option<f64>,
    pub active: Vec<ObservedActive>,
    pub adapters: VecDeque<(String, f64)>,
    pub last_used_s: f64,
    pub is_ready: bool,
    pub host: HostObservation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StartupPhase {
    Booting,
    Hydrating,
    Initializing,
    Ready,
    Ended,
}
#[derive(Debug, Clone, Serialize)]
pub struct HostProbe {
    pub at_s: f64,
    pub started_s: f64,
    pub received_bytes: u64,
    pub volume_bytes_per_s: f64,
    /// None means the provider or probe cannot observe container-image cache status.
    pub image_cached: Option<bool>,
    pub model_cached: bool,
    pub hydration_bytes: u64,
}
impl HostProbe {
    pub fn fresh(&self, now: f64, policy: &ProbePolicy) -> bool {
        let window = self.at_s - self.started_s;
        self.at_s.is_finite()
            && self.started_s.is_finite()
            && self.started_s >= 0.0
            && now.is_finite()
            && now >= self.at_s
            && now - self.at_s <= policy.maximum_age_s
            && window >= policy.minimum_window_s
            && window <= policy.maximum_window_s
            && self.received_bytes >= policy.minimum_received_bytes
            && self.volume_bytes_per_s.is_finite()
            && self.volume_bytes_per_s > 0.0
            && (self.volume_bytes_per_s - self.received_bytes as f64 / window).abs()
                <= self.volume_bytes_per_s * 1e-9
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct HostObservation {
    pub phase: StartupPhase,
    /// Confirmed transfer progress; a lower bound when retained progress is unknown.
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    /// Whether remaining bytes can be estimated from the observed retained progress.
    /// Wire volume alone cannot establish this when failed attempts can be discarded.
    pub remaining_bytes_known: bool,
    pub probe_pending: bool,
    pub probe_attempted_s: Option<f64>,
    pub probe: Option<HostProbe>,
}

/// Snapshot borrows immutable plans and contains only observable runtime state.
#[derive(Debug, Clone)]
pub struct FleetView<'a> {
    pub topology: &'a crate::topology::FleetTopology,
    pub now: f64,
    pub allow_scale: bool,
    pub profiles: &'a [Profile],
    /// Earliest retry time for profiles whose allocation was observably rejected.
    /// These observations restrict new provisioning, not routing to existing hosts.
    pub provisioning_cooldowns: BTreeMap<usize, f64>,
    /// Accepted controller intents awaiting a provider result; never routable.
    pub pending_allocations: Vec<PendingAllocation>,
    pub requests: BTreeMap<usize, PlannedRequest>,
    pub states: BTreeMap<usize, ObservedRequest>,
    pub nodes: Vec<ObservedNode>,
    pub jobs: BTreeMap<usize, Job>,
    pub scheduling: &'a Scheduling,
    pub max_nodes_per_model: usize,
    pub idle_timeout_s: f64,
    pub reactive_delay_s: f64,
    pub build_probability: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    Provision {
        profile: usize,
    },
    Route {
        request: usize,
        node: usize,
        service_weight: f64,
    },
    Release {
        node: usize,
    },
    ProbeHost {
        node: usize,
    },
}

/// Public allocation intent, counted as reserved capacity while cloud I/O proceeds.
#[derive(Debug, Clone, Serialize)]
pub struct PendingAllocation {
    pub profile: usize,
    pub started_s: f64,
    pub expected_ready_s: f64,
}

/// Observed outcomes only. A completion must never be sent before client delivery.
#[derive(Debug, Clone)]
pub enum SchedulerFeedback {
    Allocation {
        profile: usize,
        accepted: bool,
        elapsed_s: f64,
    },
    Completion {
        node: usize,
        profile: usize,
        request: Box<PlannedRequest>,
        output_tokens: u64,
        input_work: f64,
        started_s: f64,
        completed_s: f64,
    },
    FailedRequest {
        node: usize,
    },
}

/// Implement this trait to test an independent scheduler against the same fleet.
pub trait Scheduler {
    fn name(&self) -> &str;
    fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<Action>;
    fn observe(&mut self, _feedback: SchedulerFeedback) {}
    fn diagnostics(&self) -> serde_json::Value {
        serde_json::Value::Null
    }
}
