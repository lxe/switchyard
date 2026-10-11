use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Workload {
    Text,
    Vision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    WarmOnly,
    Reactive,
    CheapestElastic,
    PlannedHybrid,
    Adaptive,
}

impl fmt::Display for Policy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::WarmOnly => "warm_only",
            Self::Reactive => "reactive",
            Self::CheapestElastic => "cheapest_elastic",
            Self::PlannedHybrid => "planned_hybrid",
            Self::Adaptive => "adaptive",
        })
    }
}

pub const POLICIES: [Policy; 5] = [
    Policy::WarmOnly,
    Policy::Reactive,
    Policy::CheapestElastic,
    Policy::PlannedHybrid,
    Policy::Adaptive,
];

impl Policy {
    pub fn plans_ahead(self) -> bool {
        matches!(self, Self::PlannedHybrid | Self::Adaptive)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityPoint {
    pub in_flight: usize,
    pub work_units_per_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Startup {
    pub allocation_s: f64,
    pub container_s: f64,
    pub weights_s: f64,
    pub engine_s: f64,
    /// Symmetric fractional jitter applied to the total; not an empirical CDF.
    pub jitter: f64,
    /// Optional empirical end-to-end samples, overriding the component sum.
    #[serde(default)]
    pub restore_samples_s: Vec<f64>,
    #[serde(default)]
    pub build_samples_s: Vec<f64>,
}
impl Startup {
    pub fn total(&self) -> f64 {
        self.allocation_s + self.container_s + self.weights_s + self.engine_s
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub id: String,
    pub model: String,
    pub workload: Workload,
    pub gpu: String,
    pub usd_per_hour: f64,
    pub startup: Startup,
    pub network_s: f64,
    pub network_bytes_per_s: f64,
    pub max_in_flight: usize,
    /// Conservative total input + maximum output reservation, after weights.
    pub kv_tokens: u64,
    /// Estimated input work in output-equivalent units; not benchmark in_tok_s.
    pub input_work_per_token: f64,
    pub image_work_per_megapixel: f64,
    pub adapter_load_s: f64,
    pub multi_adapter_factor: f64,
    /// Zero supports base-model requests only; positive values bound adapter residency.
    pub adapter_cache_slots: usize,
    pub minimum_bill_s: f64,
    pub curve: Vec<CapacityPoint>,
    /// Always readable in the output: measured, boundary_adjusted, or assumed.
    pub evidence: String,
    pub source: String,
    #[serde(default)]
    pub host: HostProfile,
}

/// Host priors describe possible realities, never the state of a sampled host.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HostProfile {
    pub volume_bandwidth_bytes_per_s: Vec<f64>,
    pub container_image_bytes: u64,
    pub model_bytes: u64,
    pub image_cache_probability: f64,
    pub model_cache_probability: f64,
    pub probe_duration_s: f64,
    pub measurement_noise: f64,
}
impl Default for HostProfile {
    fn default() -> Self {
        Self {
            volume_bandwidth_bytes_per_s: vec![100_000_000.0],
            container_image_bytes: 0,
            model_bytes: 0,
            image_cache_probability: 0.5,
            model_cache_probability: 0.5,
            probe_duration_s: 1.0,
            measurement_noise: 0.05,
        }
    }
}

impl Profile {
    pub fn capacity(&self, n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }
        let first = &self.curve[0];
        if n <= first.in_flight {
            return first.work_units_per_s * n as f64 / first.in_flight as f64;
        }
        for pair in self.curve.windows(2) {
            if n <= pair[1].in_flight {
                let fraction =
                    (n - pair[0].in_flight) as f64 / (pair[1].in_flight - pair[0].in_flight) as f64;
                return pair[0].work_units_per_s
                    + fraction * (pair[1].work_units_per_s - pair[0].work_units_per_s);
            }
        }
        self.curve
            .last()
            .expect("validated nonempty curve")
            .work_units_per_s
    }
    pub fn peak(&self) -> f64 {
        self.capacity(self.max_in_flight)
    }
    pub fn compatible(&self, r: &impl InferenceInput) -> bool {
        self.model == r.model()
            && self.workload == r.workload()
            && (r.adapter().is_none() || self.adapter_cache_slots > 0)
            && r.input_tokens()
                .checked_add(r.max_output())
                .is_some_and(|tokens| tokens <= self.kv_tokens)
    }
    pub fn prefill_work(&self, r: &impl InferenceInput) -> f64 {
        r.input_tokens() as f64 * (1.0 - r.cached_fraction()) * self.input_work_per_token
            + r.image_megapixels() * self.image_work_per_megapixel
    }
}

/// Known request inputs used by both simulation traces and scheduler views.
pub trait InferenceInput {
    fn model(&self) -> &str;
    fn workload(&self) -> Workload;
    fn input_tokens(&self) -> u64;
    fn max_output(&self) -> u64;
    fn cached_fraction(&self) -> f64;
    fn image_megapixels(&self) -> f64;
    /// Requested serving adapter, if any. Existing base-model callers can use the default.
    fn adapter(&self) -> Option<&str> {
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: usize,
    pub job: usize,
    pub model: String,
    pub workload: Workload,
    /// User-defined service class; no application-specific roles in the engine.
    pub class: String,
    pub arrival_s: f64,
    pub input_tokens: u64,
    pub planned_output: u64,
    /// Hard generation limit used for memory admission; planning may be lower.
    pub max_output: u64,
    pub actual_output: u64,
    pub image_megapixels: f64,
    pub upload_bytes: u64,
    pub cached_fraction: f64,
    pub adapter: Option<String>,
    pub latency_target_s: f64,
    /// Request indices that must deliver their responses before this can run.
    #[serde(default)]
    pub depends_on: Vec<usize>,
}
impl InferenceInput for Request {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub id: usize,
    pub name: String,
    pub announced_s: f64,
    pub arrival_s: f64,
    pub completion_budget_s: f64,
    /// Aggregate completed output token/s, including cold start and queue delay.
    pub desired_output_tps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmNode {
    pub profile: String,
    pub copies: usize,
}

/// Provider environment and convenience-router limits for HTTP experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CloudConfig {
    pub tick_s: f64,
    pub max_instances_per_model: usize,
    pub instance_queue_limit: usize,
    pub router_queue_limit: usize,
    pub request_completion_budget_s: f64,
    pub idle_timeout_s: f64,
    pub launch_fail_probability: f64,
    pub reactive_delay_s: f64,
    pub build_probability: f64,
    pub capacity_scale: f64,
    pub startup_scale: f64,
    pub host_reality: Option<HostProfile>,
    pub warm: Vec<WarmNode>,
    pub faults: Vec<Fault>,
    pub router_policy: Policy,
    pub scheduling: Scheduling,
}
impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            tick_s: 0.25,
            max_instances_per_model: 64,
            instance_queue_limit: 1000,
            router_queue_limit: 10_000,
            request_completion_budget_s: 60.0,
            idle_timeout_s: 30.0,
            launch_fail_probability: 0.0,
            reactive_delay_s: 0.0,
            build_probability: 0.0,
            capacity_scale: 1.0,
            startup_scale: 1.0,
            host_reality: None,
            warm: vec![],
            faults: vec![],
            router_policy: Policy::PlannedHybrid,
            scheduling: Scheduling::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fault {
    pub at_s: f64,
    /// Zero-based allocated instance; one simulated failure, no real GPU.
    pub node: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
    pub notes: String,
    pub horizon_s: f64,
    pub tick_s: f64,
    pub idle_timeout_s: f64,
    pub max_nodes_per_model: usize,
    pub launch_fail_probability: f64,
    pub reactive_delay_s: f64,
    pub capacity_scale: f64,
    pub startup_scale: f64,
    #[serde(default)]
    pub build_probability: f64,
    #[serde(default)]
    pub scheduling: Scheduling,
    /// Optional hidden reality override for stress testing stale host priors.
    #[serde(default)]
    pub host_reality: Option<HostProfile>,
    pub warm: Vec<WarmNode>,
    pub jobs: Vec<Job>,
    pub requests: Vec<Request>,
    pub faults: Vec<Fault>,
}

/// Policy controls are exposed in trace configuration for sensitivity studies.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scheduling {
    /// EWMA update weight for newly measured cold-host observations.
    pub learning_rate: f64,
    pub allow_queue_overtaking: bool,
    pub plan_interval_s: f64,
    pub reactive_drain_s: f64,
    pub minimum_plan_window_s: f64,
    pub warm_batch_capacity_fraction: f64,
    pub reserve_warm_slots: usize,
    pub aging_s: f64,
    pub replan_output_floor: f64,
    pub classes: BTreeMap<String, ServiceClass>,
    pub minimum_hydration_bytes_per_s: f64,
    pub maximum_startup_s: f64,
    pub retry_cooldown_s: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceClass {
    /// Lower priority numbers are admitted first.
    pub priority: u32,
    /// Relative GPU service share under contention on the warm pool.
    pub weight: f64,
    pub can_use_reserved_warm_slots: bool,
    pub plan_ahead: bool,
}
impl Default for Scheduling {
    fn default() -> Self {
        Self {
            learning_rate: 0.25,
            allow_queue_overtaking: true,
            plan_interval_s: 2.0,
            reactive_drain_s: 30.0,
            minimum_plan_window_s: 5.0,
            warm_batch_capacity_fraction: 0.65,
            reserve_warm_slots: 1,
            aging_s: 60.0,
            replan_output_floor: 32.0,
            classes: [
                (
                    "latency".into(),
                    ServiceClass {
                        priority: 0,
                        weight: 4.0,
                        can_use_reserved_warm_slots: true,
                        plan_ahead: false,
                    },
                ),
                (
                    "standard".into(),
                    ServiceClass {
                        priority: 5,
                        weight: 1.0,
                        can_use_reserved_warm_slots: true,
                        plan_ahead: false,
                    },
                ),
                (
                    "batch".into(),
                    ServiceClass {
                        priority: 10,
                        weight: 1.0,
                        can_use_reserved_warm_slots: false,
                        plan_ahead: true,
                    },
                ),
            ]
            .into(),
            minimum_hydration_bytes_per_s: 25_000_000.0,
            maximum_startup_s: 180.0,
            retry_cooldown_s: 60.0,
        }
    }
}

pub fn validate(profiles: &[Profile], scenario: &Scenario) -> Result<(), String> {
    let finite_nonnegative = |v: f64| v.is_finite() && v >= 0.0;
    if profiles.is_empty() {
        return Err("no hardware profiles".into());
    }
    let mut profile_ids = std::collections::HashSet::new();
    for p in profiles {
        if p.id.trim().is_empty() || p.model.trim().is_empty() || p.gpu.trim().is_empty() {
            return Err("profile identifiers must be nonempty".into());
        }
        if !profile_ids.insert(&p.id) {
            return Err(format!("duplicate profile {}", p.id));
        }
        if p.curve.is_empty() || p.max_in_flight == 0 || p.kv_tokens == 0 {
            return Err(format!("{} has empty capacity or memory", p.id));
        }
        let numbers = [
            p.usd_per_hour,
            p.network_s,
            p.input_work_per_token,
            p.image_work_per_megapixel,
            p.adapter_load_s,
            p.minimum_bill_s,
            p.startup.allocation_s,
            p.startup.container_s,
            p.startup.weights_s,
            p.startup.engine_s,
            p.startup.jitter,
        ];
        if numbers.into_iter().any(|v| !finite_nonnegative(v))
            || !p.network_bytes_per_s.is_finite()
            || p.network_bytes_per_s <= 0.0
            || p.startup.jitter > 1.0
            || !p.multi_adapter_factor.is_finite()
            || !(0.0..=1.0).contains(&p.multi_adapter_factor)
            || p.multi_adapter_factor == 0.0
        {
            return Err(format!("{} has invalid timing or pricing", p.id));
        }
        if p.startup
            .restore_samples_s
            .iter()
            .chain(&p.startup.build_samples_s)
            .any(|v| !finite_nonnegative(*v))
        {
            return Err("invalid empirical startup sample".into());
        }
        validate_host(&p.host)?;
        let mut previous = 0;
        for point in &p.curve {
            if point.in_flight <= previous
                || point.in_flight > p.max_in_flight
                || !point.work_units_per_s.is_finite()
                || point.work_units_per_s <= 0.0
            {
                return Err(format!("{} has invalid capacity curve", p.id));
            }
            previous = point.in_flight;
        }
    }
    if !scenario.horizon_s.is_finite()
        || scenario.horizon_s <= 0.0
        || !scenario.tick_s.is_finite()
        || scenario.tick_s < 0.001
        || scenario.tick_s > 1.0
        || !scenario.capacity_scale.is_finite()
        || scenario.capacity_scale <= 0.0
        || !finite_nonnegative(scenario.startup_scale)
        || !finite_nonnegative(scenario.idle_timeout_s)
        || !finite_nonnegative(scenario.reactive_delay_s)
        || !scenario.launch_fail_probability.is_finite()
        || !(0.0..=1.0).contains(&scenario.launch_fail_probability)
        || scenario.max_nodes_per_model == 0
    {
        return Err("invalid scenario timing, scale, limits, or probability".into());
    }
    for warm in &scenario.warm {
        if !profiles.iter().any(|p| p.id == warm.profile) {
            return Err(format!("unknown warm profile {}", warm.profile));
        }
    }
    let mut warm_counts = std::collections::HashMap::new();
    for warm in &scenario.warm {
        let p = profiles
            .iter()
            .find(|p| p.id == warm.profile)
            .expect("validated profile");
        *warm_counts.entry(&p.model).or_insert(0) += warm.copies;
    }
    if warm_counts
        .values()
        .any(|count| *count > scenario.max_nodes_per_model)
    {
        return Err("warm fleet exceeds max_nodes_per_model".into());
    }
    for (id, job) in scenario.jobs.iter().enumerate() {
        if job.id != id
            || !finite_nonnegative(job.announced_s)
            || !finite_nonnegative(job.arrival_s)
            || job.announced_s > job.arrival_s
            || !job.completion_budget_s.is_finite()
            || job.completion_budget_s <= 0.0
            || !finite_nonnegative(job.desired_output_tps)
        {
            return Err("invalid job identity or planning window".into());
        }
    }
    for (id, r) in scenario.requests.iter().enumerate() {
        validate_request_inputs(r, &scenario.scheduling)?;
        if r.id != id
            || r.job >= scenario.jobs.len()
            || !finite_nonnegative(r.arrival_s)
            || r.arrival_s < scenario.jobs[r.job].arrival_s
        {
            return Err(format!("invalid request {}", r.id));
        }
    }
    scenario.requests.iter().try_fold(0_u64, |total, r| {
        total
            .checked_add(r.actual_output)
            .ok_or("aggregate output token count overflows")
    })?;
    let reality = scenario.host_reality.as_ref();
    for p in profiles {
        let host = reality.unwrap_or(&p.host);
        let slowest = host
            .volume_bandwidth_bytes_per_s
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .ok_or("empty host distribution")?;
        let startup = p
            .startup
            .restore_samples_s
            .iter()
            .chain(&p.startup.build_samples_s)
            .copied()
            .fold(p.startup.total() * 8.0 * (1.0 + p.startup.jitter), f64::max);
        if !startup.is_finite()
            || !(startup * scenario.startup_scale
                + (host.container_image_bytes as f64 + host.model_bytes as f64) / slowest)
                .is_finite()
            || !(p.peak() * scenario.capacity_scale).is_finite()
            || !(p.usd_per_hour * scenario.horizon_s.max(p.minimum_bill_s)).is_finite()
            || scenario.requests.iter().any(|r| {
                p.compatible(r) && !(p.prefill_work(r) + r.actual_output as f64).is_finite()
            })
        {
            return Err("derived startup, service, work, or billing overflows".into());
        }
    }
    if scenario.faults.iter().any(|f| !finite_nonnegative(f.at_s)) {
        return Err("invalid fault time".into());
    }
    if !scenario.build_probability.is_finite() || !(0.0..=1.0).contains(&scenario.build_probability)
    {
        return Err("invalid build probability".into());
    }
    if let Some(host) = &scenario.host_reality {
        validate_host(host)?;
    }
    validate_scheduling(&scenario.scheduling)?;
    for r in &scenario.requests {
        if r.depends_on.iter().any(|id| *id >= r.id) {
            return Err("dependencies must reference earlier requests (acyclic)".into());
        }
    }
    Ok(())
}

/// Validate controller settings independently of a trace or execution backend.
pub fn validate_scheduling(s: &Scheduling) -> Result<(), String> {
    if !s.learning_rate.is_finite()
        || !(0.0..=1.0).contains(&s.learning_rate)
        || s.learning_rate == 0.0
    {
        return Err("invalid host learning rate".into());
    }
    if [
        s.plan_interval_s,
        s.reactive_drain_s,
        s.minimum_plan_window_s,
        s.aging_s,
        s.replan_output_floor,
        s.minimum_hydration_bytes_per_s,
        s.maximum_startup_s,
        s.retry_cooldown_s,
    ]
    .iter()
    .any(|v| !v.is_finite() || *v <= 0.0)
        || !s.warm_batch_capacity_fraction.is_finite()
        || !(0.0..=1.0).contains(&s.warm_batch_capacity_fraction)
        || s.classes
            .values()
            .any(|c| !c.weight.is_finite() || c.weight <= 0.0)
    {
        return Err("invalid scheduling control".into());
    }
    Ok(())
}

fn validate_host(host: &HostProfile) -> Result<(), String> {
    if host.volume_bandwidth_bytes_per_s.is_empty()
        || host
            .volume_bandwidth_bytes_per_s
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        || [
            host.image_cache_probability,
            host.model_cache_probability,
            host.measurement_noise,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || !host.probe_duration_s.is_finite()
        || host.probe_duration_s <= 0.0
        || host
            .container_image_bytes
            .checked_add(host.model_bytes)
            .is_none()
    {
        return Err("invalid host distribution".into());
    }
    Ok(())
}

/// Validate an individual accepted request without copying the experiment's
/// accumulated history. Identity and dependency checks remain trace-level.
pub fn validate_request_inputs(r: &Request, scheduling: &Scheduling) -> Result<(), String> {
    if r.input_tokens.checked_add(r.max_output).is_none()
        || r.actual_output == 0
        || r.planned_output == 0
        || r.actual_output > r.max_output
        || r.planned_output > r.max_output
        || !r.image_megapixels.is_finite()
        || r.image_megapixels < 0.0
        || !r.cached_fraction.is_finite()
        || !(0.0..=1.0).contains(&r.cached_fraction)
        || !r.latency_target_s.is_finite()
        || r.latency_target_s <= 0.0
    {
        return Err(format!("invalid request {}", r.id));
    }
    if !scheduling.classes.contains_key(&r.class) {
        return Err(format!("unknown service class {}", r.class));
    }
    Ok(())
}
