//! Deterministic discrete-event fleet lifecycle with bounded fluid GPU service.
//!
//! Arrivals, readiness, faults, and expiry are events. Active GPU work advances
//! on a configurable quantum. This is a macroscopic model, not a CUDA emulator.
use crate::config::*;

mod cloud;
mod results;
mod service;
use crate::observations::{Observations, Sampling};
use crate::scheduler::{
    Action, BuiltinScheduler, FleetView, HostObservation, HostProbe, ObservedActive, ObservedNode,
    ObservedRequest, PlannedRequest, RequestStatus, Scheduler, StartupPhase,
};
pub use cloud::{CloudSimulator, InferenceSpec};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

#[derive(Debug, Clone, Serialize)]
pub struct RequestResult {
    pub id: usize,
    pub job: usize,
    pub class: String,
    pub started_s: Option<f64>,
    pub first_token_s: Option<f64>,
    pub completed_s: Option<f64>,
    pub attempts: usize,
    pub rejected: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeResult {
    pub host: usize,
    pub endpoint: Option<String>,
    pub id: usize,
    pub profile: String,
    pub warm: bool,
    pub allocated_s: f64,
    pub ready_s: f64,
    pub ended_s: f64,
    pub busy_s: f64,
    pub emitted_tokens: f64,
    pub usd: f64,
    pub failed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobResult {
    pub name: String,
    pub requests: usize,
    pub completed: usize,
    pub completion_s: Option<f64>,
    pub elapsed_s: Option<f64>,
    pub output_tps: f64,
    pub desired_output_tps: f64,
    pub target_met: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub scenario: String,
    pub policy: String,
    pub seed: u64,
    pub completed: usize,
    pub rejected: usize,
    pub unfinished: usize,
    pub cost_usd: f64,
    pub idle_cost_usd: f64,
    pub classes: Vec<ClassResult>,
    pub queue_p95_s: Option<f64>,
    pub first_token_p95_s: Option<f64>,
    pub batch_elapsed_s: Option<f64>,
    pub batch_output_tps: f64,
    pub launched: usize,
    pub peak_nodes: usize,
    pub failed_launches: usize,
    pub capacity_rejections: usize,
    pub attempts: usize,
    pub output_tokens: u64,
    pub wasted_output_tokens: f64,
    /// Useful generated output in unfinished streams or network transit at cutoff.
    pub incomplete_output_tokens: f64,
    pub target_jobs_met: usize,
    pub target_jobs_total: usize,
    pub max_active_per_node: usize,
    pub max_reserved_kv_tokens: u64,
    pub profiles_used: Vec<String>,
    pub jobs: Vec<JobResult>,
    pub invalid_decisions: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassResult {
    pub name: String,
    pub arrived: usize,
    pub completed: usize,
    pub latency_p95_s: Option<f64>,
    pub slo_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Run {
    pub hosts: Vec<HostResult>,
    pub summary: Summary,
    pub requests: Vec<RequestResult>,
    pub nodes: Vec<NodeResult>,
    pub timeline: Vec<Timeline>,
    pub controller_diagnostics: serde_json::Value,
}

/// Rental accounting belongs to the physical allocation, not each endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct HostResult {
    pub id: usize,
    pub plan: Option<String>,
    pub endpoints: Vec<usize>,
    pub allocated_s: f64,
    pub ended_s: f64,
    pub usd: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Timeline {
    pub at_s: f64,
    pub ready: usize,
    pub starting: usize,
    pub queued: usize,
    pub active: usize,
    pub completed: usize,
    pub emitted_tokens: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Future,
    Queued,
    Active,
    Done,
    Rejected,
}
struct RequestState {
    state: State,
    result: RequestResult,
    delivery: Vec<OutputSegment>,
    egress_until_s: f64,
    node: Option<usize>,
    pinned: bool,
}
struct OutputSegment {
    start_s: f64,
    end_s: f64,
    before: f64,
    after: f64,
}
struct Active {
    id: usize,
    prefill: f64,
    output: f64,
    generated: f64,
    available_s: f64,
    weight: f64,
}
struct Node {
    placement: Option<switchyard_core::topology::EndpointPlacement>,
    first_use_s: f64,
    first_use_until_s: Option<f64>,
    initialized: bool,
    profile: usize,
    warm: bool,
    allocated_s: f64,
    ready_s: f64,
    ended_s: Option<f64>,
    active: Vec<Active>,
    adapters: VecDeque<(String, f64)>,
    last_used_s: f64,
    busy_s: f64,
    emitted: f64,
    failed: bool,
    host: HostState,
}
struct HostState {
    bandwidth: f64,
    image_cached: bool,
    model_cached: bool,
    bytes: u64,
    setup_until_s: f64,
    probe_duration_s: f64,
    noise: f64,
    probe_pending: bool,
    probe_attempted_s: Option<f64>,
    probe_window: switchyard_core::probe::BandwidthWindow,
    probe: Option<HostProbe>,
}

#[derive(Debug, Clone, Copy)]
enum EventKind {
    Arrival(usize),
    Ready(usize),
    Fault(usize),
    ProbeDone(usize),
    AllocationRejected(usize, f64),
    Tick,
}
#[derive(Debug, Clone, Copy)]
struct Event {
    at: f64,
    serial: usize,
    kind: EventKind,
}
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.at.to_bits() == other.at.to_bits() && self.serial == other.serial
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at
            .total_cmp(&self.at)
            .then_with(|| other.serial.cmp(&self.serial))
    }
}

/// Small fixed PRNG avoids dependency/platform differences in replay fixtures.
struct Random(u64);
impl Random {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((x ^ (x >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }
}

struct Simulation {
    observations: Observations,
    startup_indices: std::collections::BTreeMap<usize, usize>,
    provisioning_cooldowns: std::collections::BTreeMap<usize, f64>,
    pending_rejections: std::collections::BTreeMap<usize, (usize, f64)>,
    next_allocation_intent: usize,
    capacity_rejections: usize,
    reported_completions: std::collections::BTreeSet<usize>,
    profiles: Vec<Profile>,
    scenario: Scenario,
    policy: String,
    random: Random,
    probe_random: Random,
    output_random: Random,
    plans: Vec<PlannedRequest>,
    invalid_decisions: usize,
    states: Vec<RequestState>,
    nodes: Vec<Node>,
    events: BinaryHeap<Event>,
    serial: usize,
    now: f64,
    next_plan: f64,
    next_sample: f64,
    timeline: Vec<Timeline>,
    launches: usize,
    failed_launches: usize,
    startup_outcomes: std::collections::BTreeSet<usize>,
    peak_nodes: usize,
    max_active: usize,
    max_kv: u64,
    wasted_output: f64,
    pinned: VecDeque<(usize, usize)>,
    record_delivery: bool,
}

/// Run a configuration without performing any network requests or provisioning.
///
/// Replays with identical inputs and seed produce identical serialized results.
pub fn simulate(
    profiles: &[Profile],
    scenario: &Scenario,
    policy: Policy,
    seed: u64,
) -> Result<Run, String> {
    simulate_with_scheduler(profiles, scenario, &mut BuiltinScheduler::new(policy), seed)
}

/// Run any controller implementing [`Scheduler`] against the hardware simulator.
pub fn simulate_with_scheduler(
    profiles: &[Profile],
    scenario: &Scenario,
    controller: &mut dyn Scheduler,
    seed: u64,
) -> Result<Run, String> {
    simulate_with_observations(
        profiles,
        scenario,
        &Observations::default(),
        controller,
        seed,
    )
}

/// Replay empirical environment samples without revealing private outcomes to the controller.
pub fn simulate_with_observations(
    profiles: &[Profile],
    scenario: &Scenario,
    observations: &Observations,
    controller: &mut dyn Scheduler,
    seed: u64,
) -> Result<Run, String> {
    validate(profiles, scenario)?;
    observations.validate(profiles, scenario)?;
    let mut sim = Simulation::new(
        profiles.to_vec(),
        scenario.clone(),
        seed,
        controller.name().to_owned(),
    );
    sim.observations = observations.clone();
    sim.run_until(scenario.horizon_s, Some(controller));
    sim.sample();
    Ok(sim.finish(seed, controller.diagnostics()))
}

impl Simulation {
    fn new(profiles: Vec<Profile>, scenario: Scenario, seed: u64, policy: String) -> Self {
        let mut sim = Self {
            observations: Observations::default(),
            startup_indices: Default::default(),
            provisioning_cooldowns: Default::default(),
            pending_rejections: Default::default(),
            next_allocation_intent: 0,
            capacity_rejections: 0,
            reported_completions: Default::default(),
            policy,
            random: Random(seed),
            probe_random: Random(seed ^ 0x70726f6265),
            output_random: Random(seed ^ 0x6f7574707574),
            plans: scenario.requests.iter().map(PlannedRequest::from).collect(),
            invalid_decisions: 0,
            states: scenario
                .requests
                .iter()
                .map(|r| RequestState {
                    state: State::Future,
                    result: RequestResult {
                        id: r.id,
                        job: r.job,
                        class: r.class.clone(),
                        started_s: None,
                        first_token_s: None,
                        completed_s: None,
                        attempts: 0,
                        rejected: false,
                    },
                    delivery: vec![],
                    egress_until_s: 0.0,
                    node: None,
                    pinned: false,
                })
                .collect(),
            nodes: vec![],
            events: BinaryHeap::new(),
            serial: 0,
            now: 0.0,
            next_plan: 0.0,
            next_sample: 0.0,
            timeline: vec![],
            launches: 0,
            failed_launches: 0,
            startup_outcomes: Default::default(),
            peak_nodes: 0,
            max_active: 0,
            max_kv: 0,
            wasted_output: 0.0,
            profiles,
            scenario,
            pinned: VecDeque::new(),
            record_delivery: false,
        };
        for warm in sim.scenario.warm.clone() {
            let p = sim
                .profiles
                .iter()
                .position(|p| p.id == warm.profile)
                .expect("validated warm profile");
            for _ in 0..warm.copies {
                sim.launch(p, true);
            }
        }
        for r in sim.scenario.requests.clone() {
            sim.push(r.arrival_s, EventKind::Arrival(r.id));
        }
        for f in sim.scenario.faults.clone() {
            sim.push(f.at_s, EventKind::Fault(f.node));
        }
        sim.push(0.0, EventKind::Tick);
        sim
    }

    fn run_until(&mut self, until: f64, mut controller: Option<&mut dyn Scheduler>) {
        while self.events.peek().is_some_and(|event| event.at <= until) {
            let at = self.events.peek().expect("peeked event").at;
            self.advance(at);
            self.now = at;
            let mut allow_scale = false;
            // A burst at one instant is one observed arrival set. Calling a
            // controller for each row would repeatedly copy the entire plan
            // and introduce allocation-order bias before time has advanced.
            while self.events.peek().is_some_and(|event| event.at == at) {
                let event = self.events.pop().expect("same-time event");
                match event.kind {
                    EventKind::Arrival(id) => {
                        if self
                            .profiles
                            .iter()
                            .any(|p| p.compatible(&self.scenario.requests[id]))
                        {
                            self.states[id].state = State::Queued;
                        } else {
                            self.states[id].state = State::Rejected;
                            self.states[id].result.rejected = true;
                        }
                    }
                    EventKind::Ready(id) => {
                        if self.nodes[id].ended_s.is_none() {
                            self.nodes[id].last_used_s = self.now;
                            let host = self.nodes[id].placement.as_ref().map_or(id, |p| p.host);
                            if self.startup_outcomes.insert(host)
                                && self.random.unit() < self.scenario.launch_fail_probability
                            {
                                self.fail(id);
                                self.failed_launches += 1;
                            } else {
                                self.nodes[id].initialized = true;
                            }
                        }
                    }
                    EventKind::Fault(id) => {
                        if id < self.nodes.len() {
                            if let Some(controller) = controller.as_deref_mut() {
                                controller.observe(
                                    switchyard_core::scheduler::SchedulerFeedback::FailedRequest {
                                        node: id,
                                    },
                                );
                            }
                            self.fail(id);
                        }
                    }
                    EventKind::AllocationRejected(intent, elapsed_s) => {
                        if let Some((profile, _)) = self.pending_rejections.remove(&intent) {
                            self.capacity_rejections += 1;
                            let retry = self.now + self.scenario.scheduling.retry_cooldown_s;
                            self.provisioning_cooldowns.insert(profile, retry);
                            if let Some(plan) =
                                self.scenario.topology.plan_for(&self.profiles[profile].id)
                            {
                                for endpoint in &plan.endpoints {
                                    let index = self
                                        .profiles
                                        .iter()
                                        .position(|p| p.id == endpoint.profile)
                                        .expect("validated topology");
                                    self.provisioning_cooldowns.insert(index, retry);
                                }
                            }
                            if let Some(controller) = controller.as_deref_mut() {
                                controller.observe(
                                    switchyard_core::scheduler::SchedulerFeedback::Allocation {
                                        profile,
                                        accepted: false,
                                        elapsed_s,
                                    },
                                );
                            }
                        }
                    }
                    EventKind::ProbeDone(id) => {
                        let node = &mut self.nodes[id];
                        node.host.probe_pending = false;
                        if node.ended_s.is_none() {
                            let duration =
                                self.now - node.host.probe_attempted_s.expect("started probe");
                            let received = (duration
                                * node.host.bandwidth
                                * (1.0 + (self.probe_random.unit() * 2.0 - 1.0) * node.host.noise))
                                as u64;
                            node.host.probe_window.record(
                                self.now,
                                vec![switchyard_core::probe::TransferCounter {
                                    source: "simulated-active-probe".into(),
                                    total_bytes: 0,
                                    received_bytes: received,
                                }],
                                &self.scenario.scheduling.probe,
                            );
                            if let Some(sample) = node
                                .host
                                .probe_window
                                .qualify(&self.scenario.scheduling.probe)
                            {
                                node.host.probe = Some(HostProbe {
                                    at_s: sample.at_s,
                                    started_s: sample.started_s,
                                    received_bytes: sample.received_bytes,
                                    volume_bytes_per_s: sample.bytes_per_s,
                                    image_cached: Some(node.host.image_cached),
                                    model_cached: node.host.model_cached,
                                    hydration_bytes: node.host.bytes,
                                });
                            }
                        }
                    }
                    EventKind::Tick => {
                        if self.now >= self.next_plan {
                            allow_scale = true;
                            self.next_plan = self.now + self.scenario.scheduling.plan_interval_s;
                        }
                        if controller.is_some() && self.now >= self.next_sample {
                            self.sample();
                            self.next_sample = self.now + 1.0;
                        }
                        self.push(self.now + self.scenario.tick_s, EventKind::Tick);
                    }
                }
            }
            self.dispatch_pinned();
            if let Some(controller) = controller.as_deref_mut()
                && (allow_scale || self.states.iter().any(|s| s.state == State::Queued))
            {
                self.control(controller, allow_scale);
            }
        }
        self.advance(until);
        self.now = until;
    }

    fn dispatch_pinned(&mut self) {
        let pending = std::mem::take(&mut self.pinned);
        for (id, node) in pending {
            if self.nodes[node].ended_s.is_some() {
                self.states[id].state = State::Rejected;
                self.states[id].result.rejected = true;
            } else if !self.admit(id, node, 1.0) {
                self.pinned.push_back((id, node));
            }
        }
    }

    fn push(&mut self, at: f64, kind: EventKind) {
        self.serial += 1;
        self.events.push(Event {
            at,
            serial: self.serial,
            kind,
        });
    }
    fn launch(&mut self, profile: usize, warm: bool) {
        let host = self.nodes.len();
        if let Some(plan) = self
            .scenario
            .topology
            .plan_for(&self.profiles[profile].id)
            .cloned()
        {
            for endpoint in &plan.endpoints {
                let profile = self
                    .profiles
                    .iter()
                    .position(|p| p.id == endpoint.profile)
                    .expect("validated host plan");
                self.launch_endpoint(profile, warm);
                self.nodes.last_mut().unwrap().placement = Some(plan.placement(host, endpoint));
            }
            if !warm {
                self.launches -= plan.endpoints.len() - 1;
            }
        } else {
            self.launch_endpoint(profile, warm);
        }
    }
    fn launch_endpoint(&mut self, profile: usize, warm: bool) {
        let p = &self.profiles[profile];
        let jitter = 1.0 + (self.random.unit() * 2.0 - 1.0) * p.startup.jitter;
        let build = self.random.unit() < self.scenario.build_probability;
        let samples = if build {
            &p.startup.build_samples_s
        } else {
            &p.startup.restore_samples_s
        };
        let total = if samples.is_empty() {
            p.startup.total() * jitter * if build { 8.0 } else { 1.0 }
        } else {
            samples[(self.random.unit() * samples.len() as f64) as usize]
        };
        let reality = self.scenario.host_reality.as_ref().unwrap_or(&p.host);
        let bandwidth = reality.volume_bandwidth_bytes_per_s
            [(self.random.unit() * reality.volume_bandwidth_bytes_per_s.len() as f64) as usize];
        let image_cached = warm || self.random.unit() < reality.image_cache_probability;
        let model_cached = warm || self.random.unit() < reality.model_cache_probability;
        let bytes = if image_cached {
            0
        } else {
            reality.container_image_bytes
        } + if model_cached { 0 } else { reality.model_bytes };
        let mut setup_until_s = self.now
            + if warm {
                0.0
            } else {
                total * self.scenario.startup_scale * 0.25
            };
        let mut ready_s = self.now
            + if warm {
                0.0
            } else {
                total * self.scenario.startup_scale + bytes as f64 / bandwidth
            };
        let mut bandwidth = bandwidth;
        let mut bytes = bytes;
        let mut first_use_s = 0.0;
        if !warm
            && let Some(reality) = self.observations.profiles.get(&p.id)
            && !reality.startups.is_empty()
        {
            let index = self.startup_indices.entry(profile).or_default();
            let chosen = match reality.sampling {
                Sampling::Sequential => *index % reality.startups.len(),
                Sampling::Bootstrap => {
                    (self.random.unit() * reality.startups.len() as f64) as usize
                }
            };
            *index += 1;
            let sample = &reality.startups[chosen];
            let scale = self.scenario.startup_scale;
            setup_until_s = self.now + sample.boot_s * scale;
            ready_s = self.now + sample.ready_s * scale;
            bytes = if scale == 0.0 {
                0
            } else {
                sample.hydration_bytes
            };
            bandwidth = if bytes > 0 {
                bytes as f64 / (sample.hydration_s * scale)
            } else {
                bandwidth
            };
            first_use_s = sample.first_use_s;
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            placement: None,
            first_use_s,
            first_use_until_s: None,
            initialized: warm,
            profile,
            warm,
            allocated_s: self.now,
            ready_s,
            ended_s: None,
            active: vec![],
            adapters: VecDeque::new(),
            last_used_s: self.now,
            busy_s: 0.0,
            emitted: 0.0,
            failed: false,
            host: HostState {
                bandwidth,
                image_cached,
                model_cached,
                bytes,
                setup_until_s,
                probe_duration_s: reality.probe_duration_s,
                noise: reality.measurement_noise,
                probe_pending: false,
                probe_attempted_s: None,
                probe_window: switchyard_core::probe::BandwidthWindow::default(),
                probe: None,
            },
        });
        if !warm {
            self.launches += 1;
            self.push(ready_s, EventKind::Ready(id));
        }
        self.peak_nodes = self
            .peak_nodes
            .max(self.nodes.iter().filter(|n| n.ended_s.is_none()).count());
    }
    fn fail(&mut self, id: usize) {
        let host = self.nodes[id].placement.as_ref().map_or(id, |p| p.host);
        let siblings: Vec<_> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| n.placement.as_ref().map_or(*i, |p| p.host) == host)
            .map(|(i, _)| i)
            .collect();
        for id in siblings {
            self.fail_endpoint(id);
        }
    }
    fn fail_endpoint(&mut self, id: usize) {
        let n = &mut self.nodes[id];
        if n.ended_s.is_some() {
            return;
        }
        n.ended_s = Some(self.now);
        n.failed = true;
        for a in n.active.drain(..) {
            self.wasted_output += a.generated;
            self.states[a.id].state = if self.states[a.id].pinned {
                State::Rejected
            } else {
                State::Queued
            };
            self.states[a.id].result.rejected = self.states[a.id].pinned;
            // Restart failed streams from their full input/output; no double completion.
            self.states[a.id].result.first_token_s = None;
            self.states[a.id].delivery.clear();
            self.states[a.id].egress_until_s = self.now;
        }
    }
    fn control(&mut self, controller: &mut dyn Scheduler, allow_scale: bool) {
        if self
            .observations
            .controller_pauses
            .iter()
            .any(|p| p.start_s <= self.now && self.now < p.end_s)
        {
            return;
        }
        // Reveal output usage only once it has reached the client.
        for (id, state) in self.states.iter().enumerate() {
            if state.result.completed_s.is_some_and(|at| at <= self.now)
                && self.reported_completions.insert(id)
                && let Some(node) = state.node
            {
                controller.observe(switchyard_core::scheduler::SchedulerFeedback::Completion {
                    node,
                    profile: self.nodes[node].profile,
                    request: Box::new(self.plans[id].clone()),
                    output_tokens: self.scenario.requests[id].actual_output,
                    input_work: self.profiles[self.nodes[node].profile]
                        .prefill_work(&self.plans[id]),
                    started_s: state.result.started_s.unwrap_or(self.now),
                    completed_s: state.result.completed_s.unwrap(),
                });
            }
        }
        let view = FleetView {
            topology: &self.scenario.topology,
            now: self.now,
            allow_scale,
            profiles: &self.profiles,
            provisioning_cooldowns: self.provisioning_cooldowns.clone(),
            pending_allocations: self
                .pending_rejections
                .values()
                .map(
                    |(profile, started_s)| switchyard_core::scheduler::PendingAllocation {
                        profile: *profile,
                        started_s: *started_s,
                        expected_ready_s: *started_s + self.profiles[*profile].startup.total(),
                    },
                )
                .collect(),
            requests: self
                .plans
                .iter()
                .filter(|r| self.scenario.jobs[r.job].announced_s <= self.now)
                .map(|r| {
                    let mut plan = r.clone();
                    plan.routing_allowed = !self.states[r.id].pinned;
                    (r.id, plan)
                })
                .collect(),
            jobs: self
                .scenario
                .jobs
                .iter()
                .filter(|j| j.announced_s <= self.now)
                .map(|j| (j.id, j.clone()))
                .collect(),
            scheduling: &self.scenario.scheduling,
            states: self
                .states
                .iter()
                .enumerate()
                .filter(|(id, _)| {
                    self.scenario.jobs[self.scenario.requests[*id].job].announced_s <= self.now
                })
                .map(|(id, s)| {
                    (
                        id,
                        ObservedRequest {
                            state: match s.state {
                                State::Future => RequestStatus::Future,
                                State::Queued => RequestStatus::Queued,
                                State::Active => RequestStatus::Active,
                                State::Done => {
                                    if s.result.completed_s.is_some_and(|t| t <= self.now) {
                                        RequestStatus::Done
                                    } else {
                                        RequestStatus::Delivering
                                    }
                                }
                                State::Rejected => RequestStatus::Rejected,
                            },
                            completed_s: s.result.completed_s.filter(|t| *t <= self.now),
                        },
                    )
                })
                .collect(),
            nodes: self
                .nodes
                .iter()
                .map(|n| {
                    let p = &self.profiles[n.profile];
                    let host = self.host_observation(n);
                    let rate = host
                        .probe
                        .as_ref()
                        .filter(|probe| probe.fresh(self.now, &self.scenario.scheduling.probe))
                        .map_or(p.host.volume_bandwidth_bytes_per_s[0], |probe| {
                            probe.volume_bytes_per_s
                        });
                    let estimated = if n.ended_s.is_none() && n.initialized {
                        n.ready_s
                    } else {
                        (n.allocated_s + p.startup.total()).max(self.now + 1.0)
                            + host.bytes_total.saturating_sub(host.bytes_downloaded) as f64 / rate
                    };
                    ObservedNode {
                        placement: n.placement.clone(),
                        profile: n.profile,
                        warm: n.warm,
                        allocated_s: n.allocated_s,
                        ready_s: estimated,
                        ended_s: n.ended_s,
                        active: n
                            .active
                            .iter()
                            .map(|a| ObservedActive {
                                id: a.id,
                                prefill: a.prefill,
                                generated: a.generated,
                                available_s: a.available_s,
                            })
                            .collect(),
                        adapters: n.adapters.clone(),
                        last_used_s: n.last_used_s,
                        is_ready: n.initialized && n.ended_s.is_none(),
                        host,
                    }
                })
                .collect(),
            max_nodes_per_model: self.scenario.max_nodes_per_model,
            idle_timeout_s: self.scenario.idle_timeout_s,
            reactive_delay_s: self.scenario.reactive_delay_s,
            build_probability: self.scenario.build_probability,
        };
        let actions = controller.decide(&view);
        for action in actions {
            let valid = match action {
                Action::Provision { profile } => {
                    if let Some(p) = self.profiles.get(profile) {
                        if allow_scale && self.placement_admitted(profile) {
                            if !self.capacity_available(profile) {
                                let reality = &self.observations.profiles[&p.id];
                                let delay = reality.rejection_delay_s;
                                let intent = self.next_allocation_intent;
                                self.next_allocation_intent += 1;
                                self.pending_rejections.insert(intent, (profile, self.now));
                                self.push(
                                    self.now + delay,
                                    EventKind::AllocationRejected(intent, delay),
                                );
                                // Allocation completion is asynchronous; hot routing proceeds.
                            } else {
                                self.launch(profile, false);
                                controller.observe(
                                    switchyard_core::scheduler::SchedulerFeedback::Allocation {
                                        profile,
                                        accepted: true,
                                        elapsed_s: 0.0,
                                    },
                                );
                            }
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                Action::Release { node } => {
                    if let Some(n) = self.nodes.get(node) {
                        let host = n.placement.as_ref().map_or(node, |p| p.host);
                        let siblings: Vec<_> = self
                            .nodes
                            .iter()
                            .enumerate()
                            .filter(|(i, n)| n.placement.as_ref().map_or(*i, |p| p.host) == host)
                            .map(|(i, _)| i)
                            .collect();
                        if siblings.iter().all(|i| self.nodes[*i].active.is_empty()) {
                            for id in siblings {
                                self.nodes[id].ended_s = Some(self.now);
                            }
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                Action::Route {
                    request,
                    node,
                    service_weight,
                } => self.admit(request, node, service_weight),
                Action::ProbeHost { node } => self.start_probe(node),
            };
            if !valid {
                self.invalid_decisions += 1;
            }
        }
    }

    fn placement_admitted(&self, profile: usize) -> bool {
        let mut additions = std::collections::BTreeMap::<&str, usize>::new();
        if let Some(plan) = self.scenario.topology.plan_for(&self.profiles[profile].id) {
            for endpoint in &plan.endpoints {
                let p = self
                    .profiles
                    .iter()
                    .find(|p| p.id == endpoint.profile)
                    .expect("validated topology");
                *additions.entry(&p.model).or_default() += 1;
            }
        } else {
            additions.insert(&self.profiles[profile].model, 1);
        }
        for (pending, _) in self.pending_rejections.values() {
            if let Some(plan) = self.scenario.topology.plan_for(&self.profiles[*pending].id) {
                for endpoint in &plan.endpoints {
                    let p = self
                        .profiles
                        .iter()
                        .find(|p| p.id == endpoint.profile)
                        .expect("validated topology");
                    *additions.entry(&p.model).or_default() += 1;
                }
            } else {
                *additions.entry(&self.profiles[*pending].model).or_default() += 1;
            }
        }
        additions.into_iter().all(|(model, count)| {
            count
                + self
                    .nodes
                    .iter()
                    .filter(|n| n.ended_s.is_none() && self.profiles[n.profile].model == model)
                    .count()
                <= self.scenario.max_nodes_per_model
        })
    }
    fn capacity_available(&self, profile: usize) -> bool {
        let Some(reality) = self.observations.profiles.get(&self.profiles[profile].id) else {
            return true;
        };
        let limit = reality.capacity.iter().rev().find(|l| l.at_s <= self.now);
        let live = self
            .nodes
            .iter()
            .filter(|n| n.profile == profile && n.ended_s.is_none())
            .count();
        limit.is_none_or(|l| live < l.instances)
    }

    fn start_probe(&mut self, id: usize) -> bool {
        let Some(node) = self.nodes.get_mut(id) else {
            return false;
        };
        let policy = &self.scenario.scheduling.probe;
        if node.ended_s.is_some()
            || node.host.probe_pending
            || self.now < node.host.setup_until_s
            || !policy.due(self.now, node.host.probe_attempted_s)
        {
            return false;
        }
        node.host.probe_pending = true;
        node.host.probe_attempted_s = Some(self.now);
        node.host.probe_window = switchyard_core::probe::BandwidthWindow::default();
        node.host.probe_window.record(
            self.now,
            vec![switchyard_core::probe::TransferCounter {
                source: "simulated-active-probe".into(),
                total_bytes: 0,
                received_bytes: 0,
            }],
            policy,
        );
        // The existing synthetic active probe must now measure enough time and
        // payload. Very slow hosts finish an unqualified bounded measurement.
        let duration = node.host.probe_duration_s.max(policy.minimum_window_s).max(
            (policy.minimum_received_bytes as f64 / node.host.bandwidth)
                .min(policy.maximum_window_s),
        );
        self.push(self.now + duration, EventKind::ProbeDone(id));
        true
    }

    fn host_observation(&self, n: &Node) -> HostObservation {
        let bytes = ((self.now - n.host.setup_until_s).max(0.0) * n.host.bandwidth)
            .min(n.host.bytes as f64) as u64;
        let phase = if n.ended_s.is_some() {
            StartupPhase::Ended
        } else if n.initialized {
            StartupPhase::Ready
        } else if self.now < n.host.setup_until_s {
            StartupPhase::Booting
        } else if bytes < n.host.bytes {
            StartupPhase::Hydrating
        } else {
            StartupPhase::Initializing
        };
        HostObservation {
            phase,
            bytes_downloaded: bytes,
            bytes_total: if self.now >= n.host.setup_until_s {
                n.host.bytes
            } else {
                0
            },
            remaining_bytes_known: true,
            probe_pending: n.host.probe_pending,
            probe_attempted_s: n.host.probe_attempted_s,
            probe: n.host.probe.clone(),
        }
    }

    fn admit(&mut self, id: usize, index: usize, weight: f64) -> bool {
        if id >= self.states.len()
            || index >= self.nodes.len()
            || !weight.is_finite()
            || weight <= 0.0
            || self.states[id].state != State::Queued
        {
            return false;
        }
        let r = &self.scenario.requests[id];
        if !r.depends_on.iter().all(|dependency| {
            self.states[*dependency]
                .result
                .completed_s
                .is_some_and(|t| t <= self.now)
        }) {
            return false;
        }
        let n = &mut self.nodes[index];
        let p = &self.profiles[n.profile];
        let used = n.active.iter().try_fold(0_u64, |total, a| {
            let r = &self.scenario.requests[a.id];
            total.checked_add(r.input_tokens + r.max_output)
        });
        if n.ended_s.is_some()
            || !n.initialized
            || !p.compatible(r)
            || n.active.len() >= p.max_in_flight
            || !used
                .and_then(|used| used.checked_add(r.input_tokens + r.max_output))
                .is_some_and(|used| used <= p.kv_tokens)
        {
            return false;
        }
        let upload = r.upload_bytes as f64 / p.network_bytes_per_s;
        let first_use_until_s = *n.first_use_until_s.get_or_insert(self.now + n.first_use_s);
        let adapter_ready = if let Some(adapter) = &r.adapter {
            if let Some(pos) = n.adapters.iter().position(|(a, _)| a == adapter) {
                let entry = n.adapters.remove(pos).expect("existing adapter");
                let ready = entry.1;
                n.adapters.push_back(entry);
                ready
            } else {
                if n.adapters.len() == p.adapter_cache_slots {
                    let Some(pos) = n.adapters.iter().position(|(a, _)| {
                        !n.active.iter().any(|active| {
                            self.scenario.requests[active.id].adapter.as_ref() == Some(a)
                        })
                    }) else {
                        return false;
                    };
                    n.adapters.remove(pos);
                }
                let ready = self.now + p.network_s + upload + p.adapter_load_s;
                n.adapters.push_back((adapter.clone(), ready));
                ready
            }
        } else {
            self.now
        };
        n.active.push(Active {
            id,
            prefill: p.prefill_work(r),
            output: r.actual_output as f64,
            generated: 0.0,
            available_s: (self.now + p.network_s + upload)
                .max(adapter_ready)
                .max(first_use_until_s),
            weight,
        });
        n.last_used_s = self.now;
        self.max_active = self.max_active.max(n.active.len());
        self.max_kv = self.max_kv.max(
            n.active
                .iter()
                .map(|a| {
                    let r = &self.scenario.requests[a.id];
                    r.input_tokens + r.max_output
                })
                .sum(),
        );
        self.states[id].state = State::Active;
        self.states[id].node = Some(index);
        let result = &mut self.states[id].result;
        result.started_s.get_or_insert(self.now);
        result.attempts += 1;
        true
    }
    fn sample(&mut self) {
        self.timeline.push(Timeline {
            at_s: self.now,
            ready: self
                .nodes
                .iter()
                .filter(|n| n.ended_s.is_none() && n.initialized)
                .count(),
            starting: self
                .nodes
                .iter()
                .filter(|n| n.ended_s.is_none() && !n.initialized)
                .count(),
            queued: self
                .states
                .iter()
                .filter(|s| s.state == State::Queued)
                .count(),
            active: self.nodes.iter().map(|n| n.active.len()).sum(),
            completed: self
                .states
                .iter()
                .filter(|s| s.result.completed_s.is_some_and(|t| t <= self.now))
                .count(),
            emitted_tokens: self.nodes.iter().map(|n| n.emitted).sum(),
        });
    }
}

pub use switchyard_core::stats::percentile;
