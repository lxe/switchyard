//! Reference inference scheduling policies and routing runtime.
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, VecDeque};
use switchyard_core::config::*;
pub use switchyard_core::scheduler::*;
use switchyard_core::stats::percentile;
mod learning;
mod policy;
pub mod runtime;

pub struct BuiltinScheduler {
    pub policy: Policy,
    observations: std::collections::BTreeMap<usize, HostProbe>,
    blocked_profiles: HashMap<usize, f64>,
    cancellations: usize,
    beliefs: BTreeMap<usize, HostBelief>,
    learning: learning::Learning,
    history_limit: usize,
}

#[derive(Debug, Clone, Serialize)]
struct HostBelief {
    samples: usize,
    volume_bytes_per_s: f64,
    hydration_bytes: f64,
    image_cache_hit_rate: f64,
    model_cache_hit_rate: f64,
}
impl HostBelief {
    fn observe(&mut self, probe: &HostProbe, alpha: f64) {
        let blend = |old, value| (1.0 - alpha) * old + alpha * value;
        self.volume_bytes_per_s = blend(self.volume_bytes_per_s, probe.volume_bytes_per_s);
        self.hydration_bytes = blend(self.hydration_bytes, probe.hydration_bytes as f64);
        if let Some(cached) = probe.image_cached {
            self.image_cache_hit_rate = blend(self.image_cache_hit_rate, f64::from(cached));
        }
        self.model_cache_hit_rate = blend(self.model_cache_hit_rate, f64::from(probe.model_cached));
        self.samples += 1;
    }
}

impl BuiltinScheduler {
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            observations: Default::default(),
            blocked_profiles: Default::default(),
            cancellations: 0,
            beliefs: BTreeMap::new(),
            learning: Default::default(),
            history_limit: Heuristics::default().history_limit,
        }
    }
}
impl Scheduler for BuiltinScheduler {
    fn name(&self) -> &str {
        match self.policy {
            Policy::WarmOnly => "warm_only",
            Policy::Reactive => "reactive",
            Policy::CheapestElastic => "cheapest_elastic",
            Policy::PlannedHybrid => "planned_hybrid",
            Policy::Adaptive => "adaptive",
        }
    }
    fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<Action> {
        self.history_limit = fleet.scheduling.heuristics.history_limit;
        if self.policy == Policy::Adaptive {
            self.learning.update(fleet);
        }
        let mut planned_job_outputs = HashMap::<usize, f64>::new();
        for request in fleet.requests.values() {
            *planned_job_outputs.entry(request.job).or_default() += request.planned_output as f64;
        }
        if self.policy == Policy::Adaptive {
            for (id, node) in fleet.nodes.iter().enumerate() {
                if let Some(probe) = &node.host.probe
                    && probe.fresh(fleet.now, &fleet.scheduling.probe)
                    && self
                        .observations
                        .get(&id)
                        .is_none_or(|old| probe.at_s > old.at_s)
                {
                    self.observations.insert(id, probe.clone());
                    self.beliefs
                        .entry(node.profile)
                        .and_modify(|belief| belief.observe(probe, fleet.scheduling.learning_rate))
                        .or_insert(HostBelief {
                            samples: 1,
                            volume_bytes_per_s: probe.volume_bytes_per_s,
                            hydration_bytes: probe.hydration_bytes as f64,
                            image_cache_hit_rate: probe.image_cached.map_or(
                                fleet.profiles[node.profile].host.image_cache_probability,
                                f64::from,
                            ),
                            model_cache_hit_rate: f64::from(probe.model_cached),
                        });
                }
            }
        }
        let mut world = SchedulerWorld {
            profiles: fleet.profiles,
            scenario: fleet,
            policy: self.policy,
            now: fleet.now,
            states: fleet.states.clone(),
            nodes: fleet.nodes.clone(),
            actions: vec![],
            avoided_profiles: fleet
                .provisioning_cooldowns
                .iter()
                .filter(|(_, until)| **until > fleet.now)
                .map(|(profile, _)| *profile)
                .collect(),
            startup_extra: self
                .beliefs
                .iter()
                .map(|(profile, belief)| {
                    (
                        *profile,
                        belief.hydration_bytes / belief.volume_bytes_per_s.max(1.0),
                    )
                })
                .collect(),
            startup_observed: (0..fleet.profiles.len())
                .filter_map(|p| self.learning.startup(p).map(|s| (p, s)))
                .collect(),
            placement_penalties: (0..fleet.profiles.len())
                .map(|p| {
                    (
                        p,
                        if self.policy == Policy::Adaptive {
                            self.learning.placement_penalty(p)
                        } else {
                            1.0
                        },
                    )
                })
                .collect(),
            learning: &self.learning,
            job_outputs: planned_job_outputs,
            output_estimates: if self.policy == Policy::Adaptive {
                fleet
                    .requests
                    .iter()
                    .map(|(id, r)| (*id, self.learning.estimate(r, &fleet.scheduling.heuristics)))
                    .collect()
            } else {
                Default::default()
            },
            progress: fleet
                .nodes
                .iter()
                .flat_map(|n| {
                    n.active.iter().map(move |a| {
                        let work = fleet.profiles[n.profile].prefill_work(&fleet.requests[&a.id]);
                        (a.id, (a.prefill / work.max(1e-8), a.generated))
                    })
                })
                .collect(),
        };
        // Pending intents contribute forecast capacity, but have no routable node ID.
        for pending in &fleet.pending_allocations {
            world.launch_forecast(pending.profile, pending.started_s, pending.expected_ready_s);
        }
        if self.policy == Policy::Adaptive {
            let can_cancel: Vec<_> = world
                .nodes
                .iter()
                .map(|node| {
                    node.placement.as_ref().is_none_or(|placement| {
                        world
                            .nodes
                            .iter()
                            .filter(|other| {
                                other
                                    .placement
                                    .as_ref()
                                    .is_some_and(|p| p.host == placement.host)
                            })
                            .all(|other| other.active.is_empty() && !other.is_ready)
                    })
                })
                .collect();
            for (id, node) in world.nodes.iter_mut().take(fleet.nodes.len()).enumerate() {
                if node.ended_s.is_some() || node.is_ready {
                    continue;
                }
                let timed_out = fleet.now - node.allocated_s > fleet.scheduling.maximum_startup_s;
                let slow = node.host.probe.as_ref().is_some_and(|probe| {
                    probe.fresh(fleet.now, &fleet.scheduling.probe)
                        && node.host.remaining_bytes_known
                        && node.host.bytes_total > node.host.bytes_downloaded
                        && probe.volume_bytes_per_s < fleet.scheduling.minimum_hydration_bytes_per_s
                        && node
                            .host
                            .bytes_total
                            .saturating_sub(node.host.bytes_downloaded)
                            as f64
                            / probe.volume_bytes_per_s
                            > fleet.scheduling.maximum_startup_s
                });
                if !timed_out
                    && !slow
                    && node.host.bytes_total > node.host.bytes_downloaded
                    && matches!(
                        node.host.phase,
                        StartupPhase::Hydrating | StartupPhase::Initializing
                    )
                    && fleet
                        .scheduling
                        .probe
                        .due(fleet.now, node.host.probe_attempted_s)
                    && !node.host.probe_pending
                {
                    world.actions.push(Action::ProbeHost { node: id });
                }
                if (slow || timed_out) && can_cancel[id] {
                    node.ended_s = Some(fleet.now);
                    world.actions.push(Action::Release { node: id });
                    self.cancellations += 1;
                    self.blocked_profiles
                        .insert(node.profile, fleet.now + fleet.scheduling.retry_cooldown_s);
                }
            }
            world.avoided_profiles.extend(
                self.blocked_profiles
                    .iter()
                    .filter(|(_, until)| **until > fleet.now)
                    .map(|(profile, _)| *profile),
            );
        }
        world.finalize_releases();
        if fleet.allow_scale {
            let releasable: Vec<_> = (0..world.nodes.len())
                .map(|id| world.host_idle(id))
                .collect();
            for (id, node) in world.nodes.iter_mut().enumerate() {
                if releasable[id]
                    && !node.warm
                    && node.ended_s.is_none()
                    && node.is_ready
                    && node.active.is_empty()
                    && fleet.now - node.last_used_s >= fleet.idle_timeout_s
                {
                    node.ended_s = Some(fleet.now);
                    world.actions.push(Action::Release { node: id });
                }
            }
            world.finalize_releases();
            world.plan();
        }
        world.dispatch();
        world.actions
    }
    fn observe(&mut self, feedback: SchedulerFeedback) {
        if self.policy == Policy::Adaptive {
            self.learning.observe(feedback, self.history_limit);
        }
    }
    fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({"probed_hosts": self.observations, "host_beliefs": self.beliefs, "cancelled_slow_starts": self.cancellations, "serving_learning": self.learning.diagnostics()})
    }
}

type State = RequestStatus;
type Active = ObservedActive;
struct SchedulerWorld<'a> {
    profiles: &'a [Profile],
    scenario: &'a FleetView<'a>,
    policy: Policy,
    now: f64,
    states: BTreeMap<usize, ObservedRequest>,
    nodes: Vec<ObservedNode>,
    actions: Vec<Action>,
    avoided_profiles: Vec<usize>,
    startup_extra: BTreeMap<usize, f64>,
    progress: HashMap<usize, (f64, f64)>,
    learning: &'a learning::Learning,
    job_outputs: HashMap<usize, f64>,
    output_estimates: HashMap<usize, u64>,
    startup_observed: BTreeMap<usize, f64>,
    placement_penalties: BTreeMap<usize, f64>,
}
impl SchedulerWorld<'_> {
    fn launch_forecast(&mut self, profile: usize, started: f64, ready: f64) {
        let begin = self.nodes.len();
        self.launch(profile, false);
        self.actions.pop();
        for node in &mut self.nodes[begin..] {
            node.allocated_s = started;
            node.ready_s = ready;
        }
    }
    fn launch(&mut self, profile: usize, warm: bool) {
        let host = self.nodes.len();
        let deployments: Vec<_> =
            if let Some(plan) = self.scenario.topology.plan_for(&self.profiles[profile].id) {
                plan.endpoints
                    .iter()
                    .map(|e| {
                        (
                            self.profiles
                                .iter()
                                .position(|p| p.id == e.profile)
                                .expect("validated topology"),
                            Some(plan.placement(host, e)),
                        )
                    })
                    .collect()
            } else {
                vec![(profile, None)]
            };
        for (profile, placement) in deployments {
            self.nodes.push(ObservedNode {
                placement,
                profile,
                warm,
                allocated_s: self.now,
                ready_s: self.now + self.predicted_start(&self.profiles[profile]),
                ended_s: None,
                active: vec![],
                adapters: VecDeque::new(),
                last_used_s: self.now,
                is_ready: false,
                host: HostObservation {
                    phase: StartupPhase::Booting,
                    bytes_downloaded: 0,
                    bytes_total: 0,
                    remaining_bytes_known: false,
                    probe_pending: false,
                    probe_attempted_s: None,
                    probe: None,
                },
            });
        }
        self.actions.push(Action::Provision { profile });
    }
    fn placement_fraction(&self, profile: &Profile) -> f64 {
        self.scenario
            .topology
            .plan_for(&profile.id)
            .map_or(1.0, |p| {
                p.endpoints
                    .iter()
                    .filter(|e| e.profile == profile.id)
                    .map(|e| e.compute_fraction * p.contention_factor)
                    .sum()
            })
    }
    fn finalize_releases(&mut self) {
        let mut released = std::collections::BTreeSet::new();
        self.actions.retain(|action| {
            if let Action::Release { node } = action {
                let host = self.nodes[*node]
                    .placement
                    .as_ref()
                    .map_or(*node, |p| p.host);
                return released.insert(host);
            }
            true
        });
        self.actions.retain(|action| match action {
            Action::ProbeHost { node } => !released.contains(
                &self.nodes[*node]
                    .placement
                    .as_ref()
                    .map_or(*node, |p| p.host),
            ),
            _ => true,
        });
        for (id, node) in self.nodes.iter_mut().enumerate() {
            if released.contains(&node.placement.as_ref().map_or(id, |p| p.host)) {
                node.ended_s = Some(self.now);
                node.is_ready = false;
            }
        }
    }
    fn host_idle(&self, id: usize) -> bool {
        let n = &self.nodes[id];
        n.placement.as_ref().is_none_or(|placement| {
            self.nodes
                .iter()
                .filter(|other| {
                    other
                        .placement
                        .as_ref()
                        .is_some_and(|p| p.host == placement.host)
                })
                .all(|other| {
                    other.active.is_empty()
                        && !other.warm
                        && other.is_ready
                        && self.now - other.last_used_s >= self.scenario.idle_timeout_s
                })
        })
    }
}

#[cfg(test)]
mod observation_tests {
    use super::*;
    #[test]
    fn unknown_image_cache_preserves_existing_belief() {
        let mut belief = HostBelief {
            samples: 1,
            volume_bytes_per_s: 100.0,
            hydration_bytes: 1000.0,
            image_cache_hit_rate: 0.75,
            model_cache_hit_rate: 0.0,
        };
        let mut probe = HostProbe {
            at_s: 1.0,
            started_s: 0.0,
            received_bytes: 200,
            volume_bytes_per_s: 200.0,
            hydration_bytes: 1000,
            image_cached: None,
            model_cached: false,
        };
        belief.observe(&probe, 0.5);
        assert_eq!(belief.image_cache_hit_rate, 0.75);
        assert_eq!(belief.volume_bytes_per_s, 150.0);
        probe.image_cached = Some(false);
        belief.observe(&probe, 0.5);
        assert_eq!(belief.image_cache_hit_rate, 0.375);
    }
}
