use std::collections::{BTreeMap, VecDeque};
use switchyard_gauntlet::scheduler::{
    BuiltinScheduler, HostObservation, HostProbe, ObservedNode, ObservedRequest, PlannedRequest,
    RequestStatus, StartupPhase,
};
use switchyard_gauntlet::*;

#[test]
fn smaller_requests_can_fill_residual_memory_without_starving_the_large_head() {
    let (mut profiles, mut trace) = performance::workload(3);
    profiles.truncate(1);
    profiles[0].max_in_flight = 3;
    profiles[0].kv_tokens = 1800;
    profiles[0].curve = vec![
        CapacityPoint {
            in_flight: 1,
            work_units_per_s: 100.0,
        },
        CapacityPoint {
            in_flight: 3,
            work_units_per_s: 100.0,
        },
    ];
    profiles[0].input_work_per_token = 0.0;
    profiles[0].network_s = 0.0;
    for request in &mut trace.requests {
        request.max_output = 100;
        request.planned_output = 100;
        request.actual_output = 100;
    }
    trace.requests[0].input_tokens = 900;
    trace.requests[1].input_tokens = 1000;
    trace.requests[2].input_tokens = 200;
    trace.requests[2].actual_output = 10;
    let filled = simulate(&profiles, &trace, Policy::WarmOnly, 7).unwrap();
    trace.scheduling.allow_queue_overtaking = false;
    let fifo = simulate(&profiles, &trace, Policy::WarmOnly, 7).unwrap();
    assert_eq!(filled.requests[2].started_s, Some(0.0));
    assert!(fifo.requests[2].started_s.unwrap() > 0.0);
    assert!(filled.requests[1].completed_s.is_some());
    assert_eq!(filled.summary.invalid_decisions, 0);
    assert!(filled.summary.max_reserved_kv_tokens <= 1800);
}

#[test]
fn learned_hydration_time_changes_provisioning_choices() {
    let (mut profiles, trace) = performance::workload(1);
    profiles[0].id = "cheap-unmeasured".into();
    profiles[0].usd_per_hour = 0.5;
    profiles[0].startup.engine_s = 1.0;
    let mut alternative = profiles[0].clone();
    alternative.id = "alternative".into();
    alternative.usd_per_hour = 1.0;
    alternative.startup.engine_s = 2.0;
    profiles.push(alternative);
    let plans = BTreeMap::from([(0, PlannedRequest::from(&trace.requests[0]))]);
    let states = BTreeMap::from([(
        0,
        ObservedRequest {
            state: RequestStatus::Queued,
            completed_s: None,
        },
    )]);
    let jobs = BTreeMap::from([(0, trace.jobs[0].clone())]);
    let mut fleet = FleetView {
        now: 0.0,
        allow_scale: true,
        profiles: &profiles,
        provisioning_cooldowns: BTreeMap::new(),
        pending_allocations: vec![],
        requests: plans,
        states,
        nodes: vec![],
        jobs,
        scheduling: &trace.scheduling,
        max_nodes_per_model: 2,
        idle_timeout_s: 30.0,
        reactive_delay_s: 0.0,
        build_probability: 0.0,
    };
    let before = BuiltinScheduler::new(Policy::Adaptive).decide(&fleet);
    assert!(
        before
            .iter()
            .any(|a| matches!(a, Action::Provision { profile: 0 }))
    );
    fleet.nodes.push(ObservedNode {
        profile: 0,
        warm: false,
        allocated_s: 0.0,
        ready_s: 0.0,
        ended_s: Some(0.0),
        active: vec![],
        adapters: VecDeque::new(),
        last_used_s: 0.0,
        is_ready: false,
        host: HostObservation {
            phase: StartupPhase::Ended,
            bytes_downloaded: 0,
            bytes_total: 1_000_000_000,
            remaining_bytes_known: true,
            probe_pending: false,
            probe_attempted_s: Some(0.0),
            probe: Some(HostProbe {
                at_s: 10.0,
                started_s: 0.0,
                received_bytes: 10_000_000,
                volume_bytes_per_s: 1_000_000.0,
                hydration_bytes: 1_000_000_000,
                image_cached: Some(false),
                model_cached: false,
            }),
        },
    });
    fleet.now = 10.0;
    let mut learned = BuiltinScheduler::new(Policy::Adaptive);
    let after = learned.decide(&fleet);
    assert!(
        after
            .iter()
            .any(|a| matches!(a, Action::Provision { profile: 1 }))
    );
    assert!(
        !after
            .iter()
            .any(|a| matches!(a, Action::Provision { profile: 0 }))
    );
    assert_eq!(learned.diagnostics()["host_beliefs"]["0"]["samples"], 1);
    learned.decide(&fleet);
    assert_eq!(
        learned.diagnostics()["host_beliefs"]["0"]["samples"],
        1,
        "polling must not count a probe twice"
    );
}

#[test]
fn allocation_cooldowns_change_new_placement_and_preserve_existing_host_routing() {
    let (mut profiles, trace) = performance::workload(1);
    profiles.truncate(1);
    profiles[0].usd_per_hour = 0.1;
    let mut alternative = profiles[0].clone();
    alternative.id = "alternative".into();
    alternative.usd_per_hour = 10.0;
    profiles.push(alternative);
    let mut fleet = FleetView {
        now: 0.0,
        allow_scale: true,
        profiles: &profiles,
        provisioning_cooldowns: BTreeMap::new(),
        pending_allocations: vec![],
        requests: BTreeMap::from([(0, PlannedRequest::from(&trace.requests[0]))]),
        states: BTreeMap::from([(
            0,
            ObservedRequest {
                state: RequestStatus::Queued,
                completed_s: None,
            },
        )]),
        nodes: vec![],
        jobs: BTreeMap::from([(0, trace.jobs[0].clone())]),
        scheduling: &trace.scheduling,
        max_nodes_per_model: 1,
        idle_timeout_s: 30.0,
        reactive_delay_s: 0.0,
        build_probability: 0.0,
    };
    for policy in POLICIES
        .into_iter()
        .filter(|policy| *policy != Policy::WarmOnly)
    {
        fleet.now = 0.0;
        fleet.provisioning_cooldowns.clear();
        let mut scheduler = BuiltinScheduler::new(policy);
        assert!(
            scheduler
                .decide(&fleet)
                .iter()
                .any(|action| matches!(action, Action::Provision { profile: 0 })),
            "{policy:?} prefers the cheaper available profile"
        );
        fleet.provisioning_cooldowns.insert(0, 10.0);
        let alternate = scheduler.decide(&fleet);
        assert!(
            alternate
                .iter()
                .any(|action| matches!(action, Action::Provision { profile: 1 })),
            "{policy:?} uses another profile during cooldown"
        );
        assert!(
            !alternate
                .iter()
                .any(|action| matches!(action, Action::Provision { profile: 0 }))
        );
        fleet.now = 10.0;
        assert!(
            scheduler
                .decide(&fleet)
                .iter()
                .any(|action| matches!(action, Action::Provision { profile: 0 })),
            "{policy:?} permits a new attempt at expiry"
        );
    }
    fleet.now = 0.0;
    fleet.provisioning_cooldowns.insert(0, 10.0);
    fleet.nodes.push(ObservedNode {
        profile: 0,
        warm: true,
        allocated_s: 0.0,
        ready_s: 0.0,
        ended_s: None,
        active: vec![],
        adapters: VecDeque::new(),
        last_used_s: 0.0,
        is_ready: true,
        host: HostObservation {
            phase: StartupPhase::Ready,
            bytes_downloaded: 0,
            bytes_total: 0,
            remaining_bytes_known: true,
            probe_pending: false,
            probe_attempted_s: None,
            probe: None,
        },
    });
    for policy in POLICIES {
        assert!(
            BuiltinScheduler::new(policy)
                .decide(&fleet)
                .iter()
                .any(|action| matches!(
                    action,
                    Action::Route {
                        request: 0,
                        node: 0,
                        ..
                    }
                )),
            "{policy:?} still routes to a ready host of the cooled profile"
        );
    }
}

#[test]
fn dependency_on_an_unannounced_job_waits_without_exposing_the_job() {
    let (profiles, mut trace) = performance::workload(2);
    trace.jobs[0].announced_s = 10.0;
    trace.jobs[0].arrival_s = 10.0;
    let mut early = trace.jobs[0].clone();
    early.id = 1;
    early.announced_s = 0.0;
    early.arrival_s = 0.0;
    trace.jobs.push(early);
    trace.requests[0].arrival_s = 10.0;
    trace.requests[1].job = 1;
    trace.requests[1].depends_on = vec![0];
    let run = simulate(&profiles, &trace, Policy::PlannedHybrid, 7).unwrap();
    assert_eq!(run.summary.completed, 2);
    assert!(run.requests[1].started_s.unwrap() >= run.requests[0].completed_s.unwrap());
}

#[test]
fn late_client_delivery_does_not_count_as_retry_waste_or_a_received_first_token() {
    let (mut profiles, mut trace) = performance::workload(1);
    profiles[0].network_s = 0.0;
    profiles[0].network_bytes_per_s = 4.0;
    trace.requests[0].upload_bytes = 0;
    trace.requests[0].input_tokens = 0;
    trace.horizon_s = 0.5;
    let run = simulate(&profiles, &trace, Policy::WarmOnly, 7).unwrap();
    assert_eq!(run.summary.completed, 0);
    assert_eq!(run.summary.first_token_p95_s, None);
    assert_eq!(run.summary.wasted_output_tokens, 0.0);
    assert!(run.summary.incomplete_output_tokens > 0.0);
}

#[test]
fn invalid_controller_actions_cannot_bypass_physical_admission() {
    struct Faulty {
        decided: bool,
    }
    impl Scheduler for Faulty {
        fn name(&self) -> &str {
            "faulty-test-controller"
        }
        fn decide(&mut self, _view: &FleetView<'_>) -> Vec<Action> {
            if self.decided {
                return vec![];
            }
            self.decided = true;
            vec![
                Action::Provision {
                    profile: usize::MAX,
                },
                Action::Route {
                    request: 0,
                    node: 0,
                    service_weight: f64::NAN,
                },
                Action::Route {
                    request: 0,
                    node: 0,
                    service_weight: 1.0,
                },
                Action::Route {
                    request: 0,
                    node: 0,
                    service_weight: 1.0,
                },
                Action::Release { node: 0 },
            ]
        }
    }
    let (profiles, trace) = performance::workload(1);
    let run =
        simulate_with_scheduler(&profiles, &trace, &mut Faulty { decided: false }, 7).unwrap();
    assert_eq!(run.summary.invalid_decisions, 4);
    assert_eq!(run.summary.completed, 1);
    assert_eq!(run.summary.attempts, 1);
}

fn measured(at_s: f64, rate: f64) -> HostProbe {
    let window = if rate < 4_000_000.0 { 10.0 } else { 2.0 };
    HostProbe {
        at_s,
        started_s: at_s - window,
        received_bytes: (rate * window) as u64,
        volume_bytes_per_s: rate,
        hydration_bytes: 1_000_000_000,
        image_cached: None,
        model_cached: false,
    }
}
fn loading_node(probe: Option<HostProbe>, remaining: u64) -> ObservedNode {
    ObservedNode {
        profile: 0,
        warm: false,
        allocated_s: 0.0,
        ready_s: 200.0,
        ended_s: None,
        active: vec![],
        adapters: VecDeque::new(),
        last_used_s: 0.0,
        is_ready: false,
        host: HostObservation {
            phase: StartupPhase::Hydrating,
            bytes_total: 1_000_000_000,
            bytes_downloaded: 1_000_000_000 - remaining,
            remaining_bytes_known: true,
            probe_pending: false,
            probe_attempted_s: probe.as_ref().map(|p| p.at_s),
            probe,
        },
    }
}
fn probe_fleet<'a>(
    profiles: &'a [Profile],
    scheduling: &'a Scheduling,
    node: ObservedNode,
    now: f64,
) -> FleetView<'a> {
    FleetView {
        now,
        allow_scale: false,
        profiles,
        provisioning_cooldowns: BTreeMap::new(),
        pending_allocations: vec![],
        requests: BTreeMap::new(),
        states: BTreeMap::new(),
        nodes: vec![node],
        jobs: BTreeMap::new(),
        scheduling,
        max_nodes_per_model: 2,
        idle_timeout_s: 30.0,
        reactive_delay_s: 0.0,
        build_probability: 0.0,
    }
}
#[test]
fn refreshed_high_rate_is_learned_once_per_distinct_qualified_sample() {
    let (profiles, _) = performance::workload(1);
    let scheduling = Scheduling::default();
    let mut fleet = probe_fleet(
        &profiles,
        &scheduling,
        loading_node(Some(measured(10.0, 1_000_000.0)), 0),
        10.0,
    );
    let mut scheduler = BuiltinScheduler::new(Policy::Adaptive);
    scheduler.decide(&fleet);
    assert_eq!(scheduler.diagnostics()["host_beliefs"]["0"]["samples"], 1);
    fleet.now = 20.0;
    fleet.nodes[0].host.probe = Some(measured(20.0, 40_000_000.0));
    fleet.nodes[0].host.probe_attempted_s = Some(20.0);
    scheduler.decide(&fleet);
    for _ in 0..5 {
        scheduler.decide(&fleet);
    }
    let belief = &scheduler.diagnostics()["host_beliefs"]["0"];
    assert_eq!(belief["samples"], 2);
    assert_eq!(belief["volume_bytes_per_s"], 10_750_000.0);
    // An older, out-of-order observation cannot replace the learned refresh.
    fleet.nodes[0].host.probe = Some(measured(10.0, 1_000_000.0));
    scheduler.decide(&fleet);
    assert_eq!(scheduler.diagnostics()["host_beliefs"]["0"]["samples"], 2);
}
#[test]
fn stale_or_metadata_only_slow_samples_request_refresh_without_cancelling() {
    let (profiles, _) = performance::workload(1);
    let scheduling = Scheduling::default();
    for (now, probe) in [
        (21.0, measured(10.0, 1_000_000.0)),
        (
            10.0,
            HostProbe {
                at_s: 10.0,
                started_s: 8.0,
                received_bytes: 512,
                volume_bytes_per_s: 256.0,
                hydration_bytes: 1_000_000_000,
                image_cached: None,
                model_cached: false,
            },
        ),
    ] {
        let mut node = loading_node(Some(probe), 1_000_000_000);
        node.host.probe_attempted_s = Some(0.0);
        let fleet = probe_fleet(&profiles, &scheduling, node, now);
        let mut scheduler = BuiltinScheduler::new(Policy::Adaptive);
        let actions = scheduler.decide(&fleet);
        assert!(!actions.iter().any(|a| matches!(a, Action::Release { .. })));
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::ProbeHost { node: 0 }))
        );
        assert_eq!(scheduler.diagnostics()["cancelled_slow_starts"], 0);
        assert!(
            scheduler.diagnostics()["host_beliefs"]
                .as_object()
                .unwrap()
                .is_empty()
        );
    }
}
#[test]
fn sustained_slow_payload_can_cancel_but_completed_transfer_cannot() {
    let (profiles, _) = performance::workload(1);
    let scheduling = Scheduling::default();
    for remaining in [1_000_000_000, 0] {
        let fleet = probe_fleet(
            &profiles,
            &scheduling,
            loading_node(Some(measured(10.0, 1_000_000.0)), remaining),
            10.0,
        );
        let mut scheduler = BuiltinScheduler::new(Policy::Adaptive);
        let actions = scheduler.decide(&fleet);
        assert_eq!(
            actions
                .iter()
                .any(|a| matches!(a, Action::Release { node: 0 })),
            remaining > 0
        );
        if remaining > 0 {
            assert!(
                !actions
                    .iter()
                    .any(|a| matches!(a, Action::ProbeHost { .. }))
            );
        }
    }
}
#[test]
fn unknown_remaining_progress_preserves_learning_refresh_and_hard_deadline() {
    let (profiles, _) = performance::workload(1);
    let scheduling = Scheduling::default();
    let mut node = loading_node(Some(measured(10.0, 1_000_000.0)), 1_000_000_000);
    node.host.remaining_bytes_known = false;
    node.host.probe_attempted_s = Some(0.0);
    let mut fleet = probe_fleet(&profiles, &scheduling, node, 10.0);
    let mut scheduler = BuiltinScheduler::new(Policy::Adaptive);
    let actions = scheduler.decide(&fleet);
    assert!(!actions.iter().any(|a| matches!(a, Action::Release { .. })));
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, Action::ProbeHost { node: 0 }))
    );
    assert_eq!(scheduler.diagnostics()["host_beliefs"]["0"]["samples"], 1);
    fleet.now = 181.0;
    let actions = scheduler.decide(&fleet);
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, Action::Release { node: 0 }))
    );
}
#[test]
fn hard_startup_deadline_survives_missing_or_stale_probes_and_completed_transfer() {
    let (profiles, _) = performance::workload(1);
    let scheduling = Scheduling::default();
    for probe in [None, Some(measured(10.0, 1_000_000.0))] {
        let mut node = loading_node(probe, 0);
        node.host.phase = StartupPhase::Initializing;
        let fleet = probe_fleet(&profiles, &scheduling, node, 181.0);
        let mut scheduler = BuiltinScheduler::new(Policy::Adaptive);
        let actions = scheduler.decide(&fleet);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::Release { node: 0 }))
        );
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::ProbeHost { .. }))
        );
    }
}
#[test]
fn probe_attempts_are_throttled_and_pending_refreshes_are_not_duplicated() {
    let (profiles, _) = performance::workload(1);
    let scheduling = Scheduling::default();
    let mut node = loading_node(None, 1_000_000_000);
    node.host.probe_attempted_s = Some(10.0);
    let mut fleet = probe_fleet(&profiles, &scheduling, node, 11.0);
    let mut scheduler = BuiltinScheduler::new(Policy::Adaptive);
    assert!(
        !scheduler
            .decide(&fleet)
            .iter()
            .any(|a| matches!(a, Action::ProbeHost { .. }))
    );
    fleet.now = 12.0;
    assert!(
        scheduler
            .decide(&fleet)
            .iter()
            .any(|a| matches!(a, Action::ProbeHost { node: 0 }))
    );
    fleet.nodes[0].host.probe_pending = true;
    assert!(
        !scheduler
            .decide(&fleet)
            .iter()
            .any(|a| matches!(a, Action::ProbeHost { .. }))
    );
}

fn planning_fleet<'a>(profiles: &'a [Profile], trace: &'a Scenario) -> FleetView<'a> {
    FleetView {
        now: 0.0,
        allow_scale: true,
        profiles,
        provisioning_cooldowns: BTreeMap::new(),
        pending_allocations: vec![],
        requests: trace
            .requests
            .iter()
            .map(|r| (r.id, PlannedRequest::from(r)))
            .collect(),
        states: trace
            .requests
            .iter()
            .map(|r| {
                (
                    r.id,
                    ObservedRequest {
                        state: RequestStatus::Queued,
                        completed_s: None,
                    },
                )
            })
            .collect(),
        nodes: vec![],
        jobs: trace.jobs.iter().map(|j| (j.id, j.clone())).collect(),
        scheduling: &trace.scheduling,
        max_nodes_per_model: 4,
        idle_timeout_s: 30.0,
        reactive_delay_s: 0.0,
        build_probability: 0.0,
    }
}
#[test]
fn late_capacity_is_skipped_but_an_empty_fleet_gets_one_best_effort_launch() {
    let (mut profiles, mut trace) = performance::workload(10);
    profiles[0].startup.engine_s = 100.0;
    profiles[0].startup.allocation_s = 0.0;
    profiles[0].startup.container_s = 0.0;
    profiles[0].startup.weights_s = 0.0;
    trace.jobs[0].desired_output_tps = 10000.0;
    let mut fleet = planning_fleet(&profiles, &trace);
    let mut ready = loading_node(None, 0);
    ready.is_ready = true;
    ready.ready_s = 0.0;
    ready.host.phase = StartupPhase::Ready;
    fleet.nodes.push(ready);
    let actions = BuiltinScheduler::new(Policy::Adaptive).decide(&fleet);
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::Provision { .. }))
    );
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, Action::Route { node: 0, .. }))
    );
    fleet.nodes.clear();
    let actions = BuiltinScheduler::new(Policy::Adaptive).decide(&fleet);
    assert_eq!(
        actions
            .iter()
            .filter(|a| matches!(a, Action::Provision { .. }))
            .count(),
        1
    );
}
#[test]
fn pending_allocations_reserve_capacity_without_becoming_routing_targets() {
    let (profiles, trace) = performance::workload(10);
    let mut fleet = planning_fleet(&profiles, &trace);
    fleet.max_nodes_per_model = 1;
    fleet
        .pending_allocations
        .push(switchyard_core::scheduler::PendingAllocation {
            profile: 0,
            started_s: 0.0,
            expected_ready_s: 20.0,
        });
    let actions = BuiltinScheduler::new(Policy::Adaptive).decide(&fleet);
    assert!(!actions.iter().any(|a| matches!(
        a,
        Action::Provision { .. } | Action::Route { .. } | Action::Release { .. }
    )));
}
#[test]
fn rejected_and_slow_allocations_change_placement_after_cooldown_expires() {
    use switchyard_core::scheduler::SchedulerFeedback;
    let (mut profiles, trace) = performance::workload(1);
    profiles[0].usd_per_hour = 0.5;
    let mut alternate = profiles[0].clone();
    alternate.id = "available-alternate".into();
    alternate.usd_per_hour = 0.6;
    profiles.push(alternate);
    let fleet = planning_fleet(&profiles, &trace);
    let mut scheduler = BuiltinScheduler::new(Policy::Adaptive);
    for _ in 0..8 {
        scheduler.observe(SchedulerFeedback::Allocation {
            profile: 0,
            accepted: false,
            elapsed_s: 10.0,
        });
    }
    scheduler.observe(SchedulerFeedback::Allocation {
        profile: 1,
        accepted: true,
        elapsed_s: 0.1,
    });
    let actions = scheduler.decide(&fleet);
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, Action::Provision { profile: 1 }))
    );
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::Provision { profile: 0 }))
    );
}
#[test]
fn delivered_usage_learns_bounded_output_and_workload_specific_capacity() {
    use switchyard_core::scheduler::SchedulerFeedback;
    let (profiles, mut trace) = performance::workload(1);
    trace.requests[0].max_output = 128;
    trace.requests[0].planned_output = 64;
    let mut scheduler = BuiltinScheduler::new(Policy::Adaptive);
    let mut fleet = planning_fleet(&profiles, &trace);
    let mut node = loading_node(None, 0);
    node.is_ready = true;
    node.ready_s = 0.0;
    fleet.nodes.push(node);
    fleet.allow_scale = false;
    for _ in 0..200 {
        scheduler.observe(SchedulerFeedback::Completion {
            node: 0,
            profile: 0,
            request: Box::new(fleet.requests[&0].clone()),
            output_tokens: 16,
            input_work: 0.0,
            started_s: 0.0,
            completed_s: 1.0,
        });
    }
    scheduler.decide(&fleet);
    let d = scheduler.diagnostics();
    let learned = &d["serving_learning"];
    assert_eq!(
        learned["output_buckets"][0]["samples"]
            .as_array()
            .unwrap()
            .len(),
        128
    );
    assert_eq!(learned["capacity_buckets"].as_array().unwrap().len(), 1);
    assert_eq!(learned["completed_samples"], 200);
}

#[test]
fn useful_capacity_is_allowed_after_an_unattainable_throughput_target() {
    let (mut profiles, mut trace) = performance::workload(1000);
    profiles[0].curve = vec![
        CapacityPoint {
            in_flight: 1,
            work_units_per_s: 50.0,
        },
        CapacityPoint {
            in_flight: 64,
            work_units_per_s: 100.0,
        },
    ];
    profiles[0].startup.engine_s = 45.0;
    trace.jobs[0].desired_output_tps = 10000.0;
    let mut fleet = planning_fleet(&profiles, &trace);
    let mut ready = loading_node(None, 0);
    ready.is_ready = true;
    ready.ready_s = 0.0;
    ready.host.phase = StartupPhase::Ready;
    fleet.nodes.push(ready);
    let actions = BuiltinScheduler::new(Policy::Adaptive).decide(&fleet);
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, Action::Provision { .. })),
        "late relative to desired TPS can still accelerate draining"
    );
}

#[test]
fn asynchronous_allocation_rejection_preserves_hot_service_and_hides_outcome_until_completion() {
    use switchyard_core::scheduler::SchedulerFeedback;
    use switchyard_sim::observations::{CapacityLimit, Observations, ObservedEnvironment};
    struct Expansion {
        attempted: bool,
        inner: BuiltinScheduler,
        rejected_at: Option<f64>,
        rejected: bool,
        now: f64,
        pending_seen: bool,
    }
    impl Scheduler for Expansion {
        fn name(&self) -> &str {
            "async-rejection"
        }
        fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<Action> {
            self.now = fleet.now;
            if fleet.now < 10.0 {
                assert!(!self.rejected);
            }
            if self.rejected && self.rejected_at.is_none() {
                self.rejected_at = Some(fleet.now);
            }
            self.pending_seen |= !fleet.pending_allocations.is_empty();
            let mut actions = self.inner.decide(fleet);
            if !self.attempted {
                self.attempted = true;
                actions.insert(0, Action::Provision { profile: 0 });
            }
            actions
        }
        fn observe(&mut self, feedback: SchedulerFeedback) {
            if let SchedulerFeedback::Allocation {
                accepted: false,
                elapsed_s,
                ..
            } = feedback
            {
                assert_eq!(elapsed_s, 10.0);
                self.rejected = true;
            }
        }
    }
    let (profiles, trace) = performance::workload(10);
    let expected = simulate(&profiles, &trace, Policy::WarmOnly, 7).unwrap();
    let observations = Observations {
        profiles: BTreeMap::from([(
            profiles[0].id.clone(),
            ObservedEnvironment {
                capacity: vec![CapacityLimit {
                    at_s: 0.0,
                    instances: 1,
                }],
                rejection_delay_s: 10.0,
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    let mut controller = Expansion {
        attempted: false,
        inner: BuiltinScheduler::new(Policy::WarmOnly),
        rejected_at: None,
        rejected: false,
        now: 0.0,
        pending_seen: false,
    };
    let run = switchyard_sim::engine::simulate_with_observations(
        &profiles,
        &trace,
        &observations,
        &mut controller,
        7,
    )
    .unwrap();
    assert_eq!(run.summary.completed, 10);
    assert_eq!(run.summary.invalid_decisions, 0);
    assert_eq!(run.summary.capacity_rejections, 1);
    assert_eq!(run.nodes.len(), 1);
    assert!(controller.pending_seen);
    assert!(controller.rejected_at.unwrap() >= 9.9);
    assert_eq!(
        run.requests
            .iter()
            .map(|r| r.completed_s)
            .collect::<Vec<_>>(),
        expected
            .requests
            .iter()
            .map(|r| r.completed_s)
            .collect::<Vec<_>>()
    );
}
