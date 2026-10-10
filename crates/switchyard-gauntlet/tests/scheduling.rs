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
            probe_pending: false,
            probe: Some(HostProbe {
                at_s: 0.0,
                volume_bytes_per_s: 1_000_000.0,
                hydration_bytes: 1_000_000_000,
                image_cached: false,
                model_cached: false,
            }),
        },
    });
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
