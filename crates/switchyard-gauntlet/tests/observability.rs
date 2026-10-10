use serde_json::Value;
use switchyard_gauntlet::engine::{CloudSimulator, InferenceSpec};
use switchyard_gauntlet::scheduler::{BuiltinScheduler, RequestStatus};
use switchyard_gauntlet::*;

fn hardware() -> Profile {
    Profile {
        id: "analytical-gpu".into(),
        model: "example-model".into(),
        workload: Workload::Text,
        gpu: "synthetic-accelerator".into(),
        usd_per_hour: 1.0,
        startup: Startup {
            allocation_s: 0.0,
            container_s: 0.0,
            weights_s: 0.0,
            engine_s: 10.0,
            jitter: 0.0,
            restore_samples_s: vec![],
            build_samples_s: vec![],
        },
        network_s: 0.0,
        network_bytes_per_s: 100.0,
        max_in_flight: 2,
        kv_tokens: 10_000,
        input_work_per_token: 0.5,
        image_work_per_megapixel: 0.0,
        adapter_load_s: 0.0,
        multi_adapter_factor: 1.0,
        adapter_cache_slots: 2,
        minimum_bill_s: 1.0,
        curve: vec![CapacityPoint {
            in_flight: 1,
            work_units_per_s: 100.0,
        }],
        evidence: "analytical observability fixture".into(),
        source: "tests/observability.rs".into(),
        host: HostProfile::default(),
    }
}

fn request(id: usize) -> Request {
    Request {
        id,
        job: 0,
        model: "example-model".into(),
        workload: Workload::Text,
        class: "standard".into(),
        arrival_s: 0.0,
        input_tokens: 100,
        planned_output: 100,
        max_output: 400,
        actual_output: 100,
        image_megapixels: 0.0,
        upload_bytes: 0,
        cached_fraction: 0.0,
        adapter: None,
        latency_target_s: 20.0,
        depends_on: vec![],
    }
}

fn scenario(requests: Vec<Request>) -> Scenario {
    Scenario {
        name: "observability-fixture".into(),
        notes: "Analytical known-input and hidden-state boundary checks".into(),
        horizon_s: 20.0,
        tick_s: 0.25,
        idle_timeout_s: 30.0,
        max_nodes_per_model: 2,
        launch_fail_probability: 0.0,
        reactive_delay_s: 0.0,
        capacity_scale: 1.0,
        startup_scale: 1.0,
        build_probability: 0.0,
        scheduling: Scheduling::default(),
        host_reality: None,
        warm: vec![],
        jobs: vec![Job {
            id: 0,
            name: "announced-job".into(),
            announced_s: 0.0,
            arrival_s: 0.0,
            completion_budget_s: 20.0,
            desired_output_tps: 0.0,
        }],
        requests,
        faults: vec![],
    }
}

fn inference() -> InferenceSpec {
    InferenceSpec {
        model: "example-model".into(),
        input_tokens: 0,
        planned_output: 10,
        max_output: 10,
        upload_bytes: 0,
        workload: Workload::Text,
        image_megapixels: 0.0,
        adapter: None,
        class: "standard".into(),
        cached_fraction: 0.0,
    }
}

#[derive(Default)]
struct AnnouncementObserver {
    hidden_snapshots: usize,
    announced_snapshots: usize,
}

impl Scheduler for AnnouncementObserver {
    fn name(&self) -> &str {
        "announcement-observer"
    }

    fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<Action> {
        let known_requests: Vec<_> = fleet.requests.keys().copied().collect();
        let known_states: Vec<_> = fleet.states.keys().copied().collect();
        assert_eq!(known_requests, known_states);
        assert!(fleet.jobs.contains_key(&0));
        if fleet.now < 2.0 {
            assert_eq!(known_requests, vec![0]);
            assert!(!fleet.jobs.contains_key(&1));
            self.hidden_snapshots += 1;
        } else {
            assert_eq!(known_requests, vec![0, 1]);
            assert!(fleet.jobs.contains_key(&1));
            if fleet.now < 3.0 {
                assert_eq!(fleet.states[&1].state, RequestStatus::Future);
            }
            self.announced_snapshots += 1;
        }
        vec![]
    }
}

#[test]
fn unannounced_jobs_requests_and_states_are_absent_until_announcement() {
    let mut pending = request(1);
    pending.job = 1;
    pending.arrival_s = 3.0;
    let mut workload = scenario(vec![request(0), pending]);
    workload.horizon_s = 4.0;
    workload.jobs.push(Job {
        id: 1,
        name: "previously-unannounced-job".into(),
        announced_s: 2.0,
        arrival_s: 3.0,
        completion_budget_s: 20.0,
        desired_output_tps: 0.0,
    });
    let mut observer = AnnouncementObserver::default();
    simulate_with_scheduler(&[hardware()], &workload, &mut observer, 7).unwrap();
    assert!(observer.hidden_snapshots > 0);
    assert!(observer.announced_snapshots > 0);
}

struct DecisionRecorder {
    controller: BuiltinScheduler,
    initial_decisions: Vec<String>,
}

impl Scheduler for DecisionRecorder {
    fn name(&self) -> &str {
        "recorded-planned-controller"
    }

    fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<Action> {
        let actions = self.controller.decide(fleet);
        if fleet.now == 0.0 {
            self.initial_decisions.push(format!("{actions:?}"));
        }
        actions
    }
}

#[test]
fn actual_output_does_not_change_initial_controller_decisions() {
    let mut short = scenario(vec![request(0), request(1)]);
    short.warm.push(WarmNode {
        profile: "analytical-gpu".into(),
        copies: 1,
    });
    for row in &mut short.requests {
        row.actual_output = 10;
    }
    let mut long = short.clone();
    for row in &mut long.requests {
        row.actual_output = 300;
    }
    let replay = |workload: &Scenario| {
        let mut controller = DecisionRecorder {
            controller: BuiltinScheduler::new(Policy::PlannedHybrid),
            initial_decisions: vec![],
        };
        let run = simulate_with_scheduler(&[hardware()], workload, &mut controller, 7).unwrap();
        (controller.initial_decisions, run)
    };
    let (short_actions, short_run) = replay(&short);
    let (long_actions, long_run) = replay(&long);
    assert!(!short_actions.is_empty());
    assert!(
        short_actions
            .iter()
            .any(|actions| actions.contains("Route"))
    );
    assert_eq!(short_actions, long_actions);
    assert_eq!(short_run.summary.completed, 2);
    assert_eq!(long_run.summary.completed, 2);
    assert_eq!(short_run.summary.output_tokens, 20);
    assert_eq!(long_run.summary.output_tokens, 600);
}

#[derive(Default)]
struct StartupObserver {
    provisioned: bool,
    before_ready: Vec<(f64, f64, f64)>,
}

impl Scheduler for StartupObserver {
    fn name(&self) -> &str {
        "startup-observer"
    }

    fn decide(&mut self, fleet: &FleetView<'_>) -> Vec<Action> {
        if let Some(node) = fleet.nodes.first() {
            assert!(!node.is_ready);
            self.before_ready
                .push((fleet.now, node.ready_s, node.last_used_s));
        }
        if !self.provisioned && fleet.allow_scale {
            self.provisioned = true;
            vec![Action::Provision { profile: 0 }]
        } else {
            vec![]
        }
    }
}

#[test]
fn pending_readiness_estimate_and_last_use_do_not_reveal_hidden_startup_scale() {
    let replay = |scale| {
        let mut workload = scenario(vec![request(0)]);
        workload.horizon_s = 1.0;
        workload.startup_scale = scale;
        let mut observer = StartupObserver::default();
        let run = simulate_with_scheduler(&[hardware()], &workload, &mut observer, 7).unwrap();
        assert_eq!(run.nodes.len(), 1);
        (observer.before_ready, run.nodes[0].ready_s)
    };
    let (normal_observations, normal_actual_ready) = replay(1.0);
    let (slow_observations, slow_actual_ready) = replay(100.0);
    assert!(!normal_observations.is_empty());
    assert_eq!(normal_observations, slow_observations);
    assert!(
        normal_observations
            .iter()
            .all(|(_, _, last_used)| *last_used == 0.0)
    );
    assert_eq!(normal_actual_ready, 10.0);
    assert_eq!(slow_actual_ready, 1000.0);
}

#[test]
fn noisy_probes_do_not_change_future_host_samples_or_readiness() {
    let replay = |noise, probe| {
        let mut profile = hardware();
        profile.startup.engine_s = 1.0;
        profile.startup.jitter = 0.5;
        profile.host = HostProfile {
            volume_bandwidth_bytes_per_s: vec![1000.0, 4000.0],
            container_image_bytes: 500,
            model_bytes: 1000,
            image_cache_probability: 0.25,
            model_cache_probability: 0.5,
            probe_duration_s: 0.1,
            measurement_noise: noise,
        };
        let mut cloud = CloudSimulator::new(vec![profile], 19).unwrap();
        let first = cloud.provision("analytical-gpu").unwrap();
        cloud.advance(0.5).unwrap();
        for _ in 0..5 {
            if probe {
                cloud.probe(first).unwrap();
            }
            cloud.advance(0.11).unwrap();
        }
        let measured =
            cloud.snapshot()["instances"][0]["host"]["probe"]["volume_bytes_per_s"].clone();
        cloud.provision("analytical-gpu").unwrap();
        cloud.provision("analytical-gpu").unwrap();
        cloud.advance(10.0).unwrap();
        let snapshot = cloud.snapshot();
        let readiness: Vec<_> = snapshot["instances"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| {
                assert_eq!(node["status"], "ready");
                (
                    node["ready_at_s"].clone(),
                    node["host"]["bytes_total"].clone(),
                )
            })
            .collect();
        (readiness, measured)
    };
    let (unprobed, no_measurement) = replay(0.0, false);
    let (quiet_probes, quiet_measurement) = replay(0.0, true);
    let (noisy_probes, noisy_measurement) = replay(0.8, true);
    assert_eq!(no_measurement, Value::Null);
    assert!(quiet_measurement.is_number());
    assert!(noisy_measurement.is_number());
    assert_ne!(quiet_measurement, noisy_measurement);
    assert_eq!(unprobed, quiet_probes);
    assert_eq!(quiet_probes, noisy_probes);
}

#[test]
fn client_tokens_and_usage_wait_for_egress_and_delivery_keeps_instance_busy() {
    let mut profile = hardware();
    profile.startup.engine_s = 0.0;
    profile.network_s = 0.25;
    // One output token per second on the wire, versus 100 on the GPU.
    profile.network_bytes_per_s = 4.0;
    let mut cloud = CloudSimulator::new(vec![profile], 7).unwrap();
    let node = cloud.provision("analytical-gpu").unwrap();
    cloud.advance(0.0).unwrap();
    let id = cloud.submit(node, inference()).unwrap();
    cloud.advance(1.49).unwrap();
    let before = cloud.request(id).unwrap();
    assert_eq!(before["first_token_at_s"], Value::Null);
    assert_eq!(before["completed_at_s"], Value::Null);
    assert_eq!(before["generated_tokens"], 0);
    assert_eq!(before["usage"], Value::Null);
    assert!(cloud.release(node).unwrap_err().contains("undelivered"));
    cloud.advance(0.02).unwrap();
    let first = cloud.request(id).unwrap();
    assert!((first["first_token_at_s"].as_f64().unwrap() - 1.5).abs() < 1e-8);
    assert_eq!(first["generated_tokens"], 1);
    assert_eq!(first["usage"], Value::Null);
    assert_eq!(first["status"], "active");
    cloud.advance(20.0).unwrap();
    let complete = cloud.request(id).unwrap();
    let output = complete["usage"]["completion_tokens"].as_u64().unwrap();
    assert!((5..=10).contains(&output));
    assert_eq!(complete["status"], "completed");
    assert_eq!(complete["generated_tokens"], output);
    assert!((complete["completed_at_s"].as_f64().unwrap() - (0.5 + output as f64)).abs() < 1e-8);
    cloud.release(node).unwrap();
}

#[test]
fn input_length_cache_fraction_and_upload_bytes_change_client_ttft() {
    let ttft = |input_tokens, cached_fraction, upload_bytes| {
        let mut profile = hardware();
        profile.network_s = 0.1;
        let mut row = request(0);
        row.input_tokens = input_tokens;
        row.cached_fraction = cached_fraction;
        row.upload_bytes = upload_bytes;
        let mut workload = scenario(vec![row]);
        workload.warm.push(WarmNode {
            profile: "analytical-gpu".into(),
            copies: 1,
        });
        let run = simulate(&[profile], &workload, Policy::WarmOnly, 7).unwrap();
        assert_eq!(run.summary.completed, 1);
        run.requests[0].first_token_s.unwrap()
    };
    // Two 0.1s network hops, 0.5s prefill, and 0.04s token egress.
    let base = ttft(100, 0.0, 0);
    assert!((base - 0.74).abs() < 1e-8);
    assert!((ttft(200, 0.0, 0) - base - 0.5).abs() < 1e-8);
    assert!((base - ttft(100, 0.5, 0) - 0.25).abs() < 1e-8);
    assert!((ttft(100, 0.0, 200) - base - 2.0).abs() < 1e-8);
}

#[test]
fn zero_startup_does_not_bypass_readiness_event_or_launch_failure() {
    let mut profile = hardware();
    profile.startup.engine_s = 0.0;
    let config = CloudConfig {
        launch_fail_probability: 1.0,
        ..CloudConfig::default()
    };
    let mut cloud = CloudSimulator::with_config(vec![profile], config, 7).unwrap();
    let node = cloud.provision("analytical-gpu").unwrap();
    let pending = cloud.snapshot();
    assert_eq!(pending["instances"][node]["status"], "starting");
    assert_eq!(pending["instances"][node]["ready_at_s"], Value::Null);
    assert!(cloud.submit(node, inference()).is_err());
    let request = cloud.enqueue(node, inference()).unwrap();
    assert_eq!(cloud.request(request).unwrap()["status"], "queued");
    cloud.advance(0.0).unwrap();
    let after_event = cloud.snapshot();
    assert_eq!(after_event["instances"][node]["status"], "failed");
    assert_eq!(after_event["instances"][node]["ready_at_s"], Value::Null);
    let failed = cloud.request(request).unwrap();
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["first_token_at_s"], Value::Null);
    assert_eq!(failed["usage"], Value::Null);
    assert_eq!(failed["generated_tokens"], 0);
}

#[test]
fn configured_instance_queue_limit_is_atomic_and_release_cancels_waiters() {
    let config = CloudConfig {
        instance_queue_limit: 2,
        max_instances_per_model: 1,
        ..CloudConfig::default()
    };
    let mut cloud = CloudSimulator::with_config(vec![hardware()], config, 7).unwrap();
    let node = cloud.provision("analytical-gpu").unwrap();
    assert!(cloud.provision("analytical-gpu").is_err());
    let first = cloud.enqueue(node, inference()).unwrap();
    let second = cloud.enqueue(node, inference()).unwrap();
    assert_eq!(first, 0);
    assert_eq!(second, 1);
    assert!(
        cloud
            .enqueue(node, inference())
            .unwrap_err()
            .contains("queue limit")
    );
    assert_eq!(cloud.snapshot()["requests"].as_array().unwrap().len(), 2);
    cloud.release(node).unwrap();
    cloud.advance(20.0).unwrap();
    assert_eq!(cloud.snapshot()["instances"][node]["status"], "released");
    for id in [first, second] {
        let cancelled = cloud.request(id).unwrap();
        assert_eq!(cancelled["status"], "failed");
        assert_eq!(cancelled["first_token_at_s"], Value::Null);
        assert_eq!(cancelled["usage"], Value::Null);
    }
    assert_eq!(cloud.provision("analytical-gpu").unwrap(), 1);
}

#[test]
fn configured_router_queue_limit_rejects_without_creating_hidden_requests() {
    let config = CloudConfig {
        router_queue_limit: 2,
        router_policy: Policy::WarmOnly,
        ..CloudConfig::default()
    };
    let mut cloud = CloudSimulator::with_config(vec![hardware()], config, 7).unwrap();
    assert_eq!(cloud.route(inference()).unwrap(), 0);
    assert_eq!(cloud.route(inference()).unwrap(), 1);
    assert!(
        cloud
            .route(inference())
            .unwrap_err()
            .contains("router queue limit")
    );
    cloud.advance(1.0).unwrap();
    let snapshot = cloud.snapshot();
    assert!(snapshot["instances"].as_array().unwrap().is_empty());
    assert_eq!(snapshot["requests"].as_array().unwrap().len(), 2);
    for id in 0..2 {
        assert_eq!(cloud.request(id).unwrap()["status"], "queued");
    }
}
