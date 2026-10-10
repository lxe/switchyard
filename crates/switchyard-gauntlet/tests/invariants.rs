use switchyard_gauntlet::*;

fn hardware() -> Profile {
    Profile {
        id: "generic-gpu".into(),
        model: "custom-model".into(),
        workload: Workload::Text,
        gpu: "test-accelerator".into(),
        usd_per_hour: 3.6,
        startup: Startup {
            allocation_s: 0.0,
            container_s: 0.0,
            weights_s: 0.0,
            engine_s: 5.0,
            jitter: 0.0,
            restore_samples_s: vec![],
            build_samples_s: vec![],
        },
        network_s: 0.0,
        network_bytes_per_s: 100_000_000.0,
        max_in_flight: 2,
        kv_tokens: 1000,
        input_work_per_token: 0.0,
        image_work_per_megapixel: 0.0,
        adapter_load_s: 0.0,
        multi_adapter_factor: 1.0,
        adapter_cache_slots: 2,
        minimum_bill_s: 1.0,
        curve: vec![
            CapacityPoint {
                in_flight: 1,
                work_units_per_s: 100.0,
            },
            CapacityPoint {
                in_flight: 2,
                work_units_per_s: 100.0,
            },
        ],
        evidence: "exact analytical test fixture".into(),
        source: "tests/invariants.rs".into(),
        host: HostProfile::default(),
    }
}
fn request(id: usize) -> Request {
    Request {
        id,
        job: 0,
        model: "custom-model".into(),
        workload: Workload::Text,
        class: "batch".into(),
        arrival_s: 0.0,
        input_tokens: 10,
        planned_output: 100,
        max_output: 100,
        actual_output: 100,
        image_megapixels: 0.0,
        upload_bytes: 0,
        cached_fraction: 0.0,
        adapter: None,
        latency_target_s: 10.0,
        depends_on: vec![],
    }
}
fn workload(n: usize) -> Scenario {
    Scenario {
        name: "generic-fixture".into(),
        notes: "Known analytical100TPS shared server".into(),
        horizon_s: 20.0,
        tick_s: 0.1,
        idle_timeout_s: 2.0,
        max_nodes_per_model: 2,
        launch_fail_probability: 0.0,
        reactive_delay_s: 0.0,
        capacity_scale: 1.0,
        startup_scale: 1.0,
        build_probability: 0.0,
        scheduling: Scheduling::default(),
        host_reality: None,
        warm: vec![WarmNode {
            profile: "generic-gpu".into(),
            copies: 1,
        }],
        jobs: vec![Job {
            id: 0,
            name: "request-burst".into(),
            announced_s: 0.0,
            arrival_s: 0.0,
            completion_budget_s: 20.0,
            desired_output_tps: 0.0,
        }],
        requests: (0..n).map(request).collect(),
        faults: vec![],
    }
}
fn run(profiles: &[Profile], scenario: &Scenario, policy: Policy) -> engine::Run {
    simulate(profiles, scenario, policy, 7).unwrap()
}

#[test]
fn conserves_tokens_and_shared_gpu_capacity() {
    let result = run(&[hardware()], &workload(4), Policy::WarmOnly);
    assert_eq!(result.summary.completed, 4);
    assert_eq!(result.summary.output_tokens, 400);
    assert!((result.summary.jobs[0].elapsed_s.unwrap() - 4.0).abs() < 0.11);
    assert!((result.nodes[0].emitted_tokens - 400.0).abs() < 1e-6);
    assert_eq!(result.summary.max_active_per_node, 2);
}
#[test]
fn identical_seed_is_a_byte_identical_replay() {
    let profiles = scenarios::default_profiles();
    let scenario = &scenarios::suite(&profiles)[2];
    let a = run(&profiles, scenario, Policy::PlannedHybrid);
    let b = run(&profiles, scenario, Policy::PlannedHybrid);
    assert_eq!(
        serde_json::to_vec(&a).unwrap(),
        serde_json::to_vec(&b).unwrap()
    );
}
#[test]
fn startup_and_minimum_billing_are_charged() {
    let mut p = hardware();
    p.minimum_bill_s = 10.0;
    let mut s = workload(1);
    s.warm.clear();
    s.horizon_s = 9.0;
    let result = run(&[p], &s, Policy::PlannedHybrid);
    assert_eq!(result.summary.completed, 1);
    assert!(result.requests[0].started_s.unwrap() >= 5.0);
    assert!((result.summary.cost_usd - 0.01).abs() < 1e-8);
}
#[test]
fn pending_capacity_does_not_create_duplicate_instances() {
    let mut s = workload(4);
    s.warm.clear();
    s.max_nodes_per_model = 10;
    let result = run(&[hardware()], &s, Policy::PlannedHybrid);
    assert_eq!(result.summary.launched, 1);
    assert_eq!(result.summary.completed, 4);
}
#[test]
fn wrong_model_and_oversized_rows_are_rejected() {
    let mut s = workload(3);
    s.requests[0].model = "different-weights".into();
    s.requests[1].input_tokens = 2000;
    let result = run(&[hardware()], &s, Policy::WarmOnly);
    assert_eq!(result.summary.rejected, 2);
    assert_eq!(result.summary.completed, 1);
}
#[test]
fn memory_admission_uses_hard_output_limit_not_estimate() {
    let mut p = hardware();
    p.kv_tokens = 150;
    let mut s = workload(2);
    for r in &mut s.requests {
        r.planned_output = 10;
    }
    let result = run(&[p], &s, Policy::WarmOnly);
    assert_eq!(result.summary.max_active_per_node, 1);
    assert!(result.summary.max_reserved_kv_tokens <= 150);
}
#[test]
fn dependent_request_waits_for_delivered_response() {
    let mut p = hardware();
    p.network_s = 0.2;
    let mut s = workload(2);
    s.requests[1].class = "standard".into();
    s.requests[1].depends_on = vec![0];
    let result = run(&[p], &s, Policy::WarmOnly);
    assert!(result.requests[1].started_s.unwrap() >= result.requests[0].completed_s.unwrap());
}
#[test]
fn cycles_invalid_scales_and_nonfinite_curves_fail_validation() {
    let mut s = workload(2);
    s.requests[0].depends_on = vec![1];
    assert!(validate(&[hardware()], &s).is_err());
    s.requests[0].depends_on.clear();
    s.capacity_scale = f64::NAN;
    assert!(validate(&[hardware()], &s).is_err());
    s.capacity_scale = 1.0;
    let mut p = hardware();
    p.curve[0].work_units_per_s = f64::INFINITY;
    assert!(validate(&[p], &s).is_err());
}
#[test]
fn failures_retry_once_per_attempt_without_double_completion() {
    let mut s = workload(4);
    s.faults = vec![Fault { at_s: 0.5, node: 0 }];
    let result = run(&[hardware()], &s, Policy::PlannedHybrid);
    assert_eq!(result.summary.completed, 4);
    assert_eq!(result.summary.output_tokens, 400);
    assert!(result.summary.attempts > 4);
    assert!(result.summary.wasted_output_tokens > 0.0);
    assert!(result.nodes.iter().any(|n| n.failed));
}
#[test]
fn multi_adapter_penalty_changes_throughput() {
    let mut p = hardware();
    let mut s = workload(2);
    for r in &mut s.requests {
        r.adapter = Some(format!("adapter-{}", r.id));
    }
    let a = run(&[p.clone()], &s, Policy::WarmOnly);
    p.multi_adapter_factor = 0.5;
    let b = run(&[p], &s, Policy::WarmOnly);
    assert!(b.summary.jobs[0].elapsed_s.unwrap() > a.summary.jobs[0].elapsed_s.unwrap() * 1.9);
}
#[test]
fn ready_batch_does_not_get_capacity_from_adapter_waiters() {
    let mut p = hardware();
    p.curve[1].work_units_per_s = 200.0;
    p.adapter_load_s = 10.0;
    let mut s = workload(2);
    s.requests[1].adapter = Some("not-loaded".into());
    let result = run(&[p], &s, Policy::WarmOnly);
    assert!((result.requests[0].completed_s.unwrap() - 1.0).abs() < 0.11);
}
#[test]
fn completion_precision_does_not_depend_on_large_quantum() {
    let mut s = workload(1);
    s.requests[0].actual_output = 25;
    s.requests[0].planned_output = 25;
    s.tick_s = 0.5;
    let result = run(&[hardware()], &s, Policy::WarmOnly);
    assert!((result.requests[0].completed_s.unwrap() - 0.25).abs() < 1e-8);
}
#[test]
fn unfinished_requests_count_as_slo_misses() {
    let mut s = workload(1);
    s.warm.clear();
    s.requests[0].class = "latency".into();
    let result = run(&[hardware()], &s, Policy::WarmOnly);
    assert_eq!(
        result
            .summary
            .classes
            .iter()
            .find(|c| c.name == "latency")
            .map(|c| c.slo_fraction),
        Some(0.0)
    );
    assert_eq!(result.summary.unfinished, 1);
}
#[test]
fn zero_cold_capacity_never_provisions_in_warm_only() {
    let mut s = workload(2);
    s.warm.clear();
    let result = run(&[hardware()], &s, Policy::WarmOnly);
    assert_eq!(result.summary.launched, 0);
    assert_eq!(result.summary.unfinished, 2);
}
#[test]
fn schema_round_trip_and_all_example_profiles_validate() {
    let profiles = scenarios::default_profiles();
    for s in scenarios::suite(&profiles) {
        validate(&profiles, &s).unwrap();
        let json = serde_json::to_vec(&s).unwrap();
        let parsed: Scenario = serde_json::from_slice(&json).unwrap();
        assert_eq!(s.name, parsed.name);
    }
}
