use switchyard_core::topology::{EndpointPlan, FleetTopology, HostPlan};
use switchyard_gauntlet::*;

fn fixture() -> (Vec<Profile>, Scenario) {
    let profiles: Vec<Profile> = serde_json::from_value(serde_json::json!([
        {
            "id":"a-shared", "model":"a", "workload":"text", "gpu":"virtual-24gb",
            "usd_per_hour":3.6,"minimum_bill_s":1,
            "startup":{"allocation_s":0.2,"container_s":0,"weights_s":0,"engine_s":0,"jitter":0},
            "network_s":0,"network_bytes_per_s":1000000000,
            "max_in_flight":2,"kv_tokens":4096,"input_work_per_token":0,"image_work_per_megapixel":0,
            "adapter_load_s":0,"multi_adapter_factor":1,"adapter_cache_slots":0,
            "curve":[{"in_flight":1,"work_units_per_s":100},{"in_flight":2,"work_units_per_s":100}],
            "evidence":"analytical","source":"shared_hosts test"
        }
    ])).unwrap();
    let mut b = profiles[0].clone();
    b.id = "b-shared".into();
    b.model = "b".into();
    let mut profiles = profiles;
    profiles.push(b);
    let topology = FleetTopology {
        host_plans: vec![HostPlan {
            id: "warm-pair".into(),
            gpu: "virtual-24gb".into(),
            gpu_memory_bytes: 24 << 30,
            reserved_gpu_memory_bytes: 2 << 30,
            contention_factor: 1.0,
            endpoints: vec![
                EndpointPlan {
                    id: "a-0".into(),
                    profile: "a-shared".into(),
                    gpu_memory_bytes: 8 << 30,
                    compute_fraction: 0.5,
                },
                EndpointPlan {
                    id: "b-0".into(),
                    profile: "b-shared".into(),
                    gpu_memory_bytes: 8 << 30,
                    compute_fraction: 0.5,
                },
            ],
        }],
    };
    let mut scenario: Scenario = serde_json::from_value(serde_json::json!({
        "name":"shared-gpu", "notes":"Analytical reserved-compute fixture", "horizon_s":10,"tick_s":0.01,
        "idle_timeout_s":1,"max_nodes_per_model":2,"launch_fail_probability":0,"reactive_delay_s":0,
        "capacity_scale":1,"startup_scale":1,"warm":[{"profile":"a-shared","copies":1}],
        "jobs":[{"id":0,"name":"pair","announced_s":0,"arrival_s":0,"completion_budget_s":10,"desired_output_tps":0}],
        "requests":[],"faults":[]
    })).unwrap();
    scenario.topology = topology;
    for (id, model) in ["a", "b"].into_iter().enumerate() {
        scenario.requests.push(Request {
            id,
            job: 0,
            model: model.into(),
            workload: Workload::Text,
            class: "standard".into(),
            arrival_s: 0.0,
            input_tokens: 1,
            planned_output: 100,
            max_output: 100,
            actual_output: 100,
            image_megapixels: 0.0,
            upload_bytes: 0,
            cached_fraction: 0.0,
            adapter: None,
            latency_target_s: 10.0,
            depends_on: vec![],
        });
    }
    (profiles, scenario)
}

#[test]
fn shared_gpu_has_two_endpoints_one_rent_and_bounded_service() {
    let (profiles, scenario) = fixture();
    let run = simulate(&profiles, &scenario, Policy::WarmOnly, 7).unwrap();
    assert_eq!(run.summary.completed, 2);
    assert_eq!(run.summary.invalid_decisions, 0);
    assert_eq!(run.hosts.len(), 1);
    assert_eq!(run.hosts[0].endpoints, vec![0, 1]);
    assert!((run.summary.cost_usd - 0.01).abs() < 1e-9);
    for request in &run.requests {
        assert!((request.completed_s.unwrap() - 2.0).abs() < 1e-8);
    }
}

#[test]
fn cold_shared_launch_forecasts_all_models_and_allocates_once() {
    let (profiles, mut scenario) = fixture();
    scenario.warm.clear();
    let run = simulate(&profiles, &scenario, Policy::Adaptive, 7).unwrap();
    assert_eq!(run.summary.completed, 2);
    assert_eq!(run.summary.launched, 1);
    assert_eq!(run.summary.invalid_decisions, 0);
    assert_eq!(run.hosts.len(), 1);
}

#[test]
fn replicas_of_one_model_use_separate_memory_and_queues() {
    let (profiles, mut scenario) = fixture();
    scenario.topology.host_plans[0].endpoints[1].profile = "a-shared".into();
    scenario.requests[1].model = "a".into();
    let run = simulate(&profiles, &scenario, Policy::WarmOnly, 7).unwrap();
    assert_eq!(run.summary.completed, 2);
    assert_eq!(run.nodes.len(), 2);
    assert!(run.nodes.iter().all(|n| n.profile == "a-shared"));
    assert_eq!(run.hosts.len(), 1);
    scenario.max_nodes_per_model = 1;
    assert!(simulate(&profiles, &scenario, Policy::WarmOnly, 7).is_err());
}

#[test]
fn releasing_idle_endpoint_cannot_terminate_busy_sibling() {
    let (profiles, mut scenario) = fixture();
    scenario.warm.clear();
    scenario.requests[0].actual_output = 1;
    scenario.requests[1].actual_output = 400;
    scenario.requests[1].max_output = 400;
    scenario.requests[1].planned_output = 400;
    scenario.idle_timeout_s = 0.05;
    let run = simulate(&profiles, &scenario, Policy::Adaptive, 7).unwrap();
    assert_eq!(run.summary.completed, 2);
    assert_eq!(run.summary.launched, 1);
    assert_eq!(run.summary.invalid_decisions, 0);
    assert!(run.hosts[0].ended_s >= run.requests[1].completed_s.unwrap());
}

#[test]
fn host_fault_ends_every_resident_engine() {
    let (profiles, mut scenario) = fixture();
    scenario.faults.push(Fault { at_s: 0.5, node: 1 });
    let run = simulate(&profiles, &scenario, Policy::WarmOnly, 7).unwrap();
    assert!(run.nodes.iter().all(|n| n.failed && n.ended_s == 0.5));
}

#[test]
fn overcommit_incompatible_hardware_and_duplicate_bindings_fail_before_launch() {
    let (profiles, mut scenario) = fixture();
    let pristine = scenario.clone();
    scenario.topology.host_plans[0].endpoints[1].gpu_memory_bytes = 24 << 30;
    assert!(validate(&profiles, &scenario).is_err());
    scenario = pristine.clone();
    scenario.topology.host_plans[0].endpoints[1].compute_fraction = 0.6;
    assert!(validate(&profiles, &scenario).is_err());
    scenario = pristine.clone();
    scenario.topology.host_plans[0].gpu = "other".into();
    assert!(validate(&profiles, &scenario).is_err());
    scenario = pristine.clone();
    let mut duplicate = scenario.topology.host_plans[0].clone();
    duplicate.id = "duplicate".into();
    scenario.topology.host_plans.push(duplicate);
    assert!(validate(&profiles, &scenario).is_err());
}

#[test]
fn provider_provision_returns_the_requested_model_endpoint_and_shared_ownership() {
    let (profiles, scenario) = fixture();
    let mut cloud = engine::CloudSimulator::with_config(
        profiles,
        CloudConfig {
            topology: scenario.topology,
            ..Default::default()
        },
        7,
    )
    .unwrap();
    let endpoint = cloud.provision("b-shared").unwrap();
    assert_eq!(endpoint, 1);
    let snapshot = cloud.snapshot();
    assert_eq!(snapshot["hosts"].as_array().unwrap().len(), 1);
    assert_eq!(snapshot["instances"].as_array().unwrap().len(), 2);
    assert_eq!(snapshot["instances"][endpoint]["profile_id"], "b-shared");
    assert_eq!(snapshot["instances"][endpoint]["physical_host"], 0);
    cloud.advance(1.0).unwrap();
    cloud.release(endpoint).unwrap();
    let snapshot = cloud.snapshot();
    assert!(
        snapshot["instances"]
            .as_array()
            .unwrap()
            .iter()
            .all(|node| node["status"] == "released")
    );
}
