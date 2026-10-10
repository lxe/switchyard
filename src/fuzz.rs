//! Seeded generative workload corpus. Every case can be exported and replayed.
use crate::*;

struct Random(u64);
impl Random {
    fn next(&mut self, max: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 16) % max
    }
}

pub fn case(seed: u64) -> (Vec<Profile>, Scenario) {
    let mut rng = Random(seed);
    let mut p = scenarios::default_profiles().remove(0);
    p.id = "generic-warm".into();
    p.model = "example-model".into();
    p.gpu = "synthetic-accelerator".into();
    p.evidence = "synthetic generative test".into();
    p.source = "seeded corpus".into();
    p.max_in_flight = (rng.next(8) + 2) as usize;
    p.kv_tokens = rng.next(80_000) + 10_000;
    p.usd_per_hour = (rng.next(100) + 10) as f64 / 100.0;
    let capacity = (rng.next(1500) + 100) as f64;
    p.curve = vec![
        CapacityPoint {
            in_flight: 1,
            work_units_per_s: capacity / 4.0,
        },
        CapacityPoint {
            in_flight: p.max_in_flight,
            work_units_per_s: capacity,
        },
    ];
    p.startup.engine_s = (rng.next(40) + 1) as f64;
    p.startup.allocation_s = 0.0;
    p.startup.container_s = 0.0;
    p.startup.weights_s = 0.0;
    p.startup.restore_samples_s.clear();
    p.startup.build_samples_s.clear();
    p.startup.jitter = 0.5;
    p.network_s = rng.next(100) as f64 / 100.0;
    p.adapter_load_s = rng.next(100) as f64 / 100.0;
    p.adapter_cache_slots = 4;
    p.host = HostProfile::default();
    let mut fast = p.clone();
    fast.id = "generic-elastic".into();
    fast.usd_per_hour *= 2.0;
    for point in &mut fast.curve {
        point.work_units_per_s *= 2.0;
    }
    fast.startup.engine_s *= 0.5;
    let mut profiles = vec![p, fast];
    let templates = [profiles[0].clone(), profiles[1].clone()];
    let model_count = rng.next(4) + 1;
    for index in 1..model_count {
        for template in &templates {
            let mut profile = template.clone();
            profile.id = format!("{}-{index}", profile.id);
            profile.model = format!("example-model-{index}");
            profile.workload = if index.is_multiple_of(2) {
                Workload::Vision
            } else {
                Workload::Text
            };
            profiles.push(profile);
        }
    }
    let arrival = rng.next(60) as f64;
    let requests = (rng.next(60) + 1) as usize;
    let mut scenario = Scenario {
        name: format!("fuzz-{seed}"),
        notes: "Generated independent OpenAI-compatible request shapes; no private data".into(),
        horizon_s: 240.0,
        tick_s: 0.25,
        idle_timeout_s: (rng.next(30) + 1) as f64,
        max_nodes_per_model: (rng.next(5) + 1) as usize,
        launch_fail_probability: if rng.next(5) == 0 { 0.25 } else { 0.0 },
        reactive_delay_s: 2.0,
        capacity_scale: 0.5 + rng.next(150) as f64 / 100.0,
        startup_scale: 0.5 + rng.next(150) as f64 / 100.0,
        build_probability: 0.0,
        scheduling: Scheduling::default(),
        host_reality: None,
        warm: if rng.next(2) == 0 {
            vec![]
        } else {
            vec![WarmNode {
                profile: "generic-warm".into(),
                copies: 1,
            }]
        },
        jobs: vec![Job {
            id: 0,
            name: "generic-requests".into(),
            announced_s: 0.0,
            arrival_s: arrival,
            completion_budget_s: 180.0,
            desired_output_tps: rng.next(1200) as f64,
        }],
        requests: vec![],
        faults: vec![],
    };
    for id in 0..requests {
        let class = if rng.next(4) == 0 { "latency" } else { "batch" };
        let model = rng.next(model_count);
        let vision = model > 0 && model.is_multiple_of(2);
        scenario.requests.push(Request {
            id,
            job: 0,
            model: if model == 0 {
                "example-model".into()
            } else {
                format!("example-model-{model}")
            },
            workload: if vision {
                Workload::Vision
            } else {
                Workload::Text
            },
            class: class.into(),
            arrival_s: arrival
                + if rng.next(2) == 0 {
                    0.0
                } else {
                    id as f64 * 0.5
                },
            input_tokens: rng.next(8000) + 1,
            planned_output: rng.next(450) + 50,
            max_output: 1000,
            actual_output: rng.next(1000) + 1,
            image_megapixels: if vision {
                rng.next(800) as f64 / 100.0
            } else {
                0.0
            },
            upload_bytes: rng.next(1_000_000),
            cached_fraction: rng.next(50) as f64 / 100.0,
            adapter: (rng.next(2) == 0).then(|| format!("adapter-{}", rng.next(4))),
            latency_target_s: 5.0,
            depends_on: if id > 0 && rng.next(8) == 0 {
                vec![rng.next(id as u64) as usize]
            } else {
                vec![]
            },
        });
    }
    if !scenario.warm.is_empty() && rng.next(4) == 0 {
        scenario.faults.push(Fault {
            at_s: arrival + 5.0,
            node: 0,
        });
    }
    if rng.next(5) == 0 {
        scenario.host_reality = Some(HostProfile {
            model_bytes: 100_000_000,
            container_image_bytes: 100_000_000,
            volume_bandwidth_bytes_per_s: vec![1_000_000.0, 200_000_000.0],
            ..HostProfile::default()
        });
    }
    (profiles, scenario)
}

/// Check invariants independently of the controller's objective or success rate.
pub fn check(run: &engine::Run, profiles: &[Profile], scenario: &Scenario) -> Result<(), String> {
    let s = &run.summary;
    if s.completed + s.rejected + s.unfinished != scenario.requests.len() {
        return Err("request conservation".into());
    }
    if !s.cost_usd.is_finite()
        || s.cost_usd < 0.0
        || s.idle_cost_usd < -1e-8
        || s.idle_cost_usd > s.cost_usd + 1e-8
    {
        return Err("invalid cost accounting".into());
    }
    if s.invalid_decisions != 0 {
        return Err(format!(
            "{} invalid controller decisions",
            s.invalid_decisions
        ));
    }
    if s.max_active_per_node > profiles.iter().map(|p| p.max_in_flight).max().unwrap_or(0) {
        return Err("batch overflow".into());
    }
    if s.max_reserved_kv_tokens > profiles.iter().map(|p| p.kv_tokens).max().unwrap_or(0) {
        return Err("memory overflow".into());
    }
    let expected: u64 = run
        .requests
        .iter()
        .filter(|r| r.completed_s.is_some_and(|t| t <= scenario.horizon_s))
        .map(|r| scenario.requests[r.id].actual_output)
        .sum();
    if expected != s.output_tokens {
        return Err("completed token accounting".into());
    }
    for r in &run.requests {
        if let Some(first) = r.first_token_s {
            if first < scenario.requests[r.id].arrival_s {
                return Err("token before arrival".into());
            }
            if r.completed_s.is_some_and(|end| first > end + 1e-7) {
                return Err("token after completion".into());
            }
        }
    }
    Ok(())
}
