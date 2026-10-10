//! Reproducible wall-clock benchmarks of the simulator and controller together.
use crate::*;
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Serialize)]
pub struct Benchmark {
    pub requests: usize,
    pub policy: Policy,
    pub repetitions: usize,
    pub wall_ms_p50: f64,
    pub wall_ms_p95: f64,
    pub simulated_requests_per_wall_second: f64,
    pub completed: usize,
    pub simulated_output_tps: f64,
}

/// A synthetic, fully specified workload isolates software execution overhead.
/// These capacities are analytical fixtures, not observations of a real GPU.
pub fn workload(count: usize) -> (Vec<Profile>, Scenario) {
    let (mut profiles, mut scenario) = fuzz::case(7);
    profiles.truncate(1);
    let profile = &mut profiles[0];
    profile.id = "benchmark-gpu".into();
    profile.model = "benchmark-model".into();
    profile.gpu = "synthetic".into();
    profile.evidence = "analytical software performance fixture".into();
    profile.max_in_flight = 64;
    profile.kv_tokens = 64 * 4096;
    profile.curve = vec![
        CapacityPoint {
            in_flight: 1,
            work_units_per_s: 1000.0,
        },
        CapacityPoint {
            in_flight: 64,
            work_units_per_s: 16_000.0,
        },
    ];
    profile.network_s = 0.002;
    profile.network_bytes_per_s = 100_000_000.0;
    profile.adapter_load_s = 0.0;
    profile.startup.engine_s = 2.0;
    profile.startup.jitter = 0.0;
    profile.host = HostProfile::default();
    let template = Request {
        id: 0,
        job: 0,
        model: "benchmark-model".into(),
        workload: Workload::Text,
        class: "batch".into(),
        arrival_s: 0.0,
        input_tokens: 1024,
        planned_output: 64,
        max_output: 64,
        actual_output: 64,
        image_megapixels: 0.0,
        upload_bytes: 4096,
        cached_fraction: 0.0,
        adapter: None,
        latency_target_s: 120.0,
        depends_on: vec![],
    };
    scenario.name = format!("software-performance-{count}");
    scenario.notes =
        "Synthetic software overhead benchmark; virtual throughput is not GPU evidence".into();
    scenario.horizon_s = 150.0_f64.max(count as f64 * 90.0 / 16_000.0 + 120.0);
    scenario.tick_s = 0.1;
    scenario.max_nodes_per_model = 8;
    scenario.launch_fail_probability = 0.0;
    scenario.capacity_scale = 1.0;
    scenario.startup_scale = 1.0;
    scenario.host_reality = None;
    scenario.warm = vec![WarmNode {
        profile: profile.id.clone(),
        copies: 1,
    }];
    scenario.faults.clear();
    scenario.jobs = vec![Job {
        id: 0,
        name: "benchmark-burst".into(),
        announced_s: 0.0,
        arrival_s: 0.0,
        completion_budget_s: scenario.horizon_s - 30.0,
        desired_output_tps: 0.0,
    }];
    scenario.requests = (0..count)
        .map(|id| Request {
            id,
            ..template.clone()
        })
        .collect();
    (profiles, scenario)
}

/// Measure full runs, validating conservation and admission on every repetition.
/// Timing thresholds belong to the caller because host CPU speed varies.
pub fn measure(count: usize, policy: Policy, repetitions: usize) -> Result<Benchmark, String> {
    if repetitions == 0 || count == 0 || count > 100_000 {
        return Err("benchmark requires positive repetitions and 1..100000 requests".into());
    }
    let (profiles, scenario) = workload(count);
    let mut elapsed = Vec::with_capacity(repetitions);
    let mut last = None;
    for _ in 0..repetitions {
        let start = Instant::now();
        let run = simulate(&profiles, &scenario, policy, 7)?;
        elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
        fuzz::check(&run, &profiles, &scenario)?;
        if run.summary.completed != count {
            return Err("software benchmark did not drain its workload".into());
        }
        last = Some(run);
    }
    let p50 = engine::percentile(elapsed.clone(), 0.5).expect("positive repetitions");
    let p95 = engine::percentile(elapsed, 0.95).expect("positive repetitions");
    let run = last.expect("positive repetitions");
    Ok(Benchmark {
        requests: count,
        policy,
        repetitions,
        wall_ms_p50: p50,
        wall_ms_p95: p95,
        simulated_requests_per_wall_second: count as f64 / (p50 / 1000.0),
        completed: run.summary.completed,
        simulated_output_tps: run.summary.jobs[0].output_tps,
    })
}
