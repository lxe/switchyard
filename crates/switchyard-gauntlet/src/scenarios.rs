//! Repeatable workloads: independent requests, large bursts, and dependency chains.
use crate::config::*;

const MODELS: [&str; 5] = [
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "gemma4-e4b",
    "gemma4-31b",
];

fn base(name: &str, notes: &str, profiles: &[Profile], model_count: usize) -> Scenario {
    Scenario {
        topology: Default::default(),
        name: name.into(),
        notes: notes.into(),
        horizon_s: 900.0,
        tick_s: 0.1,
        idle_timeout_s: 30.0,
        max_nodes_per_model: 6,
        launch_fail_probability: 0.0,
        reactive_delay_s: 8.0,
        capacity_scale: 1.0,
        startup_scale: 1.0,
        build_probability: 0.0,
        scheduling: Scheduling::default(),
        host_reality: None,
        warm: MODELS
            .iter()
            .take(model_count)
            .filter_map(|model| {
                profiles
                    .iter()
                    .find(|p| p.model == *model && p.id.starts_with("baseline-"))
                    .map(|p| WarmNode {
                        profile: p.id.clone(),
                        copies: 1,
                    })
            })
            .collect(),
        jobs: vec![],
        requests: vec![],
        faults: vec![],
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "fixture generation keeps the workload parameters explicit"
)]
fn add_job(
    s: &mut Scenario,
    name: &str,
    model: &str,
    workload: Workload,
    class: &str,
    announced: f64,
    arrival: f64,
    n: usize,
    gap: f64,
    input: u64,
    planned: u64,
    actual: u64,
    images: f64,
    target: f64,
    budget: f64,
) {
    let job = s.jobs.len();
    s.jobs.push(Job {
        id: job,
        name: name.into(),
        announced_s: announced,
        arrival_s: arrival,
        completion_budget_s: budget,
        desired_output_tps: target,
    });
    for i in 0..n {
        s.requests.push(Request {
            id: s.requests.len(),
            job,
            model: model.into(),
            workload,
            class: class.into(),
            arrival_s: arrival + i as f64 * gap,
            input_tokens: input
                + if class == "batch" {
                    (i as u64 % 7) * input / 20
                } else {
                    0
                },
            planned_output: planned,
            max_output: planned.max(actual + 16),
            actual_output: actual + if class == "batch" { i as u64 % 17 } else { 0 },
            image_megapixels: images,
            cached_fraction: 0.0,
            adapter: (class == "batch").then(|| format!("adapter-{}", i % 4)),
            latency_target_s: 5.0,
            upload_bytes: (images * 500_000.0) as u64,
            depends_on: vec![],
        });
    }
}

/// Built-in synthetic workloads mirror recorded shapes without private data.
pub fn suite(profiles: &[Profile]) -> Vec<Scenario> {
    let mut quiet = base(
        "sparse_calls",
        "Sparse turn-by-turn requests, one warm base; 15-minute rent included.",
        profiles,
        1,
    );
    add_job(
        &mut quiet,
        "short-calls",
        MODELS[0],
        Workload::Text,
        "latency",
        0.0,
        0.0,
        80,
        10.0,
        500,
        128,
        128,
        0.0,
        0.0,
        900.0,
    );

    let mut small = base(
        "small_burst_200_tps",
        "72-request burst; aggregate desired output throughput 200 TPS, not per-stream decode.",
        profiles,
        1,
    );
    add_job(
        &mut small,
        "small-burst",
        MODELS[0],
        Workload::Text,
        "batch",
        0.0,
        30.0,
        72,
        0.0,
        500,
        220,
        200,
        0.0,
        200.0,
        180.0,
    );

    let mut mixed = base(
        "mixed_request_burst",
        "1,200 Qwen2B requests arrive at 30 s; short calls and dependent calls share its base. Output bound420, actual~328.",
        profiles,
        2,
    );
    add_job(
        &mut mixed,
        "short-calls",
        MODELS[1],
        Workload::Text,
        "latency",
        0.0,
        0.0,
        180,
        4.0,
        500,
        128,
        128,
        0.0,
        0.0,
        900.0,
    );
    add_job(
        &mut mixed,
        "request-burst",
        MODELS[1],
        Workload::Text,
        "batch",
        0.0,
        30.0,
        1200,
        0.0,
        1362,
        420,
        320,
        0.0,
        3000.0,
        180.0,
    );
    add_job(
        &mut mixed,
        "dependent-call",
        MODELS[1],
        Workload::Text,
        "standard",
        30.0,
        30.0,
        300,
        1.0,
        900,
        100,
        80,
        0.0,
        0.0,
        600.0,
    );
    for i in 0..300 {
        mixed.requests[1380 + i].depends_on.push(180 + i * 4);
    }

    let mut five = base(
        "five_model_burst",
        "Five independent 1,000-request workloads; last two vision profiles are uncalibrated hypotheses.",
        profiles,
        5,
    );
    for (i, model) in MODELS.iter().enumerate() {
        let vision = i >= 3;
        add_job(
            &mut five,
            model,
            model,
            if vision {
                Workload::Vision
            } else {
                Workload::Text
            },
            "batch",
            0.0,
            30.0,
            1000,
            0.0,
            if vision { 1362 } else { 500 },
            420,
            320,
            if vision { 1.0 } else { 0.0 },
            2000.0,
            240.0,
        );
    }

    let mut receipts = base(
        "receipt_vision_burst",
        "Qwen4B image burst, 64 requests and 32-client historical shape represented with an assumed per-node vision curve.",
        profiles,
        0,
    );
    receipts.warm.push(WarmNode {
        profile: "vision-baseline-qwen3.5-4b".into(),
        copies: 1,
    });
    add_job(
        &mut receipts,
        "receipts",
        MODELS[2],
        Workload::Vision,
        "batch",
        0.0,
        10.0,
        64,
        0.0,
        1362,
        512,
        382,
        1.0,
        200.0,
        120.0,
    );

    let mut cold = mixed.clone();
    cold.name = "cold_only_burst".into();
    cold.notes =
        "Same mixed burst, with no warm baseline. Warm-only intentionally cannot serve it.".into();
    cold.warm.clear();
    let mut slow = mixed.clone();
    slow.name = "slow_start_burst".into();
    slow.notes =
        "All startup times multiplied by5; tests whether elasticity arrives too late.".into();
    slow.startup_scale = 5.0;
    let mut long = mixed.clone();
    long.name = "long_context_burst".into();
    long.notes = "8x input length; memory admission and estimated prefill dominate.".into();
    for r in &mut long.requests {
        if r.class == "batch" {
            r.input_tokens *= 8;
        }
    }
    let mut fail = mixed.clone();
    fail.name = "capacity_and_node_failure".into();
    fail.notes =
        "20% deterministic launch failures plus one warm node lost at45s; retries restart streams."
            .into();
    fail.launch_fail_probability = 0.2;
    fail.faults = vec![Fault {
        at_s: 45.0,
        node: 1,
    }];
    let mut underestimate = mixed.clone();
    underestimate.name = "output_underestimate".into();
    underestimate.notes =
        "Actual burst outputs exceed planning estimates by 50%; the runtime replans remaining work."
            .into();
    for r in &mut underestimate.requests {
        if r.class == "batch" {
            r.actual_output = r.planned_output * 3 / 2;
            r.max_output = r.actual_output;
        }
    }
    let mut lowcap = mixed.clone();
    lowcap.name = "half_capacity".into();
    lowcap.notes =
        "Every profile capacity halved; explicitly tests unmeasured profile uncertainty.".into();
    lowcap.capacity_scale = 0.5;
    let mut highcap = mixed.clone();
    highcap.name = "double_capacity".into();
    highcap.capacity_scale = 2.0;
    let mut rebuild = mixed.clone();
    rebuild.name = "snapshot_build_burst".into();
    rebuild.notes = "Force initial build rather than normal restore; empirical selected build samples, no inferred production probability.".into();
    rebuild.build_probability = 1.0;
    let mut single = base(
        "single_cold_call",
        "One unannounced OpenAI-compatible call; compare client-visible TTFT and total rental cost.",
        profiles,
        0,
    );
    single.reactive_delay_s = 0.0;
    add_job(
        &mut single,
        "one-request",
        MODELS[1],
        Workload::Text,
        "standard",
        0.0,
        0.0,
        1,
        0.0,
        500,
        200,
        180,
        0.0,
        0.0,
        60.0,
    );
    let mut thousand = base(
        "thousand_call_burst",
        "1,000 independent calls, known input lengths and output estimates; no application-specific scheduling role.",
        profiles,
        2,
    );
    add_job(
        &mut thousand,
        "1000-calls",
        MODELS[1],
        Workload::Text,
        "batch",
        0.0,
        30.0,
        1000,
        0.0,
        500,
        400,
        320,
        0.0,
        2000.0,
        180.0,
    );
    let mut hydration = cold.clone();
    hydration.name = "unknown_hydration_hosts".into();
    hydration.notes = "Hidden5/250MB/s volume links and unknown caches; probe-aware controller can cancel slow starts and try another host.".into();
    hydration.host_reality = Some(HostProfile {
        volume_bandwidth_bytes_per_s: vec![5_000_000.0, 250_000_000.0],
        container_image_bytes: 2_000_000_000,
        model_bytes: 4_000_000_000,
        image_cache_probability: 0.25,
        model_cache_probability: 0.25,
        ..HostProfile::default()
    });
    vec![
        quiet,
        small,
        mixed,
        five,
        receipts,
        cold,
        slow,
        long,
        fail,
        underestimate,
        lowcap,
        highcap,
        rebuild,
        single,
        thousand,
        hydration,
    ]
}

#[expect(
    clippy::too_many_arguments,
    reason = "fixture parameters are intentionally explicit"
)]
fn profile(
    id: &str,
    model: &str,
    workload: Workload,
    gpu: &str,
    price: f64,
    startup: f64,
    slots: usize,
    peak: f64,
    evidence: &str,
    source: &str,
) -> Profile {
    Profile {
        id: id.into(),
        model: model.into(),
        workload,
        gpu: gpu.into(),
        usd_per_hour: price,
        startup: Startup {
            allocation_s: startup * 0.3,
            container_s: startup * 0.2,
            weights_s: startup * 0.1,
            engine_s: startup * 0.4,
            jitter: 0.25,
            restore_samples_s: vec![],
            build_samples_s: vec![],
        },
        network_s: 0.03,
        network_bytes_per_s: 100_000_000.0,
        max_in_flight: slots,
        kv_tokens: slots as u64 * 4096,
        input_work_per_token: 0.025,
        image_work_per_megapixel: if workload == Workload::Vision {
            50.0
        } else {
            0.0
        },
        adapter_load_s: 0.1,
        multi_adapter_factor: 0.75,
        adapter_cache_slots: 8,
        minimum_bill_s: 1.0,
        curve: vec![
            CapacityPoint {
                in_flight: 1,
                work_units_per_s: peak / 8.0,
            },
            CapacityPoint {
                in_flight: slots,
                work_units_per_s: peak,
            },
        ],
        evidence: evidence.into(),
        source: source.into(),
        host: HostProfile::default(),
    }
}

/// Example profiles distinguish historical calibration from unmeasured hardware.
pub fn default_profiles() -> Vec<Profile> {
    let mut profiles = vec![];
    for (i, model) in MODELS.iter().enumerate() {
        let workload = if i >= 3 {
            Workload::Vision
        } else {
            Workload::Text
        };
        let baseline_peak = [1200.0, 1000.0, 750.0, 500.0, 200.0][i];
        profiles.push(profile(
            &format!("baseline-{model}"),
            model,
            workload,
            if i == 4 { "RTX-PRO-6000" } else { "A6000" },
            if i == 4 { 2.49 } else { 0.59 },
            120.0,
            16,
            baseline_peak,
            "assumed: no matching A6000 workload measurement",
            "Research rental rate; throughput/startup/batch/memory estimates for sensitivity only",
        ));
        let (gpu, price, peak, evidence) = match i {
            0 => (
                "A10",
                1.25,
                2856.667 * 1.0625,
                "boundary_adjusted: Oct8 base text,64 clients; subtract64*200/60; low-concurrency curve assumed",
            ),
            1 => (
                "A10",
                1.25,
                1813.367 * 1.0625,
                "boundary_adjusted: Oct8 base text,64 clients; subtract64*200/60; low-concurrency curve assumed",
            ),
            2 => (
                "L40S",
                2.11,
                1386.667 * 1.0625,
                "boundary_adjusted: Oct8 base text,64 clients; subtract64*200/60; low-concurrency curve assumed",
            ),
            3 => (
                "H100",
                4.15,
                900.0,
                "assumed: Oct10 GemmaE4B aggregate is fleet-level, container count unknown",
            ),
            _ => (
                "RTX-PRO-6000",
                3.25,
                450.0,
                "assumed: Gemma31B output throughput not recorded",
            ),
        };
        profiles.push(profile(&format!("elastic-{model}"), model, workload, gpu, price, if i >= 3 { 45.0 } else { 25.0 },
            if i >= 3 { 8 } else { 32 }, peak, evidence, "evidence/bullettrain-measurements.md; macro coefficient calibrated at500 input/200 output;32 engine sequences saturated by64 clients"));
        // Faster alternatives are intentional hypotheses, never mislabeled measurements.
        profiles.push(profile(
            &format!("fast-{model}"),
            model,
            workload,
            if i == 4 { "H200" } else { "H100" },
            if i == 4 { 4.75 } else { 4.15 },
            18.0,
            16,
            peak * 1.8,
            "assumed: H100 comparative capacity unmeasured for this exact workload",
            "Hypothetical fast-GPU experiment;25% startup jitter",
        ));
    }
    profiles.push(profile(
        "vision-baseline-qwen3.5-4b",
        MODELS[2],
        Workload::Vision,
        "A6000",
        0.59,
        120.0,
        8,
        450.0,
        "assumed: no A6000 vision benchmark",
        "Explicit vision-only hypothesis",
    ));
    profiles.push(profile(
        "vision-elastic-qwen3.5-4b",
        MODELS[2],
        Workload::Vision,
        "H100",
        4.15,
        22.0,
        8,
        1500.0,
        "assumed: fleet2722.6 output TPS cannot be normalized per GPU",
        "Oct10 fleet shape used as comparison only, not per-node calibration",
    ));
    crate::calibration::apply_calibration(&mut profiles);
    profiles
}
