//! Calibration and residual reports over portable, provider-independent observations.
use crate::{Policy, Profile, Scenario};
use serde::{Deserialize, Serialize};
use switchyard_core::stats::percentile;
use switchyard_scheduler::BuiltinScheduler;
use switchyard_sim::{engine::simulate_with_observations, observations::Observations};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub provenance: serde_json::Value,
    pub profiles: Vec<Profile>,
    pub cases: Vec<Case>,
}

impl Dataset {
    pub fn validate(&self) -> Result<(), String> {
        if self.cases.is_empty() {
            return Err("calibration dataset has no cases".into());
        }
        for c in &self.cases {
            crate::validate(&self.profiles, &c.scenario)?;
            c.observations.validate(&self.profiles, &c.scenario)?;
            let m = &c.measured;
            if [
                m.elapsed_s,
                m.output_tps,
                m.ttft_p50_s,
                m.ttft_p95_s,
                m.ttft_p99_s,
            ]
            .into_iter()
            .any(|v| !v.is_finite() || v <= 0.0)
                || !m.rental_usd.is_finite()
                || m.rental_usd < 0.0
                || m.completed > c.scenario.requests.len()
                || m.ttft_p50_s > m.ttft_p95_s
                || m.ttft_p95_s > m.ttft_p99_s
                || c.scenario.requests.iter().any(|r| r.model != c.model)
            {
                return Err("invalid calibration reference metrics or model identity".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub model: String,
    pub policy: Policy,
    /// Training selection must precede fitting; validation cases never enter its loss.
    pub training: bool,
    pub scenario: Scenario,
    pub observations: Observations,
    pub measured: Metrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub completed: usize,
    pub elapsed_s: f64,
    pub output_tps: f64,
    pub ttft_p50_s: f64,
    pub ttft_p95_s: f64,
    pub ttft_p99_s: f64,
    pub rental_usd: f64,
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    pub model: String,
    pub workload: String,
    pub training: bool,
    pub seed: u64,
    pub measured: Metrics,
    pub simulated: Metrics,
    pub throughput_error_fraction: f64,
    pub ttft_p95_error_fraction: f64,
    pub launched: usize,
    pub capacity_rejections: usize,
    pub invalid_decisions: usize,
}

pub fn replay(profiles: &[Profile], case: &Case, seed: u64) -> Result<Comparison, String> {
    let run = simulate_with_observations(
        profiles,
        &case.scenario,
        &case.observations,
        &mut BuiltinScheduler::new(case.policy),
        seed,
    )?;
    let done: Vec<_> = run
        .requests
        .iter()
        .filter(|r| r.completed_s.is_some_and(|t| t <= case.scenario.horizon_s))
        .collect();
    let elapsed = if done.len() == case.scenario.requests.len() {
        done.iter()
            .filter_map(|r| r.completed_s)
            .max_by(f64::total_cmp)
            .unwrap_or(case.scenario.horizon_s)
    } else {
        case.scenario.horizon_s
    };
    let tokens: u64 = done
        .iter()
        .map(|r| case.scenario.requests[r.id].actual_output)
        .sum();
    let latencies: Vec<_> = done
        .iter()
        .filter_map(|r| {
            r.first_token_s
                .map(|t| t - case.scenario.requests[r.id].arrival_s)
        })
        .collect();
    let rental_usd = run
        .nodes
        .iter()
        .map(|n| {
            if n.allocated_s >= elapsed {
                return 0.0;
            }
            let p = profiles
                .iter()
                .find(|p| p.id == n.profile)
                .expect("simulation profile");
            (n.ended_s.min(elapsed) - n.allocated_s)
                .max(0.0)
                .max(p.minimum_bill_s)
                * p.usd_per_hour
                / 3600.0
        })
        .sum();
    let simulated = Metrics {
        completed: done.len(),
        elapsed_s: elapsed,
        output_tps: tokens as f64 / elapsed.max(1e-9),
        ttft_p50_s: percentile(latencies.clone(), 0.5).unwrap_or(case.scenario.horizon_s),
        ttft_p95_s: percentile(latencies.clone(), 0.95).unwrap_or(case.scenario.horizon_s),
        ttft_p99_s: percentile(latencies, 0.99).unwrap_or(case.scenario.horizon_s),
        rental_usd,
    };
    Ok(Comparison {
        model: case.model.clone(),
        workload: case.scenario.name.clone(),
        training: case.training,
        seed,
        throughput_error_fraction: simulated.output_tps / case.measured.output_tps - 1.0,
        ttft_p95_error_fraction: simulated.ttft_p95_s / case.measured.ttft_p95_s - 1.0,
        simulated,
        measured: case.measured.clone(),
        launched: run.summary.launched,
        capacity_rejections: run.summary.capacity_rejections,
        invalid_decisions: run.summary.invalid_decisions,
    })
}

/// Fit four macroscopic service parameters on the declared training subset only.
/// These are effective end-to-end estimates, not GPU kernel benchmark claims.
pub fn fit(dataset: &mut Dataset) -> Result<(), String> {
    dataset.validate()?;
    if dataset.profiles.iter().any(|p| p.max_in_flight < 2) {
        return Err(
            "two-point capacity fitting needs max_in_flight >= 2; manual profiles can use one"
                .into(),
        );
    }
    let models: std::collections::BTreeSet<_> =
        dataset.cases.iter().map(|c| c.model.clone()).collect();
    for model in models {
        let cases: Vec<_> = dataset
            .cases
            .iter()
            .filter(|c| c.model == model && c.training)
            .collect();
        if cases.is_empty() {
            return Err(format!("{model} has no declared training cases"));
        }
        let mut parameters = [40.0, 180.0, 0.5, 0.15];
        let apply = |profiles: &mut [Profile], values: [f64; 4]| {
            for p in profiles.iter_mut().filter(|p| p.model == model) {
                p.curve = vec![
                    crate::CapacityPoint {
                        in_flight: 1,
                        work_units_per_s: values[0],
                    },
                    crate::CapacityPoint {
                        in_flight: p.max_in_flight,
                        work_units_per_s: values[1],
                    },
                ];
                p.input_work_per_token = values[2];
                p.network_s = values[3];
            }
        };
        let score = |profiles: &[Profile]| -> Result<f64, String> {
            let mut loss = 0.0;
            for c in &cases {
                let r = replay(profiles, c, 7)?;
                loss += (r.simulated.output_tps / c.measured.output_tps)
                    .ln()
                    .powi(2)
                    + (r.simulated.ttft_p50_s / c.measured.ttft_p50_s)
                        .ln()
                        .powi(2)
                    + (r.simulated.ttft_p95_s / c.measured.ttft_p95_s)
                        .ln()
                        .powi(2);
                if r.simulated.completed != c.measured.completed {
                    loss += 100.0;
                }
            }
            Ok(loss)
        };
        apply(&mut dataset.profiles, parameters);
        let mut best = score(&dataset.profiles)?;
        for step in [0.4, 0.2, 0.1, 0.05] {
            for _ in 0..5 {
                let mut improved = false;
                for axis in 0..4 {
                    for multiplier in [1.0 - step, 1.0 + step] {
                        let mut candidate = parameters;
                        candidate[axis] *= multiplier;
                        if candidate[1] < candidate[0] {
                            continue;
                        }
                        apply(&mut dataset.profiles, candidate);
                        let loss = score(&dataset.profiles)?;
                        if loss < best {
                            parameters = candidate;
                            best = loss;
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        apply(&mut dataset.profiles, parameters);
    }
    Ok(())
}
