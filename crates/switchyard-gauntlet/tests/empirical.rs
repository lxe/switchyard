use switchyard_gauntlet::{Policy, empirical::*};
use switchyard_sim::observations::Observations;

fn fixture() -> Dataset {
    Dataset {
        provenance: serde_json::json!({"kind": "analytical regression fixture"}),
        profiles: serde_json::from_str(include_str!("../../../configs/quickstart/profiles.json"))
            .unwrap(),
        cases: vec![Case {
            model: "demo-model".into(),
            policy: Policy::WarmOnly,
            training: true,
            scenario: serde_json::from_str(include_str!("../../../configs/quickstart/burst.json"))
                .unwrap(),
            observations: Observations::default(),
            measured: Metrics {
                completed: 12,
                elapsed_s: 5.0,
                output_tps: 200.0,
                ttft_p50_s: 1.0,
                ttft_p95_s: 2.0,
                ttft_p99_s: 3.0,
                rental_usd: 0.001,
            },
        }],
    }
}

#[test]
fn held_out_reference_metrics_cannot_change_fitted_parameters() {
    let mut original = fixture();
    let mut held_out = original.cases[0].clone();
    held_out.training = false;
    original.cases.push(held_out);
    let mut changed = original.clone();
    let reference = &mut changed.cases[1].measured;
    reference.elapsed_s *= 10.0;
    reference.output_tps *= 10.0;
    reference.ttft_p50_s *= 10.0;
    reference.ttft_p95_s *= 10.0;
    reference.ttft_p99_s *= 10.0;
    fit(&mut original).unwrap();
    fit(&mut changed).unwrap();
    assert_eq!(
        serde_json::to_value(&original.profiles).unwrap(),
        serde_json::to_value(&changed.profiles).unwrap()
    );
}

#[test]
fn partial_workload_uses_full_horizon_in_throughput_denominator() {
    let mut dataset = fixture();
    let case = &mut dataset.cases[0];
    case.scenario.requests.truncate(2);
    case.scenario.horizon_s = 3.0;
    case.scenario.requests[0].actual_output = 1;
    case.scenario.requests[1].arrival_s = 2.99;
    case.measured.completed = 2;
    dataset.validate().unwrap();
    let result = replay(&dataset.profiles, &dataset.cases[0], 7).unwrap();
    assert_eq!(result.simulated.completed, 1);
    assert_eq!(result.simulated.elapsed_s, 3.0);
    assert!((result.simulated.output_tps - 1.0 / 3.0).abs() < 1e-12);
}
