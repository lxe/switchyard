use inference_fleet_sim::{POLICIES, performance};

#[test]
fn software_benchmark_checks_real_work_and_all_controllers() {
    for policy in POLICIES {
        let result = performance::measure(100, policy, 1).unwrap();
        assert_eq!(result.completed, 100);
        assert!(result.wall_ms_p50.is_finite() && result.wall_ms_p50 > 0.0);
        assert!(result.simulated_output_tps > 0.0);
    }
}
