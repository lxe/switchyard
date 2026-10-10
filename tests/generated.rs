use inference_fleet_sim::*;

#[test]
fn generated_workloads_preserve_accounting_and_admission_invariants() {
    for seed in 0..128 {
        let (profiles, scenario) = fuzz::case(seed);
        for policy in POLICIES {
            let result = simulate(&profiles, &scenario, policy, seed).unwrap();
            fuzz::check(&result, &profiles, &scenario)
                .unwrap_or_else(|error| panic!("seed{seed},policy{policy}:{error}"));
        }
    }
}
