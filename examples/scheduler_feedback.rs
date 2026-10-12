//! Export an analytical late-capacity regression fixture for cross-version replay.
use std::{fs, path::PathBuf};
use switchyard_gauntlet::{CapacityPoint, HostProfile, WarmNode, performance};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "results/scheduler-feedback-fixture".into());
    let (mut profiles, mut scenario) = performance::workload(32);
    let p = &mut profiles[0];
    p.id = "analytical-late-capacity".into();
    p.max_in_flight = 8;
    p.kv_tokens = 8192;
    p.input_work_per_token = 0.0;
    p.curve = vec![
        CapacityPoint {
            in_flight: 1,
            work_units_per_s: 20.0,
        },
        CapacityPoint {
            in_flight: 8,
            work_units_per_s: 100.0,
        },
    ];
    p.usd_per_hour = 1.0;
    p.startup.allocation_s = 0.0;
    p.startup.container_s = 0.0;
    p.startup.weights_s = 0.0;
    p.startup.engine_s = 45.0;
    p.host = HostProfile::default();
    scenario.name = "late-capacity".into();
    scenario.horizon_s = 100.0;
    scenario.max_nodes_per_model = 8;
    scenario.idle_timeout_s = 5.0;
    scenario.scheduling.reserve_warm_slots = 0;
    scenario.scheduling.warm_batch_capacity_fraction = 1.0;
    scenario.jobs[0].desired_output_tps = 2000.0;
    scenario.jobs[0].completion_budget_s = 100.0;
    scenario.warm = vec![WarmNode {
        profile: p.id.clone(),
        copies: 1,
    }];
    for r in &mut scenario.requests {
        r.input_tokens = 32;
        r.planned_output = 32;
        r.actual_output = 32;
        r.max_output = 32;
        r.upload_bytes = 0;
    }
    switchyard_gauntlet::validate(&profiles, &scenario)?;
    fs::create_dir_all(&out)?;
    fs::write(
        out.join("profiles.json"),
        serde_json::to_vec_pretty(&profiles)?,
    )?;
    let mut portable = serde_json::to_value(&scenario)?;
    // Omit default new controls so older executables can read identical inputs.
    portable["scheduling"]
        .as_object_mut()
        .expect("scheduling object")
        .remove("heuristics");
    fs::write(
        out.join("scenario.json"),
        serde_json::to_vec_pretty(&portable)?,
    )?;
    Ok(())
}
