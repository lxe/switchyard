//! Keep production integrations outside the simulator and shared controller contracts.
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn simulator_and_scheduler_do_not_link_cloud_or_model_serving_integrations() {
    let output = std::process::Command::new(env!("CARGO"))
        .args(["metadata", "--locked", "--no-deps", "--format-version", "1"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(output.status.success());
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let pure = ["switchyard-core", "switchyard-scheduler", "switchyard-sim"];
    let integrations = [
        "switchyard-provider",
        "switchyard-agent",
        "switchyard-live",
        "aws-sigv4",
        "tokenizers",
        "minijinja",
        "dotenvy",
    ];
    let dependencies: BTreeMap<_, Vec<_>> = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|package| {
            let deps = package["dependencies"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|d| d["kind"].is_null())
                .map(|d| d["name"].as_str().unwrap())
                .collect();
            (package["name"].as_str().unwrap(), deps)
        })
        .collect();
    for root in pure {
        let mut pending = vec![(root, vec![root])];
        let mut visited = BTreeSet::new();
        while let Some((name, path)) = pending.pop() {
            if !visited.insert(name) {
                continue;
            }
            for &dep in dependencies.get(name).into_iter().flatten() {
                let mut next_path = path.clone();
                next_path.push(dep);
                assert!(
                    !integrations.contains(&dep),
                    "{root} must not link production integrations: {}",
                    next_path.join(" -> ")
                );
                pending.push((dep, next_path));
            }
        }
    }
}
