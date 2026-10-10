//! Keep production integrations outside the simulator and shared controller contracts.
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
    for package in metadata["packages"].as_array().unwrap() {
        let name = package["name"].as_str().unwrap();
        if pure.contains(&name) {
            for dep in package["dependencies"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|d| d["kind"].is_null())
            {
                assert!(
                    !integrations.contains(&dep["name"].as_str().unwrap()),
                    "{name} must not link {}",
                    dep["name"]
                );
            }
        }
    }
}
