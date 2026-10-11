use switchyard_live::{artifacts::runtime_launch_fingerprint, config::DeploymentConfig};

#[test]
fn engine_startup_policy_is_validated_and_invalidates_launch_compatibility() {
    let mut config: DeploymentConfig =
        serde_json::from_str(include_str!("../../../configs/runpod/qwen-small.json")).unwrap();
    let runtime = config.bootstrap.runtime.as_mut().unwrap();
    let before = runtime_launch_fingerprint(runtime).unwrap();
    runtime.environment.insert(
        "VLLM_ENGINE_STARTUP_POLICY".into(),
        r#"{"schema_version":1,"strategy":"prepared","worker_method":"spawn","tokenizer_preload":true}"#.into(),
    );
    let after = runtime_launch_fingerprint(runtime).unwrap();
    assert_ne!(before, after);
    // Existing baked caches pin the old launch policy and must be rebaked.
    assert!(config.validate().is_err());
    for model in &mut config.models {
        model.caches.clear();
    }
    config.validate().unwrap();
    config
        .bootstrap
        .runtime
        .as_mut()
        .unwrap()
        .environment
        .insert(
            "SWITCHYARD_ENGINE_POLICY".into(),
            "reserved agent-owned environment".into(),
        );
    assert!(config.validate().is_err());
}
