use switchyard_live::{artifacts::runtime_launch_fingerprint, config::DeploymentConfig};

#[test]
fn engine_startup_policy_is_validated_and_invalidates_launch_compatibility() {
    let mut config: DeploymentConfig = serde_json::from_str(include_str!(
        "../../../configs/runpod/qwen-extents-l40s-strict.json"
    ))
    .unwrap();
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

#[test]
fn invalid_native_startup_policy_is_rejected_before_provisioning() {
    let original: DeploymentConfig = serde_json::from_str(include_str!(
        "../../../configs/runpod/qwen-extents-l40s-strict.json"
    ))
    .unwrap();
    for encoded in [
        "not json",
        r#"{"worker_method":"forkserver"}"#,
        r#"{"strategy":"prepared","worker_method":"guarded_fork","fork_runtime_id":"wrong","compute_capability":"8.9"}"#,
        r#"{"schema_version":2}"#,
        r#"{"tokenizer_preload":1}"#,
    ] {
        let mut config = original.clone();
        config
            .bootstrap
            .runtime
            .as_mut()
            .unwrap()
            .environment
            .insert("VLLM_ENGINE_STARTUP_POLICY".into(), encoded.into());
        assert!(config.validate().is_err());
    }
    let mut legacy: DeploymentConfig =
        serde_json::from_str(include_str!("../../../configs/runpod/qwen-small.json")).unwrap();
    legacy
        .bootstrap
        .runtime
        .as_mut()
        .unwrap()
        .environment
        .insert(
            "VLLM_ENGINE_STARTUP_POLICY".into(),
            r#"{"strategy":"prepared"}"#.into(),
        );
    assert!(
        legacy
            .validate()
            .unwrap_err()
            .to_string()
            .contains("extent-backed")
    );
}

#[test]
fn native_guard_requirements_are_checked_before_provisioning() {
    let original: DeploymentConfig = serde_json::from_str(include_str!(
        "../../../configs/runpod/qwen-extents-l40s-strict.json"
    ))
    .unwrap();
    for (policy, disabled) in [
        (r#"{"cache_policy":"required"}"#, "VLLM_FORCE_AOT_LOAD"),
        (r#"{"memory_plan":"required"}"#, "VLLM_ENABLE_STARTUP_PLAN"),
    ] {
        let mut config = original.clone();
        for model in &mut config.models {
            model.caches.clear();
        }
        let runtime = config.bootstrap.runtime.as_mut().unwrap();
        runtime
            .environment
            .insert("VLLM_ENGINE_STARTUP_POLICY".into(), policy.into());
        config.validate().unwrap();
        config
            .bootstrap
            .runtime
            .as_mut()
            .unwrap()
            .environment
            .insert(disabled.into(), "0".into());
        assert!(config.validate().is_err());
    }
}
