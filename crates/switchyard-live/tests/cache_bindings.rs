//! Offline publication inputs use exactly the same typed identities as live cache reuse.
use switchyard_live::{
    artifacts,
    cli::{Command, run},
    config::DeploymentConfig,
};

#[tokio::test]
async fn offline_bindings_derive_live_identities_without_provider_credentials() {
    let mut nonce = [0u8; 16];
    getrandom::fill(&mut nonce).unwrap();
    let temp = std::env::temp_dir().join(format!(
        "switchyard-cache-bindings-{:032x}",
        u128::from_le_bytes(nonce)
    ));
    std::fs::create_dir(&temp).unwrap();
    let config = temp.join("fleet.json");
    let data = include_str!("../../../configs/runpod/qwen-small.json");
    std::fs::write(&config, data).unwrap();
    let out = temp.join("bindings.json");
    let gpu = "NVIDIA L40S";
    run(Command::CacheBindings {
        config: config.clone(),
        gpu: gpu.into(),
        model: Some("qwen3.5-0.8b".into()),
        out: out.clone(),
    })
    .await
    .unwrap();
    let actual: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let cfg: DeploymentConfig = serde_json::from_str(data).unwrap();
    let model = &cfg.models[0];
    let expected = artifacts::cache_requirements(
        &Default::default(),
        gpu,
        cfg.bootstrap.runtime.as_ref().unwrap(),
        model,
    )
    .unwrap();
    assert_eq!(actual[&model.id], serde_json::to_value(expected).unwrap());
    assert_eq!(actual.as_object().unwrap().len(), 1);
    assert!(!actual.to_string().contains("RUNPOD_API_KEY"));
    run(Command::CacheBindings {
        config,
        gpu: gpu.into(),
        model: Some("missing".into()),
        out: temp.join("missing.json"),
    })
    .await
    .unwrap_err();
    assert!(!temp.join("missing.json").exists());
    std::fs::remove_dir_all(temp).unwrap();
}
