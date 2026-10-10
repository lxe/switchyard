use switchyard_live::config::*;
use switchyard_provider::{ComputeConfig, StorageConfig};
fn compute() -> ComputeConfig {
    ComputeConfig {
        api_url: "https://api.runpod.io/v2".into(),
        key_env: "RUNPOD_API_KEY".into(),
        cloud: "SECURE".into(),
        gpu_types: vec!["NVIDIA RTX A6000".into(), "NVIDIA GeForce RTX 4090".into()],
        data_centers: vec!["US-KS-2".into()],
        min_cuda: "12.9".into(),
        disk_gb: 80,
        disk_usd_per_gb_month: 0.1,
        billing_hours_per_month: 730.0,
    }
}
fn storage(endpoint: &str) -> StorageConfig {
    StorageConfig {
        endpoint: endpoint.into(),
        region: "auto".into(),
        bucket: "bucket".into(),
        access_key_env: "R2_ACCESS_KEY".into(),
        secret_key_env: "R2_SECRET_KEY".into(),
        url_lifetime_s: 3600,
    }
}
fn config() -> DeploymentConfig {
    DeploymentConfig {
        compute: compute(),
        bootstrap: BootstrapConfig {
            image: "fixture".into(),
            agent_url: "https://example.com/agent".into(),
            agent_sha256: "a".repeat(64),
            ticket_prefix: "switchyard/tickets".into(),
        },
        storage: storage("https://objects.example.test"),
        models: vec![ModelConfig {
            id: "qwen-small".into(),
            repository: "Qwen/Qwen3.5-0.8B".into(),
            revision: "a".repeat(40),
            archive_prefix: "models/qwen/revision/s1".into(),
            archive_sha256: "a".repeat(64),
            engine: EngineConfig {
                max_model_len: 4096,
                max_num_seqs: 32,
                gpu_memory_utilization: 0.8,
                dtype: "bfloat16".into(),
                language_model_only: true,
                extra_args: vec!["--enforce-eager".into()],
            },
            prior: CapacityPrior {
                single_output_tps: 100.0,
                aggregate_output_tps: 1000.0,
                input_work_per_token: 0.1,
                startup_s: 120.0,
                kv_tokens: 131072,
            },
        }],
        limits: Limits {
            max_instances: 2,
            max_usd_per_hour: 2.0,
            max_run_usd: 5.0,
            max_experiment_usd: Some(5.0),
            experiment_timeout_s: Some(3600),
            run_timeout_s: 1800,
            startup_timeout_s: 900,
            request_timeout_s: 180,
            hydration_concurrency: 16,
            hydration_stall_s: 20,
            poll_ms: 100,
        },
    }
}

#[test]
fn configuration_rejects_overridden_bindings_and_invalid_resource_limits() {
    config().validate().unwrap();
    for flag in [
        "--host",
        "--port=1234",
        "--config",
        "--api-key",
        "--max-model-len=0",
    ] {
        let mut bad = config();
        bad.models[0].engine.extra_args.push(flag.into());
        assert!(bad.validate().is_err(), "{flag}");
    }
    let mut bad = config();
    bad.models[0].engine.gpu_memory_utilization = 0.0;
    assert!(bad.validate().is_err());
    let mut bad = config();
    bad.models[0].prior.single_output_tps = 2000.0;
    assert!(bad.validate().is_err());
    let mut bad = config();
    bad.models[0].prior.kv_tokens = 4095;
    assert!(bad.validate().is_err());
    let mut bad = config();
    bad.limits.max_run_usd = f64::NAN;
    assert!(bad.validate().is_err());
    let mut bad = config();
    bad.storage.url_lifetime_s = 60;
    assert!(bad.validate().is_err());
    let mut bad = config();
    bad.models[0].id = "..".into();
    assert!(bad.validate().is_err());
    let mut bad = config();
    bad.models.push(bad.models[0].clone());
    assert!(bad.validate().is_err());
    let mut bad = config();
    bad.bootstrap.agent_sha256 = "z".repeat(64);
    assert!(bad.validate().is_err());
}

#[test]
fn engine_arguments_keep_model_and_limits_as_single_managed_values() {
    let cfg = config();
    let model = &cfg.models[0];
    let args = model.engine.args(model, "/work/switchyard/model");
    assert_eq!(&args[..2], ["serve", "/work/switchyard/model"]);
    assert_eq!(args.iter().filter(|a| a.as_str() == "--host").count(), 1);
    assert_eq!(
        args.iter()
            .filter(|a| a.as_str() == "--language-model-only")
            .count(),
        1
    );
    assert!(args.windows(2).any(|a| a == ["--host", "127.0.0.1"]));
    assert!(
        args.windows(2)
            .any(|a| a == ["--served-model-name", "qwen-small"])
    );
    assert!(args.windows(2).any(|a| a == ["--max-model-len", "4096"]));
}

#[test]
fn optional_experiment_limits_validate_and_preserve_existing_configurations() {
    let cfg = config();
    let mut value = serde_json::to_value(&cfg).unwrap();
    value["limits"]
        .as_object_mut()
        .unwrap()
        .remove("max_experiment_usd");
    value["limits"]
        .as_object_mut()
        .unwrap()
        .remove("experiment_timeout_s");
    let legacy: DeploymentConfig = serde_json::from_value(value).unwrap();
    assert_eq!(legacy.limits.max_experiment_usd, None);
    assert_eq!(legacy.limits.experiment_timeout_s, None);
    legacy.validate().unwrap();
    for cost in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        let mut bad = cfg.clone();
        bad.limits.max_experiment_usd = Some(cost);
        assert!(bad.validate().is_err());
    }
    let mut bad = cfg;
    bad.limits.experiment_timeout_s = Some(0);
    assert!(bad.validate().is_err());
}
