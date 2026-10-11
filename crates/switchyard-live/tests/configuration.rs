use switchyard_core::{Scheduling, validate_scheduling};
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
        min_ram_per_gpu_gb: None,
        min_vcpu_count_per_gpu: None,
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
            startup: Default::default(),
            runtime: None,
        },
        storage: storage("https://objects.example.test"),
        models: vec![ModelConfig {
            id: "qwen-small".into(),
            repository: "Qwen/Qwen3.5-0.8B".into(),
            revision: "a".repeat(40),
            archive_prefix: "models/qwen/revision/s1".into(),
            archive_sha256: "a".repeat(64),
            caches: vec![],
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
        scheduling: Scheduling::default(),
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
fn startup_policy_preserves_defaults_and_validates_engine_dependency_gates() {
    let cfg = config();
    let mut value = serde_json::to_value(&cfg).unwrap();
    value["bootstrap"]
        .as_object_mut()
        .unwrap()
        .remove("startup");
    let legacy: DeploymentConfig = serde_json::from_value(value).unwrap();
    assert_eq!(
        legacy.bootstrap.startup.mode,
        switchyard_artifacts::HydrationMode::Verified
    );
    assert_eq!(
        legacy.bootstrap.startup.model_directory(),
        std::path::Path::new("/work/switchyard/model")
    );
    legacy.validate().unwrap();

    let mut configured = cfg;
    configured.bootstrap.startup.mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
    configured.bootstrap.startup.model_root = Some("/dev/shm/switchyard".into());
    configured.bootstrap.startup.require_model_ram = true;
    configured.validate().unwrap();
    assert_eq!(
        configured.execution().bootstrap.startup.model_directory(),
        std::path::Path::new("/dev/shm/switchyard/model")
    );
    configured.bootstrap.startup.model_root = Some("/dev/shm/../model".into());
    assert!(configured.validate().is_err());
    configured.bootstrap.startup.model_root = None;
    configured.bootstrap.startup.early_launch = true;
    assert!(configured.validate().is_err());
}

#[test]
fn serving_artifacts_are_optional_and_cache_identity_tracks_engine_configuration() {
    use switchyard_live::artifacts::{
        ArtifactRef, ServingCacheConfig, ServingRuntimeConfig, engine_fingerprint,
    };
    let mut cfg = config();
    let reference = ArtifactRef {
        prefix: format!("switchyard/artifacts/{}", "b".repeat(64)),
        id: "b".repeat(64),
    };
    cfg.bootstrap.runtime = Some(ServingRuntimeConfig {
        hydration: None,
        artifact: reference.clone(),
        compatibility: Default::default(),
        program: "venv/bin/python".into(),
        args: vec!["-m".into(), "vllm.entrypoints.cli.main".into()],
        environment: [("TOKENIZERS_PARALLELISM".into(), "false".into())]
            .into_iter()
            .collect(),
    });
    cfg.models[0].caches = vec![ServingCacheConfig {
        name: "engine".into(),
        gpu: "NVIDIA RTX A6000".into(),
        artifact: reference,
        compatibility: Default::default(),
    }];
    cfg.validate().unwrap();
    let before = cfg.models[0].caches[0]
        .requirements(cfg.bootstrap.runtime.as_ref().unwrap(), &cfg.models[0])
        .unwrap();
    assert_eq!(
        before["engine_configuration"],
        engine_fingerprint(&cfg.models[0].engine).unwrap()
    );
    cfg.models[0].engine.max_num_seqs = 16;
    let after = cfg.models[0].caches[0]
        .requirements(cfg.bootstrap.runtime.as_ref().unwrap(), &cfg.models[0])
        .unwrap();
    assert_ne!(
        before["engine_configuration"],
        after["engine_configuration"]
    );
    cfg.models[0].caches[0]
        .compatibility
        .insert("model_revision".into(), "wrong".into());
    assert!(cfg.validate().is_err());
}

#[test]
fn serving_configuration_rejects_escaping_programs_and_unbound_caches() {
    use switchyard_live::artifacts::{ArtifactRef, ServingCacheConfig, ServingRuntimeConfig};
    let mut cfg = config();
    cfg.models[0].caches = vec![ServingCacheConfig {
        name: "engine".into(),
        gpu: "gpu".into(),
        artifact: ArtifactRef {
            prefix: "artifacts/cache".into(),
            id: "b".repeat(64),
        },
        compatibility: Default::default(),
    }];
    assert!(cfg.validate().is_err());
    cfg.models[0].caches.clear();
    cfg.bootstrap.runtime = Some(ServingRuntimeConfig {
        hydration: None,
        artifact: ArtifactRef {
            prefix: "artifacts/runtime".into(),
            id: "b".repeat(64),
        },
        compatibility: Default::default(),
        program: "../bin/python".into(),
        args: vec![],
        environment: Default::default(),
    });
    assert!(cfg.validate().is_err());
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

#[test]
fn customized_scheduling_survives_deployment_projection_and_round_trip() {
    let mut value = serde_json::to_value(config()).unwrap();
    value["scheduling"] = serde_json::json!({
        "maximum_startup_s": 300.0,
        "minimum_hydration_bytes_per_s": 5_000_000.0,
        "retry_cooldown_s": 17.0,
        "plan_interval_s": 0.5,
        "allow_queue_overtaking": false
    });
    let cfg: DeploymentConfig = serde_json::from_value(value).unwrap();
    cfg.validate().unwrap();
    let execution = cfg.execution();
    assert_eq!(execution.scheduling.maximum_startup_s, 300.0);
    assert_eq!(execution.scheduling.retry_cooldown_s, 17.0);
    assert_eq!(execution.scheduling.plan_interval_s, 0.5);
    assert!(!execution.scheduling.allow_queue_overtaking);
    assert_eq!(
        execution.scheduling.learning_rate,
        Scheduling::default().learning_rate
    );
    assert_eq!(
        serde_json::to_value(&cfg.scheduling).unwrap(),
        serde_json::to_value(&execution.scheduling).unwrap()
    );
    let restored: ExecutionConfig =
        serde_json::from_value(serde_json::to_value(&execution).unwrap()).unwrap();
    restored.validate().unwrap();
    assert_eq!(
        serde_json::to_value(restored.scheduling).unwrap(),
        serde_json::to_value(execution.scheduling).unwrap()
    );
}

#[test]
fn omitted_scheduling_preserves_the_legacy_live_deadline_for_both_config_types() {
    for startup_timeout_s in [10, 900, 1800] {
        let mut cfg = config();
        cfg.limits.startup_timeout_s = startup_timeout_s;
        let mut deployment = serde_json::to_value(&cfg).unwrap();
        deployment.as_object_mut().unwrap().remove("scheduling");
        let legacy: DeploymentConfig = serde_json::from_value(deployment).unwrap();
        legacy.validate().unwrap();
        assert_eq!(
            legacy.scheduling.maximum_startup_s,
            startup_timeout_s as f64
        );
        assert_eq!(legacy.scheduling.minimum_hydration_bytes_per_s, 5_000_000.0);
        let mut execution = serde_json::to_value(cfg.execution()).unwrap();
        execution.as_object_mut().unwrap().remove("scheduling");
        let legacy_execution: ExecutionConfig = serde_json::from_value(execution).unwrap();
        legacy_execution.validate().unwrap();
        assert_eq!(
            serde_json::to_value(legacy.scheduling).unwrap(),
            serde_json::to_value(legacy_execution.scheduling).unwrap()
        );
    }
}

#[test]
fn live_configuration_reuses_generic_scheduling_validation() {
    let invalid = [
        Scheduling {
            plan_interval_s: 0.0,
            ..Scheduling::default()
        },
        Scheduling {
            maximum_startup_s: f64::NAN,
            ..Scheduling::default()
        },
        Scheduling {
            minimum_hydration_bytes_per_s: f64::INFINITY,
            ..Scheduling::default()
        },
        Scheduling {
            retry_cooldown_s: -1.0,
            ..Scheduling::default()
        },
        Scheduling {
            learning_rate: 0.0,
            ..Scheduling::default()
        },
        Scheduling {
            warm_batch_capacity_fraction: 1.1,
            ..Scheduling::default()
        },
        {
            let mut scheduling = Scheduling::default();
            scheduling.classes.get_mut("standard").unwrap().weight = 0.0;
            scheduling
        },
    ];
    for scheduling in invalid {
        let expected = validate_scheduling(&scheduling).unwrap_err();
        let mut cfg = config();
        cfg.scheduling = scheduling;
        assert!(
            matches!(cfg.validate(), Err(switchyard_live::Error::Config(message)) if message == expected)
        );
        assert!(
            matches!(cfg.execution().validate(), Err(switchyard_live::Error::Config(message)) if message == expected)
        );
    }
    let mut value = serde_json::to_value(config()).unwrap();
    value["scheduling"]["maximum_startp_s"] = 300.into();
    assert!(serde_json::from_value::<DeploymentConfig>(value).is_err());
}

#[test]
fn checked_diagnostic_fleet_retains_explicit_retry_policy() {
    let cfg: DeploymentConfig = serde_json::from_str(include_str!(
        "../../../configs/runpod/hydration-diagnostic-2b.json"
    ))
    .unwrap();
    cfg.validate().unwrap();
    assert_eq!(
        cfg.bootstrap
            .runtime
            .as_ref()
            .unwrap()
            .hydration
            .unwrap()
            .retries,
        0
    );
    let projected = cfg.execution();
    let restored: switchyard_live::config::ExecutionConfig =
        serde_json::from_value(serde_json::to_value(projected).unwrap()).unwrap();
    assert_eq!(
        restored
            .bootstrap
            .runtime
            .unwrap()
            .hydration
            .unwrap()
            .retries,
        0
    );
    let mut invalid = cfg;
    invalid
        .bootstrap
        .runtime
        .as_mut()
        .unwrap()
        .hydration
        .as_mut()
        .unwrap()
        .retries = 11;
    assert!(invalid.validate().is_err());
}
