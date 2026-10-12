use switchyard_core::{Scheduling, validate_scheduling};
use switchyard_live::config::*;
use switchyard_provider::{ComputeConfig, StorageConfig};

#[test]
fn legacy_engine_fingerprint_remains_compatible_and_sglang_requires_its_runtime() {
    let cfg: DeploymentConfig =
        serde_json::from_str(include_str!("../../../configs/runpod/qwen-small.json")).unwrap();
    let engine = &cfg.models[0].engine;
    assert_eq!(
        switchyard_live::artifacts::engine_fingerprint(engine).unwrap(),
        "55cfe37edf43491aba80ecdc8c7aaf5f68624865146baa39779d2620248e886a"
    );
    let mut changed = cfg.clone();
    changed.models[0].engine.kind = EngineKind::Sglang;
    assert!(changed.validate().is_err());
    assert_ne!(
        switchyard_live::artifacts::engine_fingerprint(&changed.models[0].engine).unwrap(),
        switchyard_live::artifacts::engine_fingerprint(engine).unwrap()
    );
    let args = changed.models[0].engine.args(&changed.models[0], "/model");
    assert_eq!(&args[..2], ["--model-path", "/model"]);
    assert!(args.contains(&"--context-length".into()));
    assert!(args.windows(2).any(|pair| pair
        == [
            "--json-model-override-args",
            "{\"language_model_only\":true}"
        ]));
    assert!(!args.contains(&"--language-model-only".into()));
    assert!(!args.contains(&"serve".into()));
}
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
            native_cache_registry: None,
        },
        storage: storage("https://objects.example.test"),
        models: vec![ModelConfig {
            id: "qwen-small".into(),
            repository: "Qwen/Qwen3.5-0.8B".into(),
            revision: "a".repeat(40),
            artifact: switchyard_live::model::ModelArtifact::Legacy(
                switchyard_live::model::LegacyReference {
                    archive_prefix: "models/qwen/revision/s1".into(),
                    archive_sha256: "a".repeat(64),
                },
            ),
            caches: vec![],
            engine: EngineConfig {
                kind: Default::default(),
                native: None,
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
        "--hos=0.0.0.0",
        "--por=1234",
        "--served-model-n=other",
        "--context-l=1",
        "--mem-fraction-s=0.99",
        "--max-running-r=999",
        "--model-p=/other",
        "--dtyp=float32",
        "--device-memory-utilization=0.99",
        "--gpu_memory_utilization=0.99",
        "--max_model_len=1",
        "--no-language-model-only",
    ] {
        let mut bad = config();
        bad.models[0].engine.extra_args.push(flag.into());
        assert!(bad.validate().is_err(), "{flag}");
    }
    let mut valid = config();
    valid.models[0].engine.extra_args.extend([
        "--max-num-batched-tokens".into(),
        "4096".into(),
        "--disable-log-requests".into(),
    ]);
    valid.validate().unwrap();
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
        components: vec![],
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
        components: vec![],
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
        {
            let mut s = Scheduling::default();
            s.probe.minimum_received_bytes = 0;
            s
        },
        {
            let mut s = Scheduling::default();
            s.probe.minimum_window_s = f64::NAN;
            s
        },
        {
            let mut s = Scheduling::default();
            s.probe.maximum_window_s = 1.0;
            s
        },
        {
            let mut s = Scheduling::default();
            s.probe.reprobe_interval_s = 0.0;
            s
        },
        {
            let mut s = Scheduling::default();
            s.probe.maximum_age_s = f64::INFINITY;
            s
        },
        {
            let mut s = Scheduling::default();
            s.probe.reprobe_interval_s = 11.0;
            s
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

#[test]
fn extent_model_configuration_preserves_a_distinct_reference_and_cache_identity() {
    let legacy = config();
    let mut wire = serde_json::to_value(&legacy).unwrap();
    let model = wire["models"][0].as_object_mut().unwrap();
    model.remove("archive_prefix");
    model.remove("archive_sha256");
    model.insert("model_artifact".into(),serde_json::json!({"kind":"extents","prefix":"switchyard/weights/layout","id":"b".repeat(64),"content_id":"c".repeat(64),"loader":{"cuda_stage":true,"pinned_bytes":1048576}}));
    let mut decoded: DeploymentConfig = serde_json::from_value(wire.clone()).unwrap();
    decoded.models[0].artifact.validate().unwrap();
    let labels = decoded.models[0].artifact.cache_labels().unwrap();
    assert_eq!(labels["model_content_id"], "c".repeat(64));
    assert!(labels.contains_key("weight_loader"));
    assert!(!labels.contains_key("model_archive_sha256"));
    decoded.validate().unwrap();
    decoded.models[0].engine.kind = EngineKind::Sglang;
    assert!(
        decoded
            .validate()
            .unwrap_err()
            .to_string()
            .contains("extent weight loader requires vLLM")
    );
    wire["models"][0]["archive_sha256"] = serde_json::Value::String("a".repeat(64));
    assert!(serde_json::from_value::<DeploymentConfig>(wire).is_err());
}

#[test]
fn runtime_component_identity_changes_cache_fingerprint_and_empty_layout_is_compatible() {
    use switchyard_live::artifacts::{
        ArtifactRef, ServingRuntimeComponent, runtime_launch_fingerprint,
    };
    let cfg: DeploymentConfig = serde_json::from_str(include_str!(
        "../../../configs/runpod/qwen-extents-l40s-strict.json"
    ))
    .unwrap();
    let mut runtime = cfg.bootstrap.runtime.unwrap();
    let original = runtime_launch_fingerprint(&runtime).unwrap();
    let legacy = (&runtime.program, &runtime.args, &runtime.environment);
    use sha2::{Digest, Sha256};
    assert_eq!(
        original,
        format!("{:x}", Sha256::digest(serde_json::to_vec(&legacy).unwrap()))
    );
    runtime.components.push(ServingRuntimeComponent {
        name: "build".into(),
        artifact: ArtifactRef {
            prefix: "runtime/build".into(),
            id: "a".repeat(64),
        },
        compatibility: Default::default(),
    });
    runtime.validate().unwrap();
    let composite = runtime_launch_fingerprint(&runtime).unwrap();
    assert_ne!(original, composite);
    runtime.components[0].artifact.id = "b".repeat(64);
    assert_ne!(composite, runtime_launch_fingerprint(&runtime).unwrap());
    runtime.components.push(runtime.components[0].clone());
    assert!(runtime.validate().is_err());
}

#[test]
fn runtime_dependency_name_conflicts_are_rejected_before_provisioning() {
    use switchyard_live::artifacts::{ArtifactRef, ServingRuntimeComponent};
    let mut cfg: DeploymentConfig = serde_json::from_str(include_str!(
        "../../../configs/runpod/qwen-extents-l40s-strict.json"
    ))
    .unwrap();
    cfg.bootstrap
        .runtime
        .as_mut()
        .unwrap()
        .components
        .push(ServingRuntimeComponent {
            name: cfg.models[0].caches[0].name.clone(),
            artifact: ArtifactRef {
                prefix: "runtime/build".into(),
                id: "a".repeat(64),
            },
            compatibility: Default::default(),
        });
    assert!(
        cfg.validate()
            .unwrap_err()
            .to_string()
            .contains("component and cache names")
    );
}

#[test]
fn native_launch_keeps_fixed_bounds_and_requires_pinned_runtime() {
    let mut cfg = config();
    let engine = &mut cfg.models[0].engine;
    engine.kind = EngineKind::Native;
    engine.gpu_memory_utilization = 0.;
    engine.dtype = "bf16".into();
    engine.extra_args.clear();
    engine.native = Some(NativeEngineConfig {
        cache: None,
        max_weight_bytes: 4 << 30,
        minimum_gpu_memory_bytes: 8 << 30,
        queue_capacity: 64,
        batch_wait_ms: 2,
        cuda_ordinal: 0,
    });
    assert!(cfg.validate().is_err());
    cfg.bootstrap.runtime = Some(switchyard_live::artifacts::ServingRuntimeConfig {
        hydration: None,
        artifact: switchyard_live::artifacts::ArtifactRef {
            prefix: "runtime/native".into(),
            id: "b".repeat(64),
        },
        components: vec![],
        compatibility: [("engine_sha256".into(), "c".repeat(64))].into(),
        program: "bin/launch-engine.sh".into(),
        args: vec![],
        environment: Default::default(),
    });
    cfg.validate().unwrap();
    let model = &cfg.models[0];
    let args = model.engine.args(model, "/published/model");
    assert_eq!(args[0], "--config-json");
    let launch: serde_json::Value = serde_json::from_str(&args[1]).unwrap();
    assert_eq!(launch["model_root"], "/published/model");
    assert_eq!(launch["max_weight_bytes"], 4_u64 << 30);
    assert_eq!(launch["max_context"], 4096);
    assert_eq!(launch["max_batch_size"], 32);
    assert_eq!(&args[2..], ["serve", "--bind", "127.0.0.1:8000"]);
    let encoded = serde_json::to_value(&model.engine).unwrap();
    assert!(encoded.get("gpu_memory_utilization").is_none());
    let mut cached = cfg.clone();
    cached.models[0].engine.native.as_mut().unwrap().cache = Some(NativeCacheConfig {
        page_tokens: 16,
        checkpoint_tokens: 32,
        max_bytes: 268435456,
        max_entries: 16,
    });
    cached.validate().unwrap();
    let args = cached.models[0]
        .engine
        .args(&cached.models[0], "/published/model");
    let launch: serde_json::Value = serde_json::from_str(&args[1]).unwrap();
    assert_eq!(launch["cache"]["max_bytes"], 268435456);
    assert_ne!(
        switchyard_live::artifacts::engine_fingerprint(&cfg.models[0].engine).unwrap(),
        switchyard_live::artifacts::engine_fingerprint(&cached.models[0].engine).unwrap()
    );
    cached.models[0]
        .engine
        .native
        .as_mut()
        .unwrap()
        .cache
        .as_mut()
        .unwrap()
        .page_tokens = 0;
    assert!(cached.validate().is_err());
    for mutation in 0..5 {
        let mut invalid = cfg.clone();
        let engine = &mut invalid.models[0].engine;
        match mutation {
            0 => engine.native = None,
            1 => engine.gpu_memory_utilization = 0.8,
            2 => engine.extra_args.push("--config-json={}".into()),
            3 => engine.native.as_mut().unwrap().queue_capacity = 1,
            _ => engine.dtype = "auto".into(),
        }
        assert!(invalid.validate().is_err());
    }
    cfg.models[0].engine.kind = EngineKind::Vllm;
    assert!(cfg.validate().is_err());
}
