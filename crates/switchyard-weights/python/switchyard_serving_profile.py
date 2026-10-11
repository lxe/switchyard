"""Fixed serving budgets validated against a baked model/runtime/GPU profile.

The selected path has no automatic sizing or dummy profiling forward. Device,
model, KV allocation and essential native kernel initialization still execute.
"""

import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def engine_factors(config):
    model, scheduler, parallel = (
        config.model_config,
        config.scheduler_config,
        config.parallel_config,
    )
    mm = model.multimodal_config
    return {
        "model_config_sha256": file_digest(Path(model.model) / "config.json"),
        "dtype": str(model.dtype),
        "max_model_len": model.max_model_len,
        "max_num_seqs": scheduler.max_num_seqs,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "pipeline_parallel_size": parallel.pipeline_parallel_size,
        "data_parallel_size": parallel.data_parallel_size,
        "quantization": model.quantization,
        "language_model_only": mm is None or mm.language_model_only,
        "compilation_mode": config.compilation_config.mode.name,
        "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
        "attention_backend": getattr(
            config.attention_config.backend,
            "name",
            str(config.attention_config.backend),
        ),
        "kv_cache_dtype": config.cache_config.cache_dtype,
        "mamba_cache_mode": config.cache_config.mamba_cache_mode,
        "block_size": config.cache_config.block_size,
        "enable_prefix_caching": config.cache_config.enable_prefix_caching,
    }


def check_profile(
    profile, *, runtime_id, model_content_id, engine, gpu, free_bytes, kv_bytes
):
    """Admission for an explicit allocation; no free-memory fraction is used."""
    if (
        profile.get("schema_version") != 1
        or profile.get("purpose") != "fixed_serving_budget"
    ):
        raise ValueError("unsupported serving profile")
    if (
        profile.get("runtime_id") != runtime_id
        or profile.get("model_content_id") != model_content_id
    ):
        raise ValueError("serving profile runtime or model identity mismatch")
    if profile.get("engine") != engine or profile.get("gpu") != gpu:
        raise ValueError("serving profile engine or GPU mismatch")
    expected = profile.get("kv_cache_memory_bytes")
    headroom = profile.get("runtime_headroom_bytes")
    if (
        type(expected) is not int
        or expected <= 0
        or type(headroom) is not int
        or headroom <= 0
    ):
        raise ValueError("serving profile needs explicit positive budgets")
    if kv_bytes != expected:
        raise ValueError("explicit KV allocation differs from serving profile")
    if free_bytes < expected + headroom:
        raise RuntimeError("GPU free memory is below the fixed serving budget")
    if not profile.get("bake", {}).get("native_warmup_completed"):
        raise ValueError("serving profile has no completed native warmup receipt")
    return expected


def install(policy, worker_module, record):
    if policy.serving_profile is None:
        return
    original = getattr(
        worker_module.Worker.determine_available_memory,
        "_switchyard_native",
        worker_module.Worker.determine_available_memory,
    )

    @functools.wraps(original)
    def fixed_budget(worker):
        from switchyard_vllm_hooks import require_gate

        require_gate()
        import torch

        cache_root = Path(os.environ["SWITCHYARD_CACHE_ROOT"])
        profile_path = cache_root / policy.serving_profile
        if not profile_path.resolve().is_relative_to(cache_root.resolve()):
            raise ValueError("serving profile escapes the selected cache")
        profile = json.loads(profile_path.read_text())
        final_state = json.loads(
            (
                Path(worker.model_config.model) / "switchyard-final-state.json"
            ).read_text()
        )
        if profile.get("final_layout_sha256") != final_state.get("layout_sha256"):
            raise ValueError("serving profile final tensor layout mismatch")
        if profile.get("versions") != {
            "vllm": importlib.metadata.version("vllm"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }:
            raise ValueError("serving profile library build mismatch")
        properties = torch.cuda.get_device_properties(worker.device)
        gpu = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
        free_bytes, _ = torch.cuda.mem_get_info(worker.device)
        kv_bytes = check_profile(
            profile,
            runtime_id=os.environ["SWITCHYARD_RUNTIME_ARTIFACT_ID"],
            model_content_id=os.environ["SWITCHYARD_MODEL_CONTENT_ID"],
            engine=engine_factors(worker.vllm_config),
            gpu=gpu,
            free_bytes=free_bytes,
            kv_bytes=worker.cache_config.kv_cache_memory_bytes,
        )
        if policy.cache_policy != "required":
            raise ValueError(
                "fixed serving profile requires native strict cache policy"
            )
        sampling = cache_root / profile["bake"]["sampling_binary"]
        if (
            not sampling.resolve().is_relative_to(cache_root.resolve())
            or file_digest(sampling) != profile["bake"]["sampling_sha256"]
        ):
            raise ValueError("serving profile sampling cache mismatch")
        worker.available_kv_cache_memory_bytes = kv_bytes
        from vllm.multimodal.gpu_ipc_memory import reserve_mm_ipc_gpu_memory

        reserved = reserve_mm_ipc_gpu_memory(
            kv_bytes,
            worker.model_config.multimodal_config,
            getattr(worker.parallel_config, "_api_process_count", 1),
        )
        record(
            policy,
            "fixed_serving_budget",
            kv_cache_memory_bytes=reserved,
            free_memory_bytes=free_bytes,
            runtime_headroom_bytes=profile["runtime_headroom_bytes"],
            profiling_forward_skipped=True,
            automatic_memory_sizing=False,
        )
        return reserved

    fixed_budget._switchyard_native = original
    worker_module.Worker.determine_available_memory = fixed_budget
