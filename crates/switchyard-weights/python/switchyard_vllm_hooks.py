"""Pinned native vLLM hooks for tokenizer reuse and startup observations.

Native configuration, cache keys, memory guards, arguments and results remain
authoritative. Hooks install independently in spawned workers.
"""

import functools
import importlib
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

from switchyard_vllm_startup import Policy, record

_installed_pid = None
_tokenizer_lock = threading.RLock()
_preloads = []


def require_gate():
    if os.environ.get("SWITCHYARD_STARTUP_GATE_RELEASED") != "1":
        raise RuntimeError("engine operation before metadata and dependency readiness")


def native_original(function):
    return getattr(function, "_switchyard_native", function)


def tokenizer_cache(registry):
    """Serialize native LRU misses without reconstructing or weakening keys."""
    original = native_original(registry.cached_get_tokenizer)

    @functools.wraps(original)
    def cached(*args, **kwargs):
        with _tokenizer_lock:
            return original(*args, **kwargs)

    cached._switchyard_native = original
    for name in ["cache_clear", "cache_info", "cache_parameters"]:
        if hasattr(original, name):
            setattr(cached, name, getattr(original, name))
    registry.cached_get_tokenizer = cached


class TokenizerPreparation:
    def __init__(self, policy, model_config, registry):
        self.policy = policy
        self.model_config = model_config
        self.registry = registry
        self.error = None
        self.thread = None

    def run(self):
        began = time.monotonic()
        try:
            require_gate()
            self.registry.cached_tokenizer_from_config(self.model_config)
            record(
                self.policy, "tokenizer_prepared", duration_s=time.monotonic() - began
            )
        # Return every thread exit to the joining process, including SystemExit.
        except BaseException as error:  # noqa: BLE001
            self.error = error
            record(
                self.policy,
                "tokenizer_preparation_failed",
                error_type=type(error).__name__,
            )

    def start(self):
        if self.policy.worker_method == "guarded_fork":
            self.run()
            self.join()
        else:
            self.thread = threading.Thread(
                target=self.run, name="switchyard-tokenizer", daemon=True
            )
            self.thread.start()

    def join(self):
        if self.thread is not None:
            self.thread.join()
        if self.error is not None:
            raise self.error


def model_preparation(policy, engine_args, registry):
    original = native_original(engine_args.create_model_config)
    original_engine = native_original(engine_args.create_engine_config)

    @functools.wraps(original)
    def create(self, *args, **kwargs):
        require_gate()
        result = original(self, *args, **kwargs)
        if policy.worker_method == "guarded_fork":
            # Load the selected native class once in the clean API process.
            # The child then inherits its module; no model is instantiated.
            from vllm.model_executor.models.registry import ModelRegistry

            began = time.monotonic()
            cls, _ = ModelRegistry.resolve_model_cls(result.architectures, result)
            record(
                policy,
                "model_class_prepared",
                module=cls.__module__,
                duration_s=time.monotonic() - began,
            )
        if policy.tokenizer_preload and not result.skip_tokenizer_init:
            preload = TokenizerPreparation(policy, result, registry)
            _preloads.append(preload)
            preload.start()
        return result

    create._switchyard_native = original
    engine_args.create_model_config = create

    @functools.wraps(original_engine)
    def engine(self, *args, **kwargs):
        result = original_engine(self, *args, **kwargs)
        join_preloads()
        compiler_cache(policy, result)
        return result

    engine._switchyard_native = original_engine
    engine_args.create_engine_config = engine


def compiler_cache(policy, config):
    """Check AOT only when the native configuration enables compilation."""
    compilation = getattr(config, "compilation_config", None)
    if policy.cache_policy != "required":
        record(
            policy,
            "native_compiler_cache",
            cache_policy="native",
            mode=getattr(getattr(compilation, "mode", None), "name", None),
        )
        return
    require_gate()
    from vllm.config.compilation import CompilationMode

    applicable = compilation.mode != CompilationMode.NONE
    cache = (
        Path(os.environ["VLLM_CACHE_ROOT"]) / "torch_compile_cache/torch_aot_compile"
    )
    available = cache.is_dir() and any(cache.iterdir())
    record(
        policy,
        "native_compiler_cache",
        cache_policy="required",
        applicable=applicable,
        mode=compilation.mode.name,
        available=available,
    )
    if applicable and not available:
        raise RuntimeError("required native Torch AOT cache is absent")


def process_method(policy, system):
    original = native_original(system._maybe_force_spawn)

    @functools.wraps(original)
    def native_guard(*args, **kwargs):
        require_gate()
        # Our tokenizer work must not retain an import/cache lock across fork.
        # This does not replace any of vLLM's device or environment guards.
        if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "fork":
            for preload in _preloads:
                preload.join()
        before = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        result = original(*args, **kwargs)
        import torch

        record(
            policy,
            "native_worker_method",
            requested=before,
            selected=os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
            cuda_initialized=torch.cuda.is_initialized(),
        )
        return result

    native_guard._switchyard_native = original
    system._maybe_force_spawn = native_guard


def memory_plan(policy, module):
    original = native_original(module.maybe_apply_startup_plan)

    @functools.wraps(original)
    def apply(worker, *args, **kwargs):
        require_gate()
        before = worker.cache_config.kv_cache_memory_bytes
        result = original(worker, *args, **kwargs)
        after = worker.cache_config.kv_cache_memory_bytes
        accepted = before is None and after is not None
        record(
            policy,
            "native_memory_plan",
            applied=accepted,
            explicit_memory_limit=before is not None,
            free_memory_bytes=worker.init_snapshot.free_memory,
        )
        if policy.memory_plan == "required" and not accepted:
            raise RuntimeError("required memory plan was not accepted by native guards")
        return result

    apply._switchyard_native = original
    module.maybe_apply_startup_plan = apply
    # The prepared API may already have imported the worker's function alias.
    worker_module = sys.modules.get("vllm.v1.worker.gpu_worker")
    if worker_module is not None:
        worker_module.maybe_apply_startup_plan = apply


def install(policy=None):
    global _installed_pid, _tokenizer_lock, _preloads
    policy = policy or Policy.from_environment()
    if policy is None or _installed_pid == os.getpid():
        return
    require_gate()
    # A fork inherits the completed cache but gets a new, unlocked coordinator.
    _tokenizer_lock = threading.RLock()
    _preloads = []
    registry = importlib.import_module("vllm.tokenizers.registry")
    tokenizer_cache(registry)
    engine_args = importlib.import_module("vllm.engine.arg_utils").EngineArgs
    model_preparation(policy, engine_args, registry)
    process_method(policy, importlib.import_module("vllm.utils.system_utils"))
    memory_plan(policy, importlib.import_module("vllm.v1.worker.startup_plan"))
    _installed_pid = os.getpid()
    record(policy, "native_hooks_ready")


def worker_start():
    policy = Policy.from_environment()
    if policy is None:
        return
    install(policy)
    import torch

    method = multiprocessing.get_start_method(allow_none=True)
    record(
        policy,
        "engine_worker_start",
        actual_start_method=method,
        cuda_initialized=torch.cuda.is_initialized(),
        prepared_parent_pid=os.environ.get("SWITCHYARD_PREPARED_PID"),
        fork_inheritance=method == "fork"
        and os.environ.get("SWITCHYARD_PREPARED_PID") is not None,
        prepared_modules_present=[
            name
            for name in ["torch", "transformers", "vllm.v1.worker.gpu_worker"]
            if name in sys.modules
        ],
    )


def join_preloads():
    for preload in _preloads:
        preload.join()
