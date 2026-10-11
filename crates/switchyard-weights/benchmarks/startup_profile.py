"""Opt-in native startup spans, loaded as sitecustomize in an owned benchmark.

Every wrapped operation still executes. Nested spans remain distinct; their
durations must not be added together. Arguments and error messages are omitted.
"""

import builtins
import functools
import importlib
import inspect
import json
import os
import sys
import threading
import time

OUTPUT = os.environ.get("NATIVE_STARTUP_PROFILE")
TARGETS = {
    "torch._C": ["_cuda_init"],
    "torch.accelerator": ["set_device_index"],
    "torch.cuda": ["init"],
    "vllm.engine.arg_utils": [
        "EngineArgs.create_engine_config",
        "EngineArgs.create_model_config",
    ],
    "vllm.model_executor.models.registry": [
        "_ModelRegistry._try_inspect_model_cls",
        "_run_in_subprocess",
    ],
    "vllm.tokenizers.registry": ["get_tokenizer", "cached_tokenizer_from_config"],
    "vllm.model_executor.model_loader.utils": [
        "initialize_model",
        "process_weights_after_loading",
    ],
    "vllm.v1.worker.gpu_worker": [
        "init_worker_distributed_environment",
        "Worker.__init__",
        "Worker.init_device",
        "Worker.load_model",
        "Worker.determine_available_memory",
        "Worker.initialize_from_config",
        "Worker.compile_or_warm_up_model",
    ],
    "vllm.v1.worker.gpu.model_runner": [
        "GPUModelRunner.load_model",
        "GPUModelRunner.profile_run",
        "GPUModelRunner.initialize_kv_cache",
    ],
    "vllm.v1.worker.gpu_model_runner": [
        "GPUModelRunner.load_model",
        "GPUModelRunner.profile_run",
        "GPUModelRunner.profile_cudagraph_memory",
        "GPUModelRunner.initialize_kv_cache",
        "GPUModelRunner.capture_model",
    ],
    "vllm.v1.worker.startup_plan": [
        "maybe_apply_startup_plan",
        "maybe_save_startup_plan",
    ],
    "vllm.v1.engine.async_llm": ["AsyncLLM.__init__", "AsyncLLM.from_vllm_config"],
    "vllm.entrypoints.launchers.api_server.app_state": ["init_app_state"],
    "vllm.entrypoints.launchers.app": ["build_app"],
    "vllm.entrypoints.launchers.launcher": ["setup_server"],
}
IMPORTS = {"torch", "transformers", "vllm", *TARGETS}
patched = set()
imported = set()
local = threading.local()


def record(event):
    event.update(pid=os.getpid(), thread=threading.get_ident())
    os.write(fd, (json.dumps(event, separators=(",", ":")) + "\n").encode())


def wrap(function, label):
    def begin():
        depth = getattr(local, "depth", 0)
        local.depth = depth + 1
        started = time.monotonic()
        record(
            {
                "kind": "operation_start",
                "name": label,
                "started_monotonic": started,
                "depth": depth,
            }
        )
        return started, depth

    def finish(started, depth, error=None):
        local.depth = depth
        record(
            {
                "kind": "operation",
                "name": label,
                "started_monotonic": started,
                "ended_monotonic": time.monotonic(),
                "depth": depth,
                "error_type": error,
            }
        )

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def asynchronous(*args, **kwargs):
            started, depth = begin()
            error = None
            try:
                return await function(*args, **kwargs)
            except BaseException as caught:
                error = type(caught).__name__
                raise
            finally:
                finish(started, depth, error)

        return asynchronous

    @functools.wraps(function)
    def synchronous(*args, **kwargs):
        started, depth = begin()
        error = None
        try:
            return function(*args, **kwargs)
        except BaseException as caught:
            error = type(caught).__name__
            raise
        finally:
            finish(started, depth, error)

    return synchronous


def patch(name):
    module = sys.modules.get(name)
    if name in patched or name not in TARGETS or module is None:
        return
    # Modules can appear in sys.modules while still executing a circular import.
    if getattr(module.__spec__, "_initializing", False):
        return
    patched.add(name)
    for target in TARGETS[name]:
        owner = module
        parts = target.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part, None)
            if owner is None:
                break
        if owner is None or not hasattr(owner, parts[-1]):
            record({"kind": "missing_target", "name": name + "." + target})
            continue
        descriptor = inspect.getattr_static(owner, parts[-1])
        if isinstance(descriptor, (classmethod, staticmethod)):
            function = wrap(descriptor.__func__, name + "." + target)
            setattr(owner, parts[-1], type(descriptor)(function))
        else:
            setattr(
                owner, parts[-1], wrap(getattr(owner, parts[-1]), name + "." + target)
            )


def importing(original, name, *args, **kwargs):
    started = time.monotonic() if name not in imported else None
    if started is not None:
        imported.add(name)
    try:
        return original(name, *args, **kwargs)
    finally:
        ended = time.monotonic()
        if started is not None and (name in IMPORTS or ended - started >= 0.005):
            record(
                {
                    "kind": "import",
                    "name": name,
                    "started_monotonic": started,
                    "ended_monotonic": ended,
                }
            )
        for target in TARGETS:
            patch(target)


if OUTPUT:
    fd = os.open(OUTPUT, os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC, 0o600)
    record({"kind": "process_start", "monotonic": time.monotonic()})
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    builtins.__import__ = lambda name, *args, **kwargs: importing(
        original_import, name, *args, **kwargs
    )
    importlib.import_module = lambda name, *args, **kwargs: importing(
        original_import_module, name, *args, **kwargs
    )
