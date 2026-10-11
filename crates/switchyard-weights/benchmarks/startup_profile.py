"""Opt-in native startup spans, loaded as sitecustomize in an owned benchmark.

Every wrapped operation still executes. Nested spans remain distinct; their
durations must not be added together. Arguments and error messages are omitted.
"""

import builtins
import functools
import importlib
import importlib.abc
import inspect
import json
import os
import sys
import threading
import time
import traceback

OUTPUT = os.environ.get("NATIVE_STARTUP_PROFILE")
TARGETS = {
    "switchyard_final_state": ["validate_destinations"],
    "multiprocessing.process": ["BaseProcess.start"],
    "vllm.utils.system_utils": ["get_mp_context", "_maybe_force_spawn"],
    "torch._C": ["_cuda_init"],
    "torch.accelerator": ["set_device_index"],
    "torch.cuda": ["init", "_lazy_init"],
    "vllm.renderers": ["renderer_from_config"],
    "vllm.v1.engine.input_processor": ["InputProcessor.__init__"],
    "vllm.v1.engine.output_processor": ["OutputProcessor.__init__"],
    "vllm.v1.engine.utils": ["CoreEngineProcManager.__init__"],
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

# A benchmark-only source overlay tests a pinned patch before repacking a
# runtime. Both source hashes remain explicit in its receipt; this does not
# identify a new published artifact or a compatible baked cache.
if source_override := os.environ.get("NATIVE_ENV_OVERRIDE_SOURCE"):

    class EnvOverrideFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "vllm.env_override":
                return None
            sys.meta_path.remove(self)
            original = importlib.util.find_spec(fullname)
            if original is None or original.origin is None:
                raise RuntimeError("native env_override source is absent")
            return importlib.util.spec_from_file_location(fullname, source_override)

    sys.meta_path.insert(0, EnvOverrideFinder())

if cli_override := os.environ.get("NATIVE_CLI_MAIN_SOURCE"):

    class CliMainFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "vllm.entrypoints.cli.main":
                return None
            sys.meta_path.remove(self)
            original = importlib.util.find_spec(fullname)
            if original is None or original.origin is None:
                raise RuntimeError("native CLI source is absent")
            return importlib.util.spec_from_file_location(fullname, cli_override)

    sys.meta_path.insert(0, CliMainFinder())


if module_sources := os.environ.get("NATIVE_MODULE_SOURCES"):
    source_paths = json.loads(module_sources)

    class ModuleSourceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            source = source_paths.get(fullname)
            if source is None:
                return None
            return importlib.util.spec_from_file_location(fullname, source)

    sys.meta_path.insert(0, ModuleSourceFinder())


def record(event):
    offset = float(os.environ.get("VLLM_CPU_SNAPSHOT_CLOCK_OFFSET_S", "0"))
    for field in ("monotonic", "started_monotonic", "ended_monotonic"):
        if field in event:
            event[field] += offset
    if offset:
        event["clock_offset_s"] = offset
        event["clock_uncertainty_s"] = float(
            os.environ.get("VLLM_CPU_SNAPSHOT_CLOCK_UNCERTAINTY_S", "0")
        )
    event.update(pid=os.getpid(), thread=threading.get_ident())
    os.write(fd, (json.dumps(event, separators=(",", ":")) + "\n").encode())


def wrap(function, label):
    def begin():
        depth = getattr(local, "depth", 0)
        local.depth = depth + 1
        started = time.monotonic()
        record(
            {
                **(
                    {
                        "caller_stack": [
                            {"file": f.filename, "line": f.lineno, "function": f.name}
                            for f in traceback.extract_stack(limit=16)[:-1]
                        ]
                    }
                    if label == "torch._C._cuda_init"
                    else {}
                ),
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
            if (
                label == "switchyard_final_state.validate_destinations"
                and len(args) >= 2
            ):
                expected = args[0].get("layout", {})
                actual = {
                    name: {
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "stride": list(t.stride()),
                    }
                    for name, t in args[1].items()
                }
                differences = [
                    {
                        "name": name,
                        "expected": expected.get(name),
                        "actual": actual.get(name),
                    }
                    for name in sorted(expected.keys() | actual.keys())
                    if expected.get(name) != actual.get(name)
                ]
                record(
                    {
                        "kind": "tensor_layout_mismatch",
                        "expected_count": len(expected),
                        "actual_count": len(actual),
                        "difference_count": len(differences),
                        "differences": differences[:8],
                    }
                )
                frame = sys._getframe(1)
                while frame:
                    model_config = frame.f_locals.get("model_config")
                    if model_config is not None and hasattr(
                        model_config, "hf_text_config"
                    ):
                        record(
                            {
                                "kind": "model_config_at_layout_mismatch",
                                "text_tied_embeddings": model_config.hf_text_config.tie_word_embeddings,
                                "root_tied_embeddings": model_config.hf_config.tie_word_embeddings,
                            }
                        )
                        break
                    frame = frame.f_back
                record(
                    {
                        "kind": "embedding_alias_at_layout_mismatch",
                        "tensors": {
                            name: {
                                "pointer": tensor.data_ptr(),
                                "storage": tensor.untyped_storage().data_ptr(),
                            }
                            for name, tensor in args[1].items()
                            if name.endswith("embed_tokens.weight")
                            or name.endswith("lm_head.weight")
                        },
                    }
                )
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
            original = getattr(owner, parts[-1])
            wrapped = wrap(original, name + "." + target)
            setattr(owner, parts[-1], wrapped)
            # Restored modules already hold references from `from ... import`.
            # Rebind their module globals too; patching only the defining module
            # misses calls through those preserved native aliases.
            if len(parts) == 1 and name.startswith("vllm."):
                for loaded in tuple(sys.modules.values()):
                    if loaded is None or not getattr(loaded, "__name__", "").startswith(
                        "vllm."
                    ):
                        continue
                    for alias, value in tuple(vars(loaded).items()):
                        if value is original:
                            setattr(loaded, alias, wrapped)


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
