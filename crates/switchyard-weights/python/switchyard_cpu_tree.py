"""CPU process-tree checkpoint at the pinned vLLM worker's device boundary.

The native API process constructs its configuration and renderer normally. Its
spawned engine worker imports the model class and prepares the native tokenizer,
then blocks before Worker.init_device. CRIU captures these live objects and their
internal control channels. Only the external launch control socket is remapped.
"""

import array
import importlib
import io
import json
import os
from pathlib import Path
import socket
import struct
import sys
import threading
import time

LIMIT = 65536
PREPARE_ADDRESS = "SWITCHYARD_CPU_PREPARE_ADDRESS"
_original_lazy_init = None
_nvml_originals = {}
_saved_stdio = None


def send(channel, value):
    data = json.dumps(value, separators=(",", ":")).encode()
    if not 0 < len(data) <= LIMIT:
        raise ValueError("CPU tree control frame exceeds limit")
    channel.sendall(struct.pack("!I", len(data)) + data)


def receive(channel):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            part = channel.recv(size - len(result))
            if not part:
                raise RuntimeError("CPU tree control disconnected")
            result.extend(part)
        return bytes(result)

    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= LIMIT:
        raise ValueError("CPU tree control frame exceeds limit")
    return json.loads(exact(size))


def cpu_state():
    import torch

    devices = []
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        if "nvidia" in target.lower() or target.startswith("/dev/dri/"):
            devices.append(target)
    if torch.cuda.is_initialized() or devices:
        raise RuntimeError(f"CPU tree accessed GPU state: {devices}")
    return {"cuda_initialized": False, "gpu_device_fds": devices}


def guard_cuda():
    global _original_lazy_init
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    # This serving profile uses baked binaries and eager execution. Importing
    # Inductor must not start unused compiler subprocesses in the CPU image.
    os.environ["TORCH_WARM_POOL"] = "0"
    os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
    import torch

    if _original_lazy_init is None:
        _original_lazy_init = torch.cuda._lazy_init

        def reject(*args, **kwargs):
            raise RuntimeError("CUDA initialization before the CPU tree checkpoint")

        torch.cuda._lazy_init = reject
        # Torch 2.13's raw NVML inventory functions call nvmlInit without a
        # matching shutdown. Their retained device descriptors cannot belong
        # to a CPU-only image. Balance these preparation-only inventory calls.
        import ctypes

        def balanced(function):
            def query(*args, **kwargs):
                result = function(*args, **kwargs)
                if result is not None and result != -1:
                    status = ctypes.CDLL("libnvidia-ml.so.1").nvmlShutdown()
                    if status != 0:
                        raise RuntimeError(
                            "cannot release CPU preparation NVML inventory"
                        )
                return result

            return query

        for name in ("_raw_device_count_nvml", "_raw_device_uuid_nvml"):
            _nvml_originals[name] = getattr(torch.cuda, name)
            setattr(torch.cuda, name, balanced(_nvml_originals[name]))


def detach_bake_stdio():
    import logging

    for logger in [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]:
        if isinstance(logger, logging.Logger):
            for handler in tuple(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
                    handler.close()
                    stream = logging.StreamHandler(sys.stderr)
                    stream.setFormatter(handler.formatter)
                    stream.setLevel(handler.level)
                    logger.addHandler(stream)
    descriptor = os.open("/dev/null", os.O_RDWR)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)


def stdio(channel):
    marker, ancillary, flags, _ = channel.recvmsg(
        1, socket.CMSG_SPACE(3 * array.array("i").itemsize)
    )
    descriptors = array.array("i")
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            descriptors.frombytes(data[: len(data) - len(data) % descriptors.itemsize])
    if marker != b"F" or flags & socket.MSG_CTRUNC or len(descriptors) != 3:
        raise RuntimeError("CPU tree needs three fresh stdio descriptors")
    return descriptors


def bind(descriptors, environment, calibration):
    global _saved_stdio
    _saved_stdio = (sys.stdin, sys.stdout, sys.stderr)
    # The agent owns the restorer, which owns the restored API process; that
    # process owns the engine worker. Preserve cancellation across both edges.
    import ctypes
    import signal

    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot bind restored process lifetime")
    for target, descriptor in enumerate(descriptors):
        os.dup2(descriptor, target)
    for descriptor in descriptors:
        if descriptor > 2:
            os.close(descriptor)
    sys.stdin = io.TextIOWrapper(io.FileIO(0, "r", closefd=False), encoding="utf-8")
    sys.stdout = io.TextIOWrapper(
        io.FileIO(1, "w", closefd=False), encoding="utf-8", write_through=True
    )
    sys.stderr = io.TextIOWrapper(
        io.FileIO(2, "w", closefd=False), encoding="utf-8", write_through=True
    )
    os.environ.clear()
    os.environ.update(environment)
    os.environ["SWITCHYARD_STARTUP_GATE_RELEASED"] = "1"
    os.environ["TORCH_WARM_POOL"] = "0"
    os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
    os.environ["VLLM_CPU_SNAPSHOT_CLOCK_OFFSET_S"] = str(calibration["offset_s"])
    os.environ["VLLM_CPU_SNAPSHOT_CLOCK_UNCERTAINTY_S"] = str(
        calibration["uncertainty_s"]
    )
    for directory in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if directory and directory not in sys.path:
            sys.path.insert(0, directory)
    import torch

    torch.cuda._lazy_init = _original_lazy_init
    for name, function in _nvml_originals.items():
        setattr(torch.cuda, name, function)
    # Instrument the already-imported modules, without replaying their imports.
    if os.environ.get("NATIVE_STARTUP_PROFILE"):
        spec = importlib.machinery.PathFinder.find_spec(
            "sitecustomize", os.environ.get("PYTHONPATH", "").split(os.pathsep)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    import vllm.envs as envs

    envs.disable_envs_cache()


def install_worker_boundary(worker_module):
    if not os.environ.get(PREPARE_ADDRESS):
        return
    guard_cuda()
    original = worker_module.Worker.init_device

    def checkpoint(worker):
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.tokenizers.registry import cached_tokenizer_from_config

        model_class, _ = ModelRegistry.resolve_model_cls(
            worker.model_config.architectures, worker.model_config
        )
        tokenizer = cached_tokenizer_from_config(worker.model_config)
        runner_module = (
            "vllm.v1.worker.gpu.model_runner"
            if worker.vllm_config.use_v2_model_runner
            else "vllm.v1.worker.gpu_model_runner"
        )
        importlib.import_module(runner_module)
        importlib.import_module("switchyard_chunk_loader")
        importlib.import_module("switchyard_final_state")
        # Native backend selection and kernel warmup import these lazily even
        # for the selected eager profile. Preserve their CPU module state too.
        importlib.import_module("vllm.v1.attention.backends.flashinfer")
        importlib.import_module("vllm.model_executor.warmup.minimax_m3_msa_warmup")
        # Include lazy CPU imports used by model and engine construction.
        import torch._dynamo  # noqa: F401

        channel = socket.socket(socket.AF_UNIX)
        channel.connect("\0" + os.environ[PREPARE_ADDRESS])
        objects = {
            "worker": id(worker),
            "config": id(worker.vllm_config),
            "tokenizer": id(tokenizer),
            "model_class": id(model_class),
        }
        safe = cpu_state()
        detach_bake_stdio()
        send(
            channel,
            {
                "event": "worker_cpu_prepared",
                "pid": os.getpid(),
                "objects": objects,
                "module_count": len(sys.modules),
                "model_configuration": {
                    "root_tied_embeddings": worker.model_config.hf_config.tie_word_embeddings,
                    "text_tied_embeddings": worker.model_config.hf_text_config.tie_word_embeddings,
                },
                **safe,
            },
        )
        descriptors = stdio(channel)
        request = receive(channel)
        if request["probe"]:
            os._exit(0)
        bind(descriptors, request["environment"], request["calibration"])
        after = {
            "worker": id(worker),
            "config": id(worker.vllm_config),
            "tokenizer": id(cached_tokenizer_from_config(worker.model_config)),
            "model_class": id(model_class),
        }
        if after != objects:
            raise RuntimeError("restored native worker/tokenizer/config objects differ")
        from switchyard_vllm_startup import Policy, record

        record(
            Policy.from_environment(),
            "worker_cpu_restored",
            objects=after,
            objects_preserved=True,
            module_count=len(sys.modules),
            **cpu_state(),
        )
        channel.close()
        policy = Policy.from_environment()
        began = time.monotonic()
        record(policy, "device_initialization_started")
        result = original(worker)
        record(
            policy,
            "device_initialization_finished",
            duration_s=time.monotonic() - began,
        )
        return result

    worker_module.Worker.init_device = checkpoint


def main(control, config):
    """Run the real native CLI until its prepared worker reaches the boundary."""
    import secrets

    serving = config["serving"]
    # This libuv version creates io_uring maps even with UV_USE_IO_URING=0;
    # CRIU 4.2.1 cannot checkpoint those mappings. Keep the native coroutine
    # on Python's selector event loop, which CRIU can preserve with epoll.
    import asyncio
    import uvloop

    uvloop.run = asyncio.run
    address = "switchyard-cpu-tree-" + secrets.token_hex(12)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind("\0" + address)
    listener.listen(1)
    os.environ[PREPARE_ADDRESS] = address
    guard_cuda()
    importlib.import_module("vllm.tokenizers.mistral")
    before = {"parent": os.getpid(), "sentinel": id(listener)}

    def restore_control():
        try:
            worker, _ = listener.accept()
            worker_receipt = receive(worker)
            safe = cpu_state()
            engine = None
            for frame in sys._current_frames().values():
                while frame:
                    instance = frame.f_locals.get("self")
                    if type(instance).__name__ == "AsyncLLM":
                        engine = instance
                        before.update(
                            {
                                "async_llm": id(instance),
                                "config": id(instance.vllm_config),
                                "renderer": id(instance.renderer),
                                "tokenizer": id(instance.renderer.tokenizer),
                            }
                        )
                        break
                    frame = frame.f_back
            if engine is None:
                raise RuntimeError("native AsyncLLM is absent at the CPU boundary")
            mapped = sorted(
                {
                    line.split(maxsplit=5)[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if len(line.split(maxsplit=5)) == 6
                    and line.split(maxsplit=5)[-1].startswith("/")
                }
            )
            detach_bake_stdio()
            send(
                control,
                {
                    "event": "cpu_prepared",
                    "pid": os.getpid(),
                    "objects": before,
                    "workers": [worker_receipt],
                    "mapped_files": mapped,
                    "vllm_import_complete": True,
                    "platform_resolved": True,
                    "boundary": "Worker.init_device",
                    "module_count": len(sys.modules),
                    **safe,
                },
            )
            descriptors = stdio(control)
            request = receive(control)
            if request["runtime_id"] != config["runtime_id"]:
                raise ValueError("CPU tree runtime identity differs")
            if not request["probe"] and request["args"] != serving["args"]:
                raise ValueError("CPU tree arguments differ from baked configuration")
            if not request["probe"]:
                for key in (
                    "VLLM_ENGINE_STARTUP_POLICY",
                    "SWITCHYARD_MODEL_CONTENT_ID",
                ):
                    if request["environment"].get(key) != serving["environment"].get(
                        key
                    ):
                        raise ValueError("CPU tree model or serving policy differs")
            after = {
                "parent": os.getpid(),
                "sentinel": id(listener),
                "async_llm": id(engine),
                "config": id(engine.vllm_config),
                "renderer": id(engine.renderer),
                "tokenizer": id(engine.renderer.tokenizer),
            }
            if after != before:
                raise RuntimeError(
                    "restored native API/config/tokenizer objects differ"
                )
            send(
                control,
                {
                    "event": "cpu_restored",
                    "pid": os.getpid(),
                    "objects": after,
                    "objects_preserved": True,
                    "workers": [worker_receipt],
                    "child_monotonic_s": time.monotonic(),
                    **cpu_state(),
                },
            )
            calibration = receive(control)
            rights = array.array("i", descriptors)
            worker.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)])
            send(worker, {**request, "calibration": calibration})
            bind(descriptors, request["environment"], calibration)
            control.close()
            worker.close()
            listener.close()
            if request["probe"]:
                print(
                    "SWITCHYARD_CPU_SNAPSHOT "
                    + json.dumps(
                        {
                            "event": "cpu_tree_probe_complete",
                            "workers": [worker_receipt],
                            "objects_preserved": True,
                        }
                    ),
                    flush=True,
                )
                os._exit(0)
            from switchyard_vllm_startup import Policy, record

            record(
                Policy.from_environment(),
                "parent_cpu_restored",
                module_count=len(sys.modules),
                **cpu_state(),
            )
        except BaseException as error:
            try:
                send(control, {"error": str(error)})
            finally:
                os._exit(1)

    threading.Thread(
        target=restore_control, name="cpu-tree-control", daemon=True
    ).start()
    # The native path installs the worker boundary through the general plugin.
    sys.argv = ["vllm", *serving["args"]]
    from switchyard_vllm_startup import main as native_main

    try:
        native_main()
    except BaseException as error:
        import traceback

        if control.fileno() >= 0:
            send(control, {"error": str(error), "traceback": traceback.format_exc()})
        raise
