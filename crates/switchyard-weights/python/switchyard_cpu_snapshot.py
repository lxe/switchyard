"""Detached CPU preparation payload for the serving-layer CRIU adapter.

The snapshot owns no agent streams, model weights, platform selection, or CUDA
context. Restore binds a fresh control socket and receives fresh stdio/config.
"""
import array
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.machinery
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
MODULES = ("torch", "transformers", "tokenizers")
FORBIDDEN = ("vllm.platforms", "vllm.v1.worker", "vllm.entrypoints", "flashinfer",
             "flashinfer_cubin", "flashinfer_jit_cache", "cutlass", "nvidia_cutlass_dsl")


class CpuImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.pending = threading.Event()
        self.resume = threading.Event()
        self.pending_module = None

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN):
            # Torch probes optional backend availability with find_spec. Preserve
            # that inventory while refusing execution of the backend's code.
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                spec.loader = DeferredLoader(fullname, spec.loader, self)
            return spec


class DeferredLoader(importlib.abc.Loader):
    def __init__(self, fullname, native_loader, fence):
        self.fullname = fullname
        self.native_loader = native_loader
        self.fence = fence

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module._switchyard_cpu_deferred = True
        self.fence.pending_module = self.fullname
        self.fence.pending.set()
        self.fence.resume.wait()
        del module._switchyard_cpu_deferred
        self.native_loader.exec_module(module)


def send(control, value):
    data = json.dumps(value, separators=(",", ":")).encode()
    if len(data) > LIMIT:
        raise ValueError("snapshot control message exceeds limit")
    control.sendall(struct.pack("!I", len(data)) + data)


def receive(control):
    def exact(size):
        value = bytearray()
        while len(value) < size:
            part = control.recv(size - len(value))
            if not part:
                raise RuntimeError("snapshot restore control disconnected")
            value.extend(part)
        return bytes(value)
    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= LIMIT:
        raise ValueError("snapshot restore message exceeds limit")
    return json.loads(exact(size))


def assert_cpu(torch):
    if torch.cuda.is_initialized():
        raise RuntimeError("snapshot has initialized CUDA")
    if any((name == prefix or name.startswith(prefix + "."))
           and not getattr(module, "_switchyard_cpu_deferred", False)
           for name, module in tuple(sys.modules.items()) for prefix in FORBIDDEN):
        raise RuntimeError("snapshot imported platform/provider state")
    fds = []
    for path in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue
        if "nvidia" in target.lower() or target.startswith("/dev/dri/"):
            fds.append(target)
    driver_maps = [line for line in Path("/proc/self/maps").read_text().splitlines()
                   if "/libcuda.so" in line or "/libnvidia-ml.so" in line]
    if fds:
        raise RuntimeError(f"snapshot accessed GPU devices: fds={fds}")
    # CUDA-enabled Torch loads cuBLAS Lt, whose ELF initializer dlopens libcuda.
    # A mapped driver ELF is not a context. Bind its bytes in the base-file
    # compatibility inventory while rejecting CUDA initialization/device fds.
    libraries = sorted({line.split()[-1] for line in driver_maps})
    return {"cuda_initialized": False, "gpu_device_fds": fds,
            "gpu_driver_mappings": libraries, "platform_resolved": False}


def main():
    control = socket.socket(fileno=int(sys.argv[1]))
    config = json.loads(Path(sys.argv[2]).read_text())
    fence = CpuImports()
    sys.meta_path.insert(0, fence)
    modules = {}
    timings = {}
    original_lazy_init = None
    for name in MODULES:
        began = time.monotonic()
        modules[name] = importlib.import_module(name)
        timings[name] = time.monotonic() - began
        if name == "torch":
            original_lazy_init = modules[name].cuda._lazy_init
            def reject_cuda():
                raise RuntimeError("CUDA initialization is forbidden before CPU checkpoint")
            modules[name].cuda._lazy_init = reject_cuda
        try:
            assert_cpu(modules["torch"])
        except RuntimeError as error:
            raise RuntimeError(f"after {name}: {error}") from error
    torch = modules["torch"]
    import_errors = []
    def prepare_vllm():
        try:
            importlib.import_module("vllm")
        except BaseException as error:
            import_errors.append(error)
            fence.pending.set()
    vllm_thread = threading.Thread(target=prepare_vllm, name="cpu-snapshot-vllm", daemon=True)
    began = time.monotonic()
    vllm_thread.start()
    if not fence.pending.wait(30) or import_errors:
        raise RuntimeError("vLLM import did not reach its CPU checkpoint boundary") from (import_errors[0] if import_errors else None)
    modules["vllm"] = sys.modules["vllm"]
    timings["vllm_before_platform"] = time.monotonic() - began
    tokenizer = None
    tokenizer_sha = None
    tokens = None
    began = time.monotonic()
    if config.get("tokenizer"):
        path = Path(config["tokenizer"])
        tokenizer_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        tokenizer = modules["tokenizers"].Tokenizer.from_file(str(path))
        tokens = tokenizer.encode("Switchyard CPU checkpoint retains tokenizer state.").ids
    timings["tokenizer_prepare"] = time.monotonic() - began
    sentinel = object()
    before = {"modules": {name: id(module) for name, module in modules.items()},
              "tokenizer": id(tokenizer), "sentinel": id(sentinel)}
    versions = {name: importlib.metadata.version(name) for name in modules}
    safe = assert_cpu(torch)
    mapped = sorted({line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if len(line.split(maxsplit=5)) == 6 and line.split(maxsplit=5)[-1].startswith("/")})
    send(control, {"event": "cpu_prepared", "pid": os.getpid(), "objects": before,
                   "versions": versions, "tokenizer_sha256": tokenizer_sha,
                   "tokenizer_tokens": tokens, "timings": timings, "mapped_files": mapped,
                   "deferred_import": fence.pending_module, "vllm_import_complete": False, **safe})
    # CRIU snapshots this blocked read. The original preparer is killed by dump;
    # restoration supplies a fresh externally remapped Unix socket endpoint.
    marker, ancillary, flags, _ = control.recvmsg(1, socket.CMSG_SPACE(3 * array.array("i").itemsize))
    if marker != b"F" or flags & socket.MSG_CTRUNC:
        raise RuntimeError("restore stdio binding is absent or truncated")
    descriptors = array.array("i")
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            descriptors.frombytes(data[:len(data) - len(data) % descriptors.itemsize])
    if len(descriptors) != 3:
        raise RuntimeError("restore must bind exactly stdin/stdout/stderr")
    request = receive(control)
    resumed_at = time.monotonic()
    if set(request) != {"schema_version", "environment", "args", "probe", "runtime_id"} or request["schema_version"] != 1:
        raise ValueError("invalid fresh restore configuration")
    if request["runtime_id"] != config["runtime_id"]:
        raise ValueError("restored runtime identity differs")
    safe = assert_cpu(torch)
    after = {"modules": {name: id(sys.modules[name]) for name in modules},
             "tokenizer": id(tokenizer), "sentinel": id(sentinel)}
    if before != after:
        raise RuntimeError("CPU preparation objects were reconstructed")
    if tokenizer is not None and tokenizer.encode("Switchyard CPU checkpoint retains tokenizer state.").ids != tokens:
        raise RuntimeError("restored tokenizer output differs")
    send(control, {"event": "cpu_restored", "child_monotonic_s": resumed_at, "pid": os.getpid(), "objects": after,
                   "objects_preserved": True, "tokenizer_sha256": tokenizer_sha,
                   "tokenizer_tokens": tokens, **safe})
    calibration = receive(control)
    if set(calibration) != {"event", "offset_s", "uncertainty_s"} or calibration["event"] != "clock_calibration":
        raise ValueError("invalid restore clock calibration")
    for destination, descriptor in enumerate(descriptors):
        os.dup2(descriptor, destination)
    for descriptor in descriptors:
        if descriptor > 2:
            os.close(descriptor)
    sys.stdin = io.TextIOWrapper(io.FileIO(0, "r", closefd=False), encoding="utf-8")
    sys.stdout = io.TextIOWrapper(io.FileIO(1, "w", closefd=False), encoding="utf-8", write_through=True)
    sys.stderr = io.TextIOWrapper(io.FileIO(2, "w", closefd=False), encoding="utf-8", write_through=True)
    os.environ.clear()
    os.environ.update(request["environment"])
    # CRIU can restore a time namespace whose monotonic clock excludes the time
    # spent as images. Correct telemetry timestamps explicitly, never global time.
    os.environ["VLLM_CPU_SNAPSHOT_CLOCK_OFFSET_S"] = str(calibration["offset_s"])
    os.environ["VLLM_CPU_SNAPSHOT_CLOCK_UNCERTAINTY_S"] = str(calibration["uncertainty_s"])
    for directory in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if directory and directory not in sys.path:
            sys.path.insert(0, directory)
    torch.cuda._lazy_init = original_lazy_init
    sys.meta_path.remove(fence)
    control.close()
    if request["probe"]:
        print("SWITCHYARD_CPU_SNAPSHOT " + json.dumps({"event": "cpu_probe_complete", "monotonic_s": time.monotonic() + calibration["offset_s"], "clock_uncertainty_s": calibration["uncertainty_s"], "objects_preserved": True, **safe}), flush=True)
        os._exit(0)
    # The existing native adapter performs policy selection, provider/worker
    # imports, normal CUDA guards and normal model/weight construction after gate.
    from switchyard_vllm_startup import Policy, configure, main as native_startup
    policy = Policy.from_environment()
    if policy is not None:
        configure(policy)
    fence.resume.set()
    vllm_thread.join()
    if import_errors:
        raise import_errors[0]
    sys.argv = ["vllm", *request["args"]]
    native_startup()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        try:
            send(socket.socket(fileno=int(sys.argv[1])), {"error": str(error)})
        except BaseException:
            pass
        raise
