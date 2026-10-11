"""Prepared vLLM process with one agent-owned prerequisite gate.

This adapter does not construct a model or change native CUDA/NUMA guards.
Runtime and cache compatibility belong to the selected, fingerprinted ticket.
"""

import importlib
import importlib.abc
import importlib.metadata
import json
import os
import re
import selectors
import sys
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path

GATE = "switchyard-metadata-ready-v1\n"
NVML_PATCH = "5b935fc5f4eacaff29876d2c9304df0b54ebe69828c6c3da8916ba36e54135f5"
NVML_RESULT = "3f19edd2c7aba0cb6a3251715199c3fe82ad1b20b7946b24b192d23bec82b828"
NVML_TARGET = "venv/lib/python3.12/site-packages/vllm/third_party/flash_linear_attention/ops/utils.py"
PREPARE_MODULES = (
    "torch",
    "transformers",
    "vllm.entrypoints.cli.serve",
    "vllm.v1.worker.gpu_worker",
)
PREREQUISITE_MODULES = {
    "flashinfer",
    "flashinfer_cubin",
    "flashinfer_jit_cache",
    "cutlass",
    "nvidia_cutlass_dsl",
}


@dataclass(frozen=True)
class Policy:
    schema_version: int = 1
    strategy: str = "native"
    worker_method: str = "spawn"
    fork_runtime_id: str | None = None
    compute_capability: str | None = None
    tokenizer_preload: bool = False
    cache_policy: str = "native"
    memory_plan: str = "native"
    serving_profile: str | None = None
    cache_aliases: tuple = ()
    telemetry: bool = True

    @classmethod
    def from_environment(cls):
        raw = os.environ.get("VLLM_ENGINE_STARTUP_POLICY")
        if raw is None:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) - {
            field.name for field in fields(cls)
        }:
            raise ValueError("unknown vLLM startup policy field")
        policy = cls(**value)
        if type(policy.schema_version) is not int or policy.schema_version != 1:
            raise ValueError("unsupported vLLM startup policy schema")
        if policy.strategy not in {"native", "prepared"}:
            raise ValueError("unsupported vLLM startup strategy")
        if policy.worker_method not in {"spawn", "guarded_fork"}:
            raise ValueError(
                "vLLM 0.31 supports native spawn or guarded fork, not forkserver"
            )
        if policy.cache_policy not in {
            "native",
            "required",
        } or policy.memory_plan not in {"native", "required"}:
            raise ValueError("unsupported native cache or memory-plan policy")
        if (
            type(policy.tokenizer_preload) is not bool
            or type(policy.telemetry) is not bool
        ):
            raise ValueError("vLLM startup boolean policy field")
        if (
            not isinstance(policy.cache_aliases, (list, tuple))
            or len(policy.cache_aliases) > 8
            or any(not isinstance(path, str) for path in policy.cache_aliases)
        ):
            raise ValueError("vLLM startup cache alias list")
        if policy.compute_capability is not None and (
            not isinstance(policy.compute_capability, str)
            or re.fullmatch(r"[1-9][0-9]?\.[0-9]", policy.compute_capability) is None
        ):
            raise ValueError("invalid selected GPU compute capability")
        if policy.worker_method == "guarded_fork" and (
            policy.strategy != "prepared"
            or not isinstance(policy.fork_runtime_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", policy.fork_runtime_id) is None
            or policy.compute_capability is None
        ):
            raise ValueError(
                "guarded fork requires prepared strategy, runtime pin, and GPU capability"
            )
        if policy.cache_policy == "required" and policy.compute_capability is None:
            raise ValueError("required cache policy needs the selected GPU capability")
        if policy.serving_profile is not None:
            profile = Path(policy.serving_profile)
            if (
                not isinstance(policy.serving_profile, str)
                or profile.is_absolute()
                or ".." in profile.parts
                or str(profile) != policy.serving_profile
                or len(profile.parts) < 2
                or policy.cache_policy != "required"
            ):
                raise ValueError(
                    "serving profile must be a normalized selected cache file with strict cache policy"
                )
        return policy


def record(policy, event, **values):
    if policy is None or not policy.telemetry:
        return
    raw_now = time.monotonic()
    offset = float(os.environ.get("VLLM_CPU_SNAPSHOT_CLOCK_OFFSET_S", "0"))
    now = raw_now + offset
    epoch = float(os.environ.get("SWITCHYARD_STARTUP_EPOCH", now))
    message = {
        "event": event,
        "pid": os.getpid(),
        "monotonic_s": now,
        "elapsed_s": now - epoch,
        "clock_offset_s": offset,
        "clock_uncertainty_s": float(
            os.environ.get("VLLM_CPU_SNAPSHOT_CLOCK_UNCERTAINTY_S", "0")
        ),
        **values,
    }
    print(
        "SWITCHYARD_ENGINE_STARTUP " + json.dumps(message, separators=(",", ":")),
        flush=True,
    )


def configure(policy):
    version = importlib.metadata.version("vllm")
    if version.split("+", 1)[0] != "0.31.0":
        raise RuntimeError("prepared startup requires pinned vLLM 0.31.0")
    os.environ.setdefault(
        "SWITCHYARD_STARTUP_EPOCH",
        str(
            time.monotonic()
            + float(os.environ.get("VLLM_CPU_SNAPSHOT_CLOCK_OFFSET_S", "0"))
        ),
    )
    if policy.worker_method == "guarded_fork":
        if os.environ.get("SWITCHYARD_RUNTIME_ARTIFACT_ID") != policy.fork_runtime_id:
            raise RuntimeError("guarded fork runtime identity mismatch")
        root = Path(os.environ["SWITCHYARD_RUNTIME_ROOT"])
        # The agent has already published and checked the pinned artifact. Read
        # its small receipt, rather than rehashing immutable runtime files.
        receipt = json.loads((root / "build-info.json").read_text())
        patches = receipt.get("source_patches", [])
        if not any(
            isinstance(patch, dict)
            and patch.get("descriptor_sha256") == NVML_PATCH
            and patch.get("after_sha256") == NVML_RESULT
            and patch.get("target") == NVML_TARGET
            and patch.get("package") == {"name": "vllm", "version": version}
            for patch in patches
        ):
            raise RuntimeError("guarded fork requires the baked pinned NVML FLA patch")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if policy.compute_capability is not None:
        previous = os.environ.get("FLASHINFER_CUDA_ARCH_LIST")
        if previous is not None and previous != policy.compute_capability:
            raise RuntimeError("FlashInfer architecture differs from selected GPU")
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = policy.compute_capability
    if policy.cache_policy == "required" and any(
        os.environ.get(name) != "1"
        for name in [
            "VLLM_FORCE_AOT_LOAD",
            "FLASHINFER_DISABLE_JIT",
            "FLASHINFER_NO_DOWNLOAD",
        ]
    ):
        raise RuntimeError(
            "required cache policy needs native strict AOT and JIT-disabled flags"
        )
    if policy.cache_policy == "required" and policy.compute_capability is None:
        raise RuntimeError("required cache policy needs a declared GPU target")
    if (
        policy.memory_plan == "required"
        and os.environ.get("VLLM_ENABLE_STARTUP_PLAN") != "1"
    ):
        raise RuntimeError("required memory plan needs the native startup-plan flag")
    record(
        policy,
        "policy_selected",
        strategy=policy.strategy,
        worker_method=policy.worker_method,
    )


class PrerequisiteImports(importlib.abc.MetaPathFinder):
    """Pause cache/cubin consumers until the agent opens the combined gate."""

    def __init__(self, policy):
        self.policy = policy
        self.ready = threading.Event()
        self.cancelled = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in PREREQUISITE_MODULES:
            if not self.ready.is_set():
                record(self.policy, "import_waiting_for_prerequisites", module=fullname)
            self.ready.wait()
            if self.cancelled:
                raise RuntimeError("native import prerequisite gate cancelled")

    def release(self, cancelled=False):
        self.cancelled = cancelled
        self.ready.set()


class Preparation:
    def __init__(self, policy):
        self.policy = policy
        self.error = None
        self.read_fd, self.write_fd = os.pipe()
        self.thread = threading.Thread(
            target=self.run, name="switchyard-native-imports", daemon=True
        )

    def run(self):
        try:
            for name in PREPARE_MODULES:
                began = time.monotonic()
                importlib.import_module(name)
                record(
                    self.policy,
                    "module_prepared",
                    module=name,
                    duration_s=time.monotonic() - began,
                )
        # Return every thread exit to the joining process, including SystemExit.
        except BaseException as error:  # noqa: BLE001
            self.error = error
            record(
                self.policy,
                "import_preparation_failed",
                error_type=type(error).__name__,
            )
        finally:
            os.write(self.write_fd, b"E" if self.error is not None else b"D")

    def join(self):
        self.thread.join()
        if self.error is not None:
            raise self.error

    def close(self):
        os.close(self.read_fd)
        os.close(self.write_fd)


def wait_gate(preparation):
    """Notice a failed import immediately while retaining one bounded gate."""
    if preparation is None:
        return sys.stdin.readline(len(GATE) + 1) == GATE
    with selectors.DefaultSelector() as selector:
        selector.register(sys.stdin.fileno(), selectors.EVENT_READ, "gate")
        selector.register(preparation.read_fd, selectors.EVENT_READ, "preparation")
        token = bytearray()
        while True:
            for key, _ in selector.select():
                if key.data == "preparation":
                    result = os.read(key.fd, 1)
                    selector.unregister(key.fd)
                    if result == b"E":
                        preparation.join()
                else:
                    part = os.read(key.fd, len(GATE) + 1 - len(token))
                    token.extend(part)
                    if not part or b"\n" in token or len(token) > len(GATE):
                        return bytes(token) == GATE.encode()


def cache_aliases(policy):
    if not policy.cache_aliases:
        return
    root = Path(os.environ["SWITCHYARD_CACHE_ROOT"])
    if not root.is_absolute() or not root.is_dir():
        raise RuntimeError("cache alias target is not ready")
    for value in policy.cache_aliases:
        alias = Path(value)
        if (
            not alias.is_absolute()
            or str(alias) != value
            or alias.parent != root.parent
            or ".." in alias.parts
        ):
            raise ValueError(
                "cache alias must be a normalized sibling of the selected cache root"
            )
        if alias == root:
            continue
        if alias.is_symlink():
            if alias.resolve() != root.resolve():
                raise RuntimeError("cache alias has another owner")
        elif alias.exists():
            raise RuntimeError("cache alias already exists without a checked link")
        else:
            alias.symlink_to(root)
    record(policy, "cache_aliases_ready", aliases=len(policy.cache_aliases))


def refresh_providers(policy):
    """Refresh the native inventory after late paths, without CUDA queries."""
    importlib.invalidate_caches()
    # The early-import fence normally leaves this module unimported. A pinned
    # refresh also repairs an inventory created earlier by sitecustomize.
    if policy.cache_policy != "required" and "flashinfer.jit.env" not in sys.modules:
        record(policy, "providers_deferred", reason="native_cache_policy")
        return
    if (
        importlib.metadata.version("flashinfer-python").split("+", 1)[0]
        != "0.7.0.post1"
    ):
        raise RuntimeError("native provider refresh requires FlashInfer 0.7.0.post1")
    environment = importlib.import_module("flashinfer.jit.env")
    providers = environment._get_aot_providers()
    environment.FLASHINFER_AOT_PROVIDERS = providers
    environment.FLASHINFER_AOT_DIRS = (environment.FLASHINFER_AOT_DIR,) + tuple(
        provider.jit_cache_dir for provider in providers
    )
    sampling = [
        provider
        for provider in providers
        if "sampling" in provider.modules
        and (provider.jit_cache_dir / "sampling/sampling.so").is_file()
    ]
    record(
        policy,
        "providers_ready",
        providers=len(providers),
        sampling_providers=len(sampling),
    )
    if policy.cache_policy == "required":
        # A declared architecture was installed before imports. This makes the
        # native resolver use that target instead of querying CUDA in the API.
        if policy.compute_capability is None:
            raise RuntimeError("required cache policy needs a declared GPU target")
        artifacts = environment.get_aot_artifacts("sampling")
        if (
            not sampling
            or not artifacts
            or not all(item.path.is_file() for item in artifacts)
        ):
            raise RuntimeError("required native AOT sampling provider is absent")
        record(
            policy,
            "strict_cache_ready",
            sampling_artifacts=len(artifacts),
            declared_compute_capability=policy.compute_capability,
        )


def main():
    policy = Policy.from_environment()
    if "VLLM_CPU_SNAPSHOT_CLOCK_OFFSET_S" in os.environ:
        # Spawn must execute the fresh guarded entrypoint, not the checkpoint
        # preparation script whose temporary path and arguments are obsolete.
        restored_main = sys.modules["__main__"]
        restored_main.__file__ = str(Path(__file__).with_name("serve.py"))
        restored_main.__spec__ = None
        record(policy, "process_entrypoint_rebound")
    if policy is None:
        # Preserve the legacy launcher and its native CLI defaults.
        import vllm

        if not wait_gate(None):
            raise RuntimeError("metadata readiness gate cancelled")
        sys.stdin.close()
        import runpy

        sys.argv = ["vllm", *sys.argv[1:]]
        runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")
        return
    configure(policy)
    finder = PrerequisiteImports(policy)
    sys.meta_path.insert(0, finder)
    preparation = Preparation(policy) if policy.strategy == "prepared" else None
    try:
        if preparation is not None:
            preparation.thread.start()
        record(policy, "waiting_for_prerequisites")
        if not wait_gate(preparation):
            finder.release(cancelled=True)
            raise RuntimeError("metadata and cache readiness gate cancelled")
        sys.stdin.close()
        cache_aliases(policy)
        importlib.invalidate_caches()
        os.environ["SWITCHYARD_STARTUP_GATE_RELEASED"] = "1"
        record(policy, "prerequisites_ready")
        finder.release()
        if preparation is not None:
            preparation.join()
        else:
            import vllm  # noqa: F401
        sys.meta_path.remove(finder)
        refresh_providers(policy)
        from switchyard_vllm_hooks import install

        install(policy)
        import torch

        os.environ["SWITCHYARD_PREPARED_PID"] = str(os.getpid())
        record(
            policy,
            "engine_construction_allowed",
            cuda_initialized=torch.cuda.is_initialized(),
        )
        sys.argv = ["vllm", *sys.argv[1:]]
        # The native parser, validator and ServeSubcommand dispatch still run.
        from vllm.entrypoints.cli.main import main as native_main

        native_main()
    finally:
        finder.release(cancelled=True)
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        if preparation is not None and preparation.thread.ident is not None:
            preparation.thread.join()
            preparation.close()


if __name__ == "__main__":
    main()
