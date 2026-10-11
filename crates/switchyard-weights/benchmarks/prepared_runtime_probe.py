"""CPU-only native import/provider acceptance inside the pinned runtime.

Mount the complete runtime read-only at /opt/switchyard/runtime, a baked cache
at /cache-source, this bundle at /engine, and writable /late and /components.
Set PYTHONPATH=/engine:/late/alias/sampling-provider before Python starts.
This proves native late discovery, not GPU binary compatibility or throughput.
"""

import importlib
import json
import os
import sys
import threading
from pathlib import Path

from switchyard_vllm_startup import (
    PREREQUISITE_MODULES,
    Policy,
    Preparation,
    PrerequisiteImports,
    cache_aliases,
    configure,
    refresh_providers,
)

entered = threading.Event()


class ObservedImports(PrerequisiteImports):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in PREREQUISITE_MODULES:
            entered.set()
        return super().find_spec(fullname, path, target)


policy = Policy(
    strategy="prepared",
    compute_capability=os.environ.get("PROBE_CUDA_ARCH", "8.9"),
    cache_policy="required",
    cache_aliases=["/late/alias"],
)
configure(policy)
finder = ObservedImports(policy)
sys.meta_path.insert(0, finder)
preparation = Preparation(policy)
preparation.thread.start()
try:
    assert entered.wait(60), "native preparation did not reach FlashInfer consumer"
    assert "flashinfer.jit.env" not in sys.modules, (
        "provider inventory discovered before prerequisites"
    )
    assert (
        sys.path_importer_cache.get("/late/alias/sampling-provider", "uncached") is None
    )
    assert not Path("/components/build").exists()
    Path("/late/selected").symlink_to("/cache-source")
    build = Path("/components/build")
    build.mkdir()
    runtime = Path("/opt/switchyard/runtime")
    (build / "cuda").symlink_to(runtime / "cuda")
    (build / "toolchain").symlink_to(runtime / "toolchain")
    (build / "flashinfer-cubins").symlink_to(
        runtime / "venv/lib/python3.12/site-packages/flashinfer_cubin/cubins"
    )
    cache_aliases(policy)
    importlib.invalidate_caches()
    os.environ["SWITCHYARD_STARTUP_GATE_RELEASED"] = "1"
    finder.release()
    preparation.join()
    if os.environ.get("PROBE_EXPECT_CACHE_MISS") == "1":
        try:
            refresh_providers(policy)
        except RuntimeError as error:
            assert "required native AOT sampling" in str(error)
        else:
            raise AssertionError("incompatible native AOT target was accepted")
        import torch

        assert not torch.cuda.is_initialized()
        print(
            json.dumps(
                {
                    "stage": "native_cache_target_rejected",
                    "cuda_initialized": False,
                    "gpu_execution_tested": False,
                }
            ),
            flush=True,
        )
        sys.exit(0)
    refresh_providers(policy)
    import torch
    from flashinfer.jit import env

    assert len(env.FLASHINFER_AOT_PROVIDERS) == 1
    assert "sampling" in env.FLASHINFER_AOT_PROVIDERS[0].modules
    assert not torch.cuda.is_initialized()
    print(
        json.dumps(
            {
                "stage": "native_prepared_runtime_passed",
                "late_provider_count": 1,
                "importer_negative_before_gate": True,
                "cuda_initialized": False,
                "gpu_execution_tested": False,
                "prepared_modules": list(
                    __import__("switchyard_vllm_startup").PREPARE_MODULES
                ),
            }
        ),
        flush=True,
    )
finally:
    finder.release(cancelled=True)
    if finder in sys.meta_path:
        sys.meta_path.remove(finder)
    preparation.thread.join()
    preparation.close()
