"""Warm native sampling in an owned GPU container with a selected composition.

Mount runtime/build trees at their fixed prefixes, the runtime JSON at /config,
and a writable compiler subtree at /cache-root/compiled. No model is loaded.
"""

import hashlib
import json
import os
from pathlib import Path

configuration = json.loads(Path("/config").read_text())
for key, value in configuration["environment"].items():
    os.environ[key] = (
        value.replace("{runtime_root}", "/opt/switchyard/runtime")
        .replace("{components_root}", "/opt/switchyard/components")
        .replace("{cache_root}", "/cache-root")
    )
os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "12.0"
os.environ.pop("FLASHINFER_DISABLE_JIT", None)
os.environ["VLLM_FORCE_AOT_LOAD"] = "0"
os.environ["MAX_JOBS"] = "4"

import torch
from flashinfer import sampling
from flashinfer.jit import env

probabilities = torch.full((1, 128), 1 / 128, device="cuda", dtype=torch.float32)
output = sampling.sampling_from_probs(probabilities)
torch.cuda.synchronize()
assert tuple(output.shape) == (1,)
files = list(Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).rglob("sampling.so"))
assert len(files) == 1
print(
    json.dumps(
        {
            "stage": "native_sampling_warmed",
            "binary": str(files[0]),
            "binary_sha256": hashlib.file_digest(
                files[0].open("rb"), "sha256"
            ).hexdigest(),
            "native_target_architectures": sorted(env._target_cuda_architectures()),
            "gpu": torch.cuda.get_device_name(),
            "gpu_execution_tested": True,
        }
    ),
    flush=True,
)
