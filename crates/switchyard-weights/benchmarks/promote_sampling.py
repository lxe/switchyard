"""Register an executed native sampling binary through FlashInfer's provider API.

The official provider-discovery shim is mounted at /provider-source. The warmed
cache is mounted at /cache-root/compiled. No model or serving engine is started.
"""

import json
import shutil
from pathlib import Path

root = Path("/cache-root/compiled")
binary = next(
    root.glob(
        "flashinfer/.cache/flashinfer/0.7.0.post1/120f/cached_ops/sampling/sampling.so"
    )
)
provider = root / "sampling-provider"
provider.mkdir(exist_ok=True)
for name in [
    "flashinfer_jit_cache",
    "flashinfer_jit_cache-0.7.0.post1+cu129.dist-info",
]:
    shutil.copytree(
        Path("/provider-source") / name, provider / name, dirs_exist_ok=True
    )
(provider / "switchyard_flashinfer_sampling.py").write_text("""from pathlib import Path
def get_provider():
    root = Path(__file__).resolve().parent.parent
    return {"schema_version": 1, "distribution": "switchyard-flashinfer-sampling",
            "version": "0.7.0.post1+cu129", "jit_cache_dir": root / "flashinfer/.cache/flashinfer/0.7.0.post1/120f/cached_ops",
            "cuda_architectures": ["sm120f"], "modules": ["sampling"]}
""")
metadata = provider / "switchyard_flashinfer_sampling-0.7.0.post1+cu129.dist-info"
metadata.mkdir(exist_ok=True)
(metadata / "METADATA").write_text(
    "Metadata-Version: 2.1\nName: switchyard-flashinfer-sampling\nVersion: 0.7.0.post1+cu129\n"
)
(metadata / "entry_points.txt").write_text(
    "[flashinfer.jit_cache.providers]\nswitchyard-sm120f-sampling = switchyard_flashinfer_sampling:get_provider\n"
)
print(
    json.dumps(
        {
            "stage": "native_sampling_provider_registered",
            "architecture": "sm120f",
            "binary": str(binary),
        }
    ),
    flush=True,
)
