"""One necessary native GPU bake: warm kernels, export final serving tensors.

Run only through the root-coordinated preparation script. This is not a serving
benchmark and never captures a GPU process checkpoint.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

BOOT = "0d55f050cee7aa0a24392b3d824869e66516685836750eca88494ea060f7aec6"
SOURCE_CONTENT = "ba047becd40d1a7b6c6dfb5939e91f93b2bfc3cf139d570ea88b58a2a32c5e1e"
CACHE = Path("/opt/switchyard/cache") / BOOT
KV_BYTES = 1024 * 1024 * 1024
configuration = json.loads(Path("/config").read_text())
runtime = configuration.get("bootstrap", {}).get("runtime", configuration)
for key, value in runtime["environment"].items():
    os.environ[key] = (
        value.replace("{runtime_root}", "/opt/switchyard/runtime")
        .replace("{components_root}", "/opt/switchyard/components")
        .replace("{cache_root}", str(CACHE))
    )
os.environ.pop("VLLM_ENGINE_STARTUP_POLICY", None)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Local owned bake only: the callable exports state from this process's worker.
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "12.0"
os.environ["PYTHONPATH"] = "/engine-source:" + str(CACHE / "compiled/sampling-provider")
os.environ["SWITCHYARD_RUNTIME_ARTIFACT_ID"] = BOOT
sys.path[:0] = ["/engine-source", str(CACHE / "compiled/sampling-provider")]

import torch
from switchyard_final_state import export_model
from switchyard_serving_profile import engine_factors
from vllm import LLM, SamplingParams


def export_worker(worker):
    model = worker.model_runner.get_model()
    output = Path("/bake/model")
    description = export_model(
        model, output, {"runtime_id": BOOT, "source_content_id": SOURCE_CONTENT}
    )
    properties = torch.cuda.get_device_properties(worker.device)
    binary = next(
        (CACHE / "compiled").glob(
            "flashinfer/.cache/flashinfer/0.7.0.post1/120f/cached_ops/sampling/sampling.so"
        )
    )
    with binary.open("rb") as source:
        binary_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    profile = {
        "schema_version": 1,
        "purpose": "fixed_serving_budget",
        "runtime_id": BOOT,
        "model_content_id": None,
        "source_content_id": SOURCE_CONTENT,
        "final_layout_sha256": description["layout_sha256"],
        "engine": engine_factors(worker.vllm_config),
        "gpu": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        },
        "versions": {
            "vllm": __import__("importlib.metadata").metadata.version("vllm"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "kv_cache_memory_bytes": KV_BYTES,
        "runtime_headroom_bytes": 1024 * 1024 * 1024,
        "bake": {
            "native_warmup_completed": True,
            "sampling_binary": str(binary.relative_to(CACHE)),
            "sampling_sha256": binary_sha256,
            "budget_source": "explicit bounded 1 GiB KV allocation and 1 GiB runtime headroom for RTX5090 .8B ctx64 seq1; no free-memory fraction",
        },
    }
    (Path("/bake") / "serving-profile.unbound.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    return description


def main():
    print(
        "[preparation] Native GPU bake begins: HF mapping/fusion occurs once here, final tensors are exported.",
        flush=True,
    )
    model_output = Path("/bake/model")
    model_output.mkdir(parents=True, exist_ok=True)
    for path in Path("/model").iterdir():
        if (
            path.is_file()
            and not path.name.endswith(".safetensors")
            and path.name != "model.safetensors.index.json"
        ):
            shutil.copy2(path, model_output / path.name)
    engine = LLM(
        model="/model",
        dtype="bfloat16",
        language_model_only=True,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
        kv_cache_memory_bytes=KV_BYTES,
        enforce_eager=True,
        attention_config={"backend": "TRITON_ATTN"},
        seed=123,
    )
    result = engine.chat(
        [{"role": "user", "content": "Reply with exactly the word Paris."}],
        sampling_params=SamplingParams(temperature=0, max_tokens=4),
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    if result[0].outputs[0].text.strip() != "Paris":
        raise RuntimeError("representative native bake decode failed")
    print(
        "[preparation] Representative Paris decode warmed driver and decode caches.",
        flush=True,
    )
    descriptions = engine.collective_rpc(export_worker, timeout=180)
    print(
        json.dumps(
            {
                "stage": "native_final_state_exported",
                "workers": descriptions,
                "serving_profile": "/bake/serving-profile.unbound.json",
                "gpu_checkpoint": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
