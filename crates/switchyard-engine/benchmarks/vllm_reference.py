"""Offline token oracle. Run inside the pinned Switchyard Ubuntu runtime.

This comparison uses explicit 1 GiB KV allocation and eager execution. It is
not a replacement for the production artifact/profile admission adapter.
"""
import json
import time
from pathlib import Path

START = time.monotonic()

from vllm.v1.worker.gpu_worker import Worker


def fixed_budget(worker):
    # No dummy profiling forward and no free-memory fraction sizing.
    if worker.cache_config.kv_cache_memory_bytes != 1024**3:
        raise ValueError("Reference requires explicit 1 GiB KV allocation")
    worker.available_kv_cache_memory_bytes = 1024**3
    return 1024**3


Worker.determine_available_memory = fixed_budget


def main():
    from vllm import LLM, SamplingParams

    requests = json.loads(Path("/output/requests.json").read_text())
    engine = LLM(
        model="/model", dtype="bfloat16", language_model_only=True,
        max_model_len=128, max_num_seqs=8, max_num_batched_tokens=128,
        kv_cache_memory_bytes=1024**3, gpu_memory_utilization=0.25, enforce_eager=True,
        enable_prefix_caching=False, attention_config={"backend": "TRITON_ATTN"},
        seed=123,
    )
    ready = time.monotonic() - START
    results = {}
    for mode in ("single", "batch"):
        groups = [[r] for r in requests] if mode == "single" else [requests]
        outputs = []
        began = time.monotonic()
        for group in groups:
            response = engine.generate(
                [{"prompt_token_ids": r["prompt"]} for r in group],
                [SamplingParams(temperature=0, max_tokens=r["max_tokens"],
                                ignore_eos=r.get("ignore_eos", False)) for r in group],
                use_tqdm=False,
            )
            outputs.extend({"token_ids": list(r.outputs[0].token_ids),
                            "text": r.outputs[0].text} for r in response)
        results[mode] = {"outputs": outputs, "elapsed_s": time.monotonic() - began}
    Path("/output/vllm.json").write_text(json.dumps({
        "engine_ready_s": ready, "vllm_version": __import__("vllm").__version__,
        "results": results, "settings": {"eager": True, "kv_bytes": 1024**3,
        "max_context": 128, "max_batch": 8, "prefix_cache": False},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
