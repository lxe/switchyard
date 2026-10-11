import json
import os
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    start = time.perf_counter()

    mode = sys.argv[1]
    extra = (
        {}
        if mode == "legacy"
        else {
            "socket_path": os.environ["SWITCHYARD_BENCH_SOCKET"],
            "cuda_stage": True,
            "pinned_bytes": 1024 * 1024,
        }
    )
    llm = LLM(
        model=sys.argv[2],
        skip_tokenizer_init=True,
        load_format="benchmark_" + mode,
        model_loader_extra_config=extra,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=False,
        attention_backend="TRITON_ATTN",
        compilation_config={"mode": 0},
        seed=123,
    )
    ready = time.perf_counter() - start
    result = llm.generate(
        [{"prompt_token_ids": [1, 5, 23, 8]}],
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
    )
    output = {
        "engine_ready_seconds": ready,
        "inference_seconds": time.perf_counter() - start - ready,
        "tokens": list(result[0].outputs[0].token_ids),
    }
    Path(sys.argv[3]).write_text(json.dumps(output))


if __name__ == "__main__":
    main()
