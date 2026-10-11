"""Run the pinned vLLM model with diagnostic teacher forcing, one row at a time."""
import json
from pathlib import Path
from vllm_reference import fixed_budget  # installs the existing explicit budget override
from vllm import LLM, SamplingParams


def main():
    traces = json.loads(Path("/output/traces.json").read_text())
    engine = LLM(model="/model", dtype="bfloat16", language_model_only=True,
        max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128,
        kv_cache_memory_bytes=1024**3, gpu_memory_utilization=0.25, enforce_eager=True,
        enable_prefix_caching=False, attention_config={"backend":"TRITON_ATTN"},
        logits_processors=["teacher_oracle:TraceOracle"], seed=123)
    for row, trace in enumerate(traces):
        result = engine.generate({"prompt_token_ids":trace["prompt"]},
            SamplingParams(temperature=0, ignore_eos=True, max_tokens=len(trace["continuation"]),
                extra_args={"trace_row":row,"trace":trace["continuation"]}), use_tqdm=False)
        if list(result[0].outputs[0].token_ids) != trace["continuation"]:
            raise RuntimeError("Reference teacher forcing did not emit the expected trace")


if __name__ == "__main__":
    main()
