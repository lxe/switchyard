"""Diagnostic-only teacher forcing; capture real logits before selecting trace IDs."""
import json
from pathlib import Path

import torch
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor


class TraceOracle(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.rows = {}
        self.req_states = req_states
        self.output = Path("/output/teacher-vllm.jsonl").open("w")

    def add_request(self, index, params):
        self.rows[index] = (params.extra_args["trace_row"], params.extra_args["trace"])
        return True

    def apply(self, logits, ctx):
        for index, slot in enumerate(ctx.expanded_idx_mapping.tolist()):
            row, trace = self.rows[slot]
            step = int(ctx.pos[index].item()) + 1 - int(self.req_states.prompt_len.np[slot])
            expected = trace[step]
            values = logits[index].float()
            logprobs = values.log_softmax(-1)
            top_values, top_ids = values.topk(20)
            top_probs = logprobs[top_ids]
            self.output.write(json.dumps({"row": row, "step": step, "expected": expected, "argmax": int(values.argmax().item()),
                "expected_logprob": logprobs[expected].item(),
                "top": [{"token": token, "logit": value, "logprob": prob}
                    for token, value, prob in zip(top_ids.tolist(), top_values.tolist(), top_probs.tolist())]}) + "\n")
            self.output.flush()
            logits[index].fill_(float("-inf"))
            logits[index, expected] = 0
        return logits
