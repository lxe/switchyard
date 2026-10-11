"""F32 greedy single and equal-length batch oracle using released model math."""
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from hf_teacher_reference import load_model


def generate(model, requests):
    tokens = torch.tensor([r['prompt'] for r in requests], device='cuda')
    cache = None
    outputs = [[] for _ in requests]
    latencies = [[] for _ in requests]
    started = time.perf_counter()
    for step in range(max(r['max_tokens'] for r in requests)):
        result = model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        tokens = result.logits[:, -1].argmax(-1).unsqueeze(-1)
        ids = tokens.flatten().tolist()
        elapsed = time.perf_counter()-started
        for index, request in enumerate(requests):
            if step < request['max_tokens']:
                outputs[index].append(ids[index])
                latencies[index].append(elapsed)
    return [{'token_ids':ids, 'token_latency_s':times} for ids,times in zip(outputs,latencies)]


def main():
    requests = json.loads(Path('/output/requests.json').read_text())
    if any(not r['ignore_eos'] for r in requests):
        raise ValueError('This diagnostic oracle requires explicit ignore_eos traces')
    started = time.perf_counter()
    model = load_model()
    ready = time.perf_counter()-started
    with torch.inference_mode():
        single = [generate(model, [r])[0] for r in requests]
        groups = defaultdict(list)
        for index, request in enumerate(requests):
            groups[len(request['prompt'])].append(index)
        batch = [None]*len(requests)
        for group in groups.values():
            for start in range(0, len(group), 8):
                indices = group[start:start+8]
                results = generate(model, [requests[i] for i in indices])
                for index, result in zip(indices, results):
                    batch[index] = result
    Path('/output/hf.json').write_text(json.dumps({'engine_ready_s':ready,
        'results':{'single':{'outputs':single},'batch':{'outputs':batch}},
        'settings':{'dtype':'f32','tf32':False,'reference':'Transformers eager Qwen3_5ForCausalLM',
            'ignore_eos':True,'max_batch_size':8}}, indent=2))


if __name__ == '__main__':
    main()
