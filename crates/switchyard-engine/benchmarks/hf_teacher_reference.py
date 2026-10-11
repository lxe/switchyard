"""Offline F32 mathematical oracle using Transformers' unoptimized Qwen math."""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def load_model():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Qwen3_5TextConfig(**json.loads(Path('/model/config.json').read_text())['text_config'])
    cfg._attn_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(cfg)
    weights = {}
    for shard in Path('/model').glob('*.safetensors'):
        with safe_open(shard, framework='pt', device='cpu') as source:
            for key in source.keys():
                if key.startswith('model.language_model.'):
                    weights[key.replace('model.language_model.', 'model.', 1)] = source.get_tensor(key).to(device='cuda', dtype=torch.float32)
                elif key == 'lm_head.weight':
                    weights[key] = source.get_tensor(key).to(device='cuda', dtype=torch.float32)
    if cfg.tie_word_embeddings:
        weights['lm_head.weight'] = weights['model.embed_tokens.weight']
    model.load_state_dict(weights, strict=True, assign=True)
    # Nonpersistent rotary buffers are absent from a safetensors checkpoint.
    # Explicitly construct them after materializing the meta model.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    with torch.device('cuda'):
        model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg)
    if model.model.rotary_emb.inv_freq[0].item() != 1.0:
        raise RuntimeError('Invalid default rotary frequencies')
    model.eval()
    return model


def main():
    model = load_model()
    captured = {}
    capturing = True
    def save(name, tensor):
        captured[name] = tensor.detach().float().cpu().flatten().tolist()
    def capture(name):
        def hook(module, inputs, output):
            if capturing:
                save(name, output)
        return hook
    model.model.embed_tokens.register_forward_hook(capture('embedding'))
    for index, layer in enumerate(model.model.layers):
        layer.register_forward_hook(capture(f'layer_{index}'))
    attention = next(x.self_attn for x in model.model.layers if hasattr(x, 'self_attn'))
    for name in ['q_proj','q_norm','k_proj','k_norm','v_proj']:
        getattr(attention, name).register_forward_hook(capture(name))
    def capture_o_input(module, inputs):
        if capturing:
            save('o_input', inputs[0])
    attention.o_proj.register_forward_pre_hook(capture_o_input)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
    original_rope = implementation.apply_rotary_pos_emb
    def capture_rope(*args, **kwargs):
        result = original_rope(*args, **kwargs)
        if capturing and 'q_rope' not in captured:
            save('q_rope', result[0])
            save('k_rope', result[1])
        return result
    implementation.apply_rotary_pos_emb = capture_rope
    traces = json.loads(Path('/output/traces.json').read_text())
    with torch.inference_mode(), Path('/output/teacher-hf.jsonl').open('w') as out:
        for row, trace in enumerate(traces):
            cache = None
            tokens = trace['prompt']
            for step, expected in enumerate(trace['continuation']):
                result = model(input_ids=torch.tensor([tokens], device='cuda'),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                if capturing:
                    Path('/output/layers-hf.json').write_text(json.dumps(captured))
                    capturing = False
                cache = result.past_key_values
                values = result.logits[0, -1].float()
                probs = values.log_softmax(-1)
                top_values, top_ids = values.topk(20)
                out.write(json.dumps({'row':row, 'step':step, 'expected':expected,
                    'argmax':int(values.argmax().item()), 'expected_logprob':probs[expected].item(),
                    'top':[{'token':t, 'logit':v, 'logprob':p} for t,v,p in
                        zip(top_ids.tolist(), top_values.tolist(), probs[top_ids].tolist())]})+'\n')
                out.flush()
                tokens = [expected]


if __name__ == '__main__':
    main()
