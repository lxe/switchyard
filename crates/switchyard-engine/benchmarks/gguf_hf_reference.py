"""Independent offline F32 Qwen3.5 oracle decoded from the same GGUF file.

Requires Torch/Transformers plus llama.cpp's MIT gguf-py package. This is a
reference test, never a serving conversion path. GGUF quantization changes the
weights, so comparison against an unrelated original checkpoint is insufficient.
"""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def grouped_indices(key_heads, value_heads, width):
    """Indexes selecting original grouped channels from GGUF tiled channels."""
    return np.arange(value_heads * width).reshape(value_heads // key_heads, key_heads, width).transpose(1, 0, 2).reshape(-1)


def language_weights(reader, config, device):
    c = config
    kh, vh, width = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_value_head_dim
    channels = grouped_indices(kh, vh, width)
    heads = grouped_indices(kh, vh, 1)
    qk = 2 * kh * c.linear_key_head_dim
    mappings = {
        'attn_norm.weight': 'input_layernorm.weight',
        'post_attention_norm.weight': 'post_attention_layernorm.weight',
        'attn_q.weight': 'self_attn.q_proj.weight',
        'attn_k.weight': 'self_attn.k_proj.weight',
        'attn_v.weight': 'self_attn.v_proj.weight',
        'attn_output.weight': 'self_attn.o_proj.weight',
        'attn_q_norm.weight': 'self_attn.q_norm.weight',
        'attn_k_norm.weight': 'self_attn.k_norm.weight',
        'ffn_gate.weight': 'mlp.gate_proj.weight',
        'ffn_up.weight': 'mlp.up_proj.weight',
        'ffn_down.weight': 'mlp.down_proj.weight',
        'attn_qkv.weight': 'linear_attn.in_proj_qkv.weight',
        'attn_gate.weight': 'linear_attn.in_proj_z.weight',
        'ssm_beta.weight': 'linear_attn.in_proj_b.weight',
        'ssm_alpha.weight': 'linear_attn.in_proj_a.weight',
        'ssm_conv1d.weight': 'linear_attn.conv1d.weight',
        'ssm_dt.bias': 'linear_attn.dt_bias',
        'ssm_a': 'linear_attn.A_log',
        'ssm_norm.weight': 'linear_attn.norm.weight',
        'ssm_out.weight': 'linear_attn.out_proj.weight',
    }
    weights = {}
    for tensor in reader.tensors:
        name = tensor.name
        if name == 'token_embd.weight':
            target = 'model.embed_tokens.weight'
        elif name == 'output_norm.weight':
            target = 'model.norm.weight'
        elif name == 'output.weight':
            target = 'lm_head.weight'
        elif name.startswith('blk.'):
            _, layer, suffix = name.split('.', 2)
            target = f'model.layers.{layer}.{mappings[suffix]}'
        else:
            raise ValueError(f'Unsupported language tensor {name}')
        values = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape)))
        if '.linear_attn.' in target:
            if target.endswith('in_proj_qkv.weight'):
                values = np.concatenate([values[:qk], values[qk:][channels]], axis=0)
            elif target.endswith('in_proj_z.weight'):
                values = values[channels]
            elif target.endswith(('in_proj_a.weight', 'in_proj_b.weight', 'dt_bias', 'A_log')):
                values = values[heads]
            elif target.endswith('out_proj.weight'):
                values = values[:, channels]
            elif target.endswith('conv1d.weight'):
                values = np.concatenate([values[:qk], values[qk:][channels]], axis=0)[:, None, :]
            if target.endswith('A_log'):
                if not np.all(values < 0):
                    raise ValueError('GGUF recurrent decay must be negative')
                values = np.log(-values)
        elif target.endswith('norm.weight'):
            values = values - 1.0
        weights[target] = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(device)
    if config.tie_word_embeddings:
        weights['lm_head.weight'] = weights['model.embed_tokens.weight']
    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--dense-native-output', type=Path, help='Offline diagnostic dense bundle; never a serving fallback')
    p.add_argument('--export-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = Qwen3_5TextConfig(**json.loads(args.config.read_text())['text_config'])
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(config)
    reader = gguf.GGUFReader(str(args.gguf))
    model.load_state_dict(language_weights(reader, config, args.device), strict=True, assign=True)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    with torch.device(args.device):
        model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    model.eval()
    if args.dense_native_output is not None:
        import shutil
        from safetensors.torch import save_file
        args.dense_native_output.mkdir(exist_ok=False)
        native = {}
        for name, value in model.state_dict().items():
            if name == 'lm_head.weight' and config.tie_word_embeddings:
                continue
            target = name.replace('model.', 'model.language_model.', 1) if name.startswith('model.') else name
            native[target] = value.detach().cpu().contiguous()
        save_file(native, str(args.dense_native_output / 'model.safetensors'))
        for name in ['config.json', 'tokenizer.json', 'chat_template.jinja', 'LICENSE']:
            source = args.config.parent / name
            if source.exists():
                shutil.copyfile(source, args.dense_native_output / name)
    if args.export_only:
        if args.dense_native_output is None:
            raise ValueError('--export-only requires --dense-native-output')
        return
    traces = json.loads(args.traces.read_text())
    with torch.inference_mode(), args.output.open('w') as out:
        for row, trace in enumerate(traces):
            cache = None
            tokens = trace['prompt']
            for step, expected in enumerate(trace['continuation']):
                result = model(input_ids=torch.tensor([tokens], device=args.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = result.past_key_values
                logits = result.logits[0, -1].float()
                probs = logits.log_softmax(-1)
                out.write(json.dumps({'row': row, 'step': step, 'expected': expected, 'argmax': int(logits.argmax()), 'expected_logprob': float(probs[expected])}) + '\n')
                out.flush()
                tokens = [expected]
            print(f'reference row={row} positions={len(trace["continuation"])}', flush=True)


if __name__ == '__main__':
    main()
