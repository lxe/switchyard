"""Independent F32 Transformers + PEFT oracle for native text-adapter serving.

Original Pocket VLM adapter keys are mapped explicitly to the corresponding
text-only HF module path. Every saved A/B pair must be installed exactly once.
"""
import argparse
import contextlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--native-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from peft import LoraConfig, get_peft_model
    from peft.utils.save_and_load import set_peft_model_state_dict

    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config=Qwen3_5TextConfig(**json.loads((args.model/'config.json').read_text())['text_config'])
    config._attn_implementation='eager'
    with torch.device('meta'):
        model=Qwen3_5ForCausalLM(config)
    weights={}
    for shard in sorted(args.model.glob('*.safetensors')):
        with safe_open(shard,framework='pt',device='cpu') as source:
            for key in source.keys():
                if key.startswith('model.language_model.'):
                    weights[key.replace('model.language_model.','model.',1)]=source.get_tensor(key).to('cuda',torch.float32)
                elif key=='lm_head.weight':
                    weights[key]=source.get_tensor(key).to('cuda',torch.float32)
    if config.tie_word_embeddings:
        weights['lm_head.weight']=weights['model.embed_tokens.weight']
    model.load_state_dict(weights,strict=True,assign=True)
    with torch.device('cuda'):
        model.model.rotary_emb=Qwen3_5TextRotaryEmbedding(config)
    model.eval()
    del weights
    peft_config=json.loads((args.adapter/'adapter_config.json').read_text())
    adapter={}
    targets=set()
    with safe_open(args.adapter/'adapter_model.safetensors',framework='pt',device='cpu') as source:
        for key in source.keys():
            prefix='base_model.model.model.language_model.'
            if not key.startswith(prefix) or not key.endswith(('.lora_A.weight','.lora_B.weight')):
                raise ValueError('Reference requires explicit standard Pocket text adapter keys: '+key)
            mapped=key.replace(prefix,'base_model.model.model.',1)
            adapter[mapped]=source.get_tensor(key).float()
            targets.add(mapped.removeprefix('base_model.model.').rsplit('.lora_',1)[0])
    if len(adapter)!=2*len(targets):
        raise ValueError('Unpaired reference adapter tensors')
    model=get_peft_model(model,LoraConfig(r=peft_config['r'],lora_alpha=peft_config['lora_alpha'],
        use_rslora=peft_config.get('use_rslora',False),target_modules=sorted(targets),lora_dropout=0,bias='none'))
    result=set_peft_model_state_dict(model,adapter,adapter_name='default')
    if result.unexpected_keys or any('.lora_' in key for key in result.missing_keys):
        raise ValueError('PEFT adapter loading is incomplete: '+str(result))
    loaded={k:v for k,v in model.state_dict().items() if '.lora_' in k}
    if len(loaded)!=len(adapter):
        raise ValueError('Installed A/B tensor count differs from source')
    for key,tensor in adapter.items():
        mapped=key.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
        if not torch.equal(loaded[mapped].cpu(),tensor):
            raise ValueError('Installed adapter tensor differs: '+mapped)
    model.eval()
    native=json.loads(args.native_receipt.read_text())
    def generate(request):
        tokens=torch.tensor([request['prompt']],device='cuda')
        cache=None
        ids=[]
        for _ in range(request['max_tokens']):
            value=model(input_ids=tokens,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=value.past_key_values
            tokens=value.logits[:,-1].argmax(-1).unsqueeze(-1)
            ids.append(int(tokens[0,0]))
        return ids
    output={}
    with torch.inference_mode():
        for name in ['base','adapter']:
            with model.disable_adapter() if name=='base' else contextlib.nullcontext():
                output[name]=[generate(request) for request in native['requests']]
    diffs=[]
    for name in output:
        expected=native[name+'_token_ids']
        for row,(reference,actual) in enumerate(zip(output[name],expected)):
            if reference!=actual:
                diffs.append({'mode':name,'row':row,'reference':reference,'native':actual})
    receipt={'reference':'Transformers Qwen3_5ForCausalLM eager + PEFT unmerged F32 LoRA',
             'torch_version':torch.__version__,'dtype':'f32','tf32':False,'loaded_adapter_tensors':len(loaded),
             'explicit_key_mapping':'base_model.model.model.language_model -> base_model.model.model; every source tensor checked against installed state',
             'generated_tokens':sum(len(row) for rows in output.values() for row in rows),'exact_match':not diffs,
             'outputs':output,'differences':diffs,'scope':'Text-only reference with explicit VLM adapter-key remapping; image/vision adapters and full VLM class are outside this reference'}
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:receipt[key] for key in ['loaded_adapter_tensors','generated_tokens','exact_match']}),flush=True)
    if diffs:
        raise RuntimeError('Native/PEFT token mismatch; see retained differences')


if __name__=='__main__':
    main()
