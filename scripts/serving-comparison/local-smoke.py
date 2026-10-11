"""Exercise a mounted serving runtime and pinned model without network access.

Run inside the clean smoke image with /runtime, /model, /cache and /config.json.
The runtime must be mounted at its fixed prefix /opt/switchyard/runtime.
Local compiler caches are a functional preflight and must not seed cloud caches.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--memory-fraction', type=float, default=0.25)
parser.add_argument('--timeout-s', type=float, default=400)
args = parser.parse_args()
config = json.loads(Path('/config.json').read_text())
model = config['models'][0]
engine = model['engine']
kind = engine.get('kind', 'vllm')
for key, value in config['bootstrap']['runtime']['environment'].items():
    os.environ[key] = value.replace('{runtime_root}', '/opt/switchyard/runtime').replace(
        '{cache_root}', '/cache')

import torch
import flashinfer.sampling
assert torch.cuda.is_available()
probs = torch.full((1, 128), 1 / 128, device='cuda', dtype=torch.float32)
assert flashinfer.sampling.sampling_from_probs(probs).shape == (1,)
torch.cuda.synchronize()
print(json.dumps({'stage':'sampling-kernel-passed','gpu':torch.cuda.get_device_name(),
                  'compute_capability':torch.cuda.get_device_capability()}), flush=True)

if kind == 'vllm':
    command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', '/model',
               '--max-model-len', str(engine['max_model_len']),
               '--max-num-seqs', str(engine['max_num_seqs']),
               '--gpu-memory-utilization', str(args.memory_fraction)]
else:
    command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', '/model',
               '--context-length', str(engine['max_model_len']),
               '--max-running-requests', str(engine['max_num_seqs']),
               '--mem-fraction-static', str(args.memory_fraction)]
command += ['--served-model-name', model['id'], '--host', '127.0.0.1', '--port', '8000',
            '--dtype', engine['dtype']]
if engine['language_model_only']:
    command += (['--json-model-override-args', '{"language_model_only":true}']
                if kind == 'sglang' and model['repository'].startswith('Qwen/Qwen3.5-')
                else ['--language-model-only'])
command += engine['extra_args']
began = time.monotonic()
child = subprocess.Popen(command, start_new_session=True)

def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request('http://127.0.0.1:8000' + path, data=data,
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response) if payload is not None else response.status

try:
    while True:
        if child.poll() is not None:
            raise RuntimeError('engine exited before readiness')
        if time.monotonic() - began > args.timeout_s:
            raise TimeoutError('local engine readiness deadline')
        try:
            if request('/health') == 200:
                break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    ready_s = time.monotonic() - began
    messages = [{'role':'user','content':'Reply exactly POCKET_MODEL_OK.\nRequest identifier: 0.'}]
    tokens = request('/tokenize', {'model':model['id'],'messages':messages,
                                  'add_generation_prompt':True})
    response = request('/v1/chat/completions', {'model':model['id'],'messages':messages,
                'temperature':0,'max_tokens':96,'chat_template_kwargs':{'enable_thinking':False}})
    text = response['choices'][0]['message']['content']
    assert text.strip() == 'POCKET_MODEL_OK', 'literal response failed'
    assert tokens['count'] == response['usage']['prompt_tokens'], 'token count mismatch'
    print(json.dumps({'stage':'engine-passed','engine':kind,'ready_s':ready_s,
                      'input_tokens':tokens['count'],'usage':response['usage'],
                      'literal_response_passed':True}), flush=True)
finally:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
