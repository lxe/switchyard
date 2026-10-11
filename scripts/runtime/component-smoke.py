# CPU-only same-process Python/vLLM resource smoke, invoked by the Rust preparation CLI.

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time

control = Path('/control')
config = json.loads((control / 'probe.json').read_text())
started = time.monotonic()

def record(name, value):
    temporary = control / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(control / name)

modules = {name: importlib.import_module(name) for name in config['imports']}
resources = []
for check in config['resources']:
    module = importlib.import_module(check['module'])
    loader_module, loader_name = check['loader'].rsplit('.', 1)
    loader = getattr(importlib.import_module(loader_module), loader_name)
    resource = Path(getattr(module, check['attribute']))
    assert str(resource) == check['directory'], (str(resource), check['directory'])
    assert not resource.exists(), resource
    resources.append((check, module, resource, loader))
torch = modules.get('torch')
assert torch is None or not torch.cuda.is_initialized(), 'boot import initialized CUDA'
root = config['runtime_root'] + '/'
loaded = set()
for line in Path('/proc/self/maps').read_text().splitlines():
    path = line.split(maxsplit=5)[-1]
    if path.startswith(root):
        loaded.add(path[len(root):])
record('boot-ready.json', {'imports': list(modules), 'cuda_initialized': False,
                         'components_absent': True, 'resource_paths': [str(r[2]) for r in resources],
                         'loaded_runtime_libraries': sorted(loaded),
                         'import_seconds': time.monotonic() - started})
deadline = time.monotonic() + 180
while not (control / 'components.ready').exists():
    if time.monotonic() > deadline:
        raise TimeoutError('component publication gate did not open')
    time.sleep(.1)
checks = []
for check, module, before, loader in resources:
    assert importlib.import_module(check['module']) is module
    after = Path(getattr(module, check['attribute']))
    assert before == after and after.is_dir()
    data = loader(check['file'], check['sha256'])
    assert len(data) == check['size']
    assert hashlib.sha256(data).hexdigest() == check['sha256']
    checks.append({'module': check['module'], 'path_stable': True, 'same_module': True,
                   'file': check['file'], 'bytes': len(data), 'sha256': check['sha256'],
                   'official_loader': check['loader']})
compiler_checks = []
for action in config['compilers']:
    if 'input' in action:
        Path(action['input_path']).write_text(action['input'])
    result = subprocess.run(action['command'], check=True, text=True, capture_output=True, timeout=120)
    output = result.stdout + result.stderr
    if action.get('expected') and action['expected'] not in output:
        raise ValueError('compiler acceptance output differs: ' + action['name'])
    compiler_checks.append({'name': action['name'], 'command': action['command'], 'output': output})
record('complete.json', {'resources': checks, 'compiler_checks': compiler_checks,
                        'cuda_initialized': torch.cuda.is_initialized() if torch else False,
                        'same_process': True, 'seconds': time.monotonic() - started})
