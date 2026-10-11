"""Derive a scoped SM89 Qwen3.5 BF16 runtime without universal FlashInfer cubins.

Preserve the pinned package code, native libraries, CUDA toolchain, headers,
checksums, directories and compiler-cache entries. Remove only packaged cubin
payload files under flashinfer_cubin/cubins. The generated 0.8B configuration
checks SM89 before imports, disables downloads and FlashInfer JIT, and requires
Torch AOT. A native GPU gauntlet must accept this derivative before deployment.
Neither this script nor artifact publication allocates a GPU.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime-tree', required=True, type=Path)
p.add_argument('--runtime-artifact', required=True, type=Path)
p.add_argument('--cache-tree', required=True, type=Path)
p.add_argument('--cache-artifact', required=True, type=Path)
p.add_argument('--config', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
p.add_argument('--profile', action='store_true')
p.add_argument('--validate-only', action='store_true', help='Check source configuration identity without cloning or packing')
p.add_argument('--agent-url', help='Optional immutable agent release pin; requires --agent-sha256')
p.add_argument('--agent-sha256', help='SHA-256 of the optional agent release')
a = p.parse_args()
if a.out.exists():
    raise SystemExit('fresh output required')
cli = 'target/release/switchyard'
cfg = json.loads(a.config.read_text())
if a.agent_url or a.agent_sha256:
    assert a.agent_url and a.agent_sha256 and len(a.agent_sha256) == 64
    cfg['bootstrap']['agent_url'] = a.agent_url
    cfg['bootstrap']['agent_sha256'] = a.agent_sha256
source = json.loads((a.runtime_artifact / 'manifest.json').read_text())
cache_source = json.loads((a.cache_artifact / 'manifest.json').read_text())
assert source['metadata']['kind'] == 'runtime'
assert source['metadata']['compatibility']['vllm'] == '0.31.0+cu129'
assert cfg['bootstrap']['runtime']['artifact']['id'] == source['id']
assert len(cfg['models']) == 1 and cfg['models'][0]['id'] == 'qwen3.5-0.8b'
engine = cfg['models'][0]['engine']
assert engine['dtype'] == 'bfloat16' and engine['language_model_only']
# Bind the entire tested configuration, including absence of speculative
# decoding or an attention-backend override, before touching any files.
engine_fingerprint = hashlib.sha256(json.dumps(engine, separators=(',', ':')).encode()).hexdigest()
assert engine_fingerprint == '2104f4dff126bdef93d357755a0d12ed48ca79a0148bb6202cc10e454b1d0e40'
assert cache_source['metadata']['compatibility']['engine_configuration'] == engine_fingerprint
assert cfg['models'][0]['caches'][0]['artifact']['id'] == cache_source['id']
compatibility = cache_source['metadata']['compatibility']
model = cfg['models'][0]
for key, value in {
    'runtime_id': source['id'], 'model_repository': model['repository'],
    'model_revision': model['revision'], 'model_archive_sha256': model['archive_sha256'],
    'gpu': model['caches'][0]['gpu'], 'gpu_compute_capability': '8.9',
}.items():
    assert compatibility[key] == value, 'source cache identity mismatch: ' + key
runtime = cfg['bootstrap']['runtime']
for key, value in runtime['compatibility'].items():
    assert source['metadata']['compatibility'][key] == value, 'source runtime identity mismatch: ' + key
for key, value in model['caches'][0]['compatibility'].items():
    assert compatibility[key] == value, 'source cache label mismatch: ' + key
launch = [runtime['program'], runtime['args'], dict(sorted(runtime['environment'].items()))]
source_launch_fingerprint = hashlib.sha256(json.dumps(launch, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
assert compatibility['runtime_launch_configuration'] == source_launch_fingerprint, 'source launch configuration changed'
assert runtime['environment']['FLASHINFER_DISABLE_JIT'] == '1'
assert runtime['environment']['VLLM_FORCE_AOT_LOAD'] == '1'
if a.validate_only:
    print(json.dumps({'source_configuration_compatible': True, 'source_launch_fingerprint': source_launch_fingerprint,
                      'engine_fingerprint': engine_fingerprint, 'no_files_changed': True}))
    raise SystemExit(0)
for artifact, tree in [(a.runtime_artifact, a.runtime_tree), (a.cache_artifact, a.cache_tree)]:
    subprocess.run([cli, 'artifacts', 'verify', '--manifest', str(artifact / 'manifest.json'),
                    '--directory', str(tree)], check=True)
a.out.mkdir(parents=True)
rtree = a.out / 'runtime'
ctree = a.out / 'cache'
for src, dst in [(a.runtime_tree, rtree), (a.cache_tree, ctree)]:
    subprocess.run(['cp', '--reflink=auto', '-a', str(src), str(dst)], check=True)
removed = []
for entry in source['entries']:
    path = entry['path']
    if '/flashinfer_cubin/cubins/' in path and path.endswith('.cubin'):
        assert entry['kind'] == 'file'
        removed.append(entry)
        (rtree / path).unlink()
assert len(removed) == 42704 and sum(e['size'] for e in removed) == 6546167652
profile_name = 'qwen3.5-0.8b-bf16-sm89-flashattn'
selection = {
    'schema_version': 1, 'profile': profile_name, 'source_runtime_id': source['id'],
    'scope': {'model': 'Qwen/Qwen3.5-0.8B', 'dtype': 'bfloat16', 'gpu_compute_capability': '8.9',
              'attention_backend': 'FLASH_ATTN', 'speculative_decoding': False},
    'selection_rule': 'remove only universal flashinfer_cubin/cubins/*.cubin payloads; native sampling provider retained in cache',
    'removed_files': len(removed), 'removed_bytes': sum(e['size'] for e in removed),
    'removed_inventory_sha256': hashlib.sha256(json.dumps(removed, separators=(',', ':')).encode()).hexdigest(),
    'retained': ['package code and metadata', 'headers', 'checksums', 'directory paths', 'native libraries', 'compiler toolchain'],
    'gpu_acceptance_required': True,
}
(rtree / 'asset-profile.json').write_text(json.dumps(selection, indent=2) + '\n')
metadata = copy.deepcopy(source['metadata'])
metadata['compatibility']['asset_profile'] = profile_name
metadata['attributes'].update({'source_runtime_id': source['id'], 'asset_profile': profile_name,
                               'removed_cubin_bytes': str(selection['removed_bytes'])})
(rtree / 'artifact-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
info = json.loads((rtree / 'build-info.json').read_text()); info['asset_profile'] = selection
(rtree / 'build-info.json').write_text(json.dumps(info, indent=2) + '\n')
(a.out / 'runtime-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
subprocess.run([cli, 'artifacts', 'pack', '--source', str(rtree), '--out', str(a.out / 'runtime-artifact'),
                '--metadata', str(a.out / 'runtime-metadata.json')], check=True)
derived = json.loads((a.out / 'runtime-artifact/manifest.json').read_text())
actual = {e['path']: e for e in derived['entries']}
removed_paths = {e['path'] for e in removed}
for entry in source['entries']:
    path = entry['path']
    if path not in removed_paths and path not in {'artifact-metadata.json', 'build-info.json'}:
        assert actual[path] == entry, 'profile changed an unselected runtime entry: ' + path
runtime = cfg['bootstrap']['runtime']
runtime['artifact'] = {'id': derived['id'], 'prefix': 'switchyard/serving-comparison/runtime/' + derived['id']}
runtime['compatibility'] = derived['metadata']['compatibility']
# Compiled Torch/Triton files may contain the original cache prefix. Preserve it
# with a fresh-host alias; never overwrite another directory or bypass identity.
original_prefix = '/opt/switchyard/cache/' + source['id']
for key, value in list(runtime['environment'].items()):
    runtime['environment'][key] = value.replace('{cache_root}', original_prefix)
runtime['environment']['SERVING_PROFILE_CACHE_ROOT'] = '{cache_root}'
runtime['environment']['FLASHINFER_NO_DOWNLOAD'] = '1'
runtime['environment'].pop('FLASHINFER_CUBIN_CHECKSUM_DISABLED', None)
program = runtime['args'][1]
query = "subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader,nounits'], check=True)"
assert query in program
program = program.replace(query,
    "gpu_info=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader,nounits'], text=True).strip()\n"
    "print(gpu_info,flush=True)\n"
    "if len(gpu_info.splitlines()) != 1 or gpu_info.rsplit(',',1)[-1].strip() != '8.9':\n"
    "    raise SystemExit('runtime asset profile requires exactly one SM89 GPU')")
alias = (
    'import os,pathlib\n'
    'actual_cache=pathlib.Path(os.environ["SERVING_PROFILE_CACHE_ROOT"]).resolve()\n'
    f'original_cache=pathlib.Path({original_prefix!r})\n'
    'if original_cache.exists() or original_cache.is_symlink():\n'
    '    if original_cache.resolve() != actual_cache:\n'
    '        raise SystemExit("compiler cache prefix alias already belongs to another tree")\n'
    'else:\n'
    '    original_cache.symlink_to(actual_cache,target_is_directory=True)\n'
)
if a.profile:
    code = Path('scripts/serving-comparison/startup-profile.py').read_text()
    alias += (
        'profile_dir=pathlib.Path(os.environ["VLLM_CACHE_ROOT"]).parent/"startup-profile"\n'
        'profile_dir.mkdir(exist_ok=True)\n'
        f'(profile_dir/"sitecustomize.py").write_text({code!r})\n'
        'os.environ["PYTHONPATH"]=str(profile_dir)+os.pathsep+os.environ.get("PYTHONPATH","")\n'
        'import sys\n'
        'sys.path.insert(0,str(profile_dir))\n'
        'import sitecustomize\n'
    )
runtime['args'][1] = alias + program
launch = [runtime['program'], runtime['args'], dict(sorted(runtime['environment'].items()))]
launch_fingerprint = hashlib.sha256(json.dumps(launch, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
cmeta = copy.deepcopy(cache_source['metadata'])
cmeta['compatibility']['runtime_id'] = derived['id']
cmeta['compatibility']['runtime_launch_configuration'] = launch_fingerprint
cmeta['attributes'].update({'source_cache_id': cache_source['id'], 'source_runtime_id': source['id'],
                           'asset_profile': profile_name, 'original_cache_prefix_alias': original_prefix,
                           'derivation': 'unchanged compiler entries and original cache prefix; only unused cubin payloads removed from runtime'})
(a.out / 'cache-metadata.json').write_text(json.dumps(cmeta, indent=2) + '\n')
subprocess.run([cli, 'artifacts', 'pack', '--source', str(ctree), '--out', str(a.out / 'cache-artifact'),
                '--metadata', str(a.out / 'cache-metadata.json')], check=True)
cache = json.loads((a.out / 'cache-artifact/manifest.json').read_text())
assert cache['entries'] == cache_source['entries']
cfg['models'][0]['caches'][0].update({'compatibility': cache['metadata']['compatibility'],
    'artifact': {'id': cache['id'], 'prefix': 'switchyard/serving-comparison/cache/' + cache['id']}})
(a.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
(a.out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
print(json.dumps({'runtime_id': derived['id'], 'cache_id': cache['id'], 'removed_bytes': selection['removed_bytes'],
                  'config': str(a.out / 'config.json'), 'gpu_acceptance_required': True}))
