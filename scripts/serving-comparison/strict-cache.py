"""Prepare a provenance-linked cache for vLLM's fail-on-AOT-miss experiment.

Original cache files remain unchanged. The launch compatibility label changes:
VLLM_FORCE_AOT_LOAD is explicitly ignored by vLLM 0.31 compile_factors().
An optional provider adds the pinned upstream shim and registers the baked
sampling binary through FlashInfer's native AOT provider API, with JIT disabled.
The normal immutable bake artifact is preserved. This script neither provisions
a GPU nor publishes; use the artifact CLI to publish the verified derivative.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', required=True, type=Path)
p.add_argument('--bake', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
p.add_argument('--profile', action='store_true', help='Time native calls with a diagnostic sitecustomize')
p.add_argument('--worker-method', choices=['fork', 'spawn'],
               help='Use the native vLLM worker policy; its CUDA-initialization safety guard remains active')
p.add_argument('--sampling-provider-shim', type=Path,
               help='Pinned official FlashInfer shim wheel; promote the baked sampling binary through its provider API')
a = p.parse_args()
if a.out.exists():
    raise SystemExit('fresh derivative output required')
cfg = json.loads(a.config.read_text())
assert cfg['bootstrap']['runtime']['compatibility']['vllm'] == '0.31.0+cu129'
source = json.loads((a.bake / 'cache-artifact/manifest.json').read_text())
assert source['metadata']['kind'] == 'cache'
cache = cfg['models'][0]['caches'][0]
assert cache['artifact']['id'] == source['id']
runtime = cfg['bootstrap']['runtime']

def fingerprint(runtime):
    value = [runtime['program'], runtime['args'], dict(sorted(runtime['environment'].items()))]
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

assert fingerprint(runtime) == source['metadata']['compatibility']['runtime_launch_configuration']
runtime['environment']['VLLM_FORCE_AOT_LOAD'] = '1'
if a.worker_method:
    runtime['environment']['VLLM_WORKER_MULTIPROC_METHOD'] = a.worker_method
if a.sampling_provider_shim:
    assert hashlib.file_digest(a.sampling_provider_shim.open('rb'), 'sha256').hexdigest() == (
        '314d97ea8c3d288d0afce09e9f3dc823a83d4a219c02b85097de71ee143cf0fb')
    runtime['environment']['PYTHONPATH'] = '{cache_root}/compiled/sampling-provider'
    runtime['environment']['FLASHINFER_DISABLE_JIT'] = '1'
if a.profile:
    source_code = Path('scripts/serving-comparison/startup-profile.py').read_text()
    prefix = ('import os,pathlib\n'
              'profile_dir=pathlib.Path(os.environ["VLLM_CACHE_ROOT"]).parent/"startup-profile"\n'
              'profile_dir.mkdir(exist_ok=True)\n'
              f'(profile_dir/"sitecustomize.py").write_text({source_code!r})\n'
              'os.environ["PYTHONPATH"]=str(profile_dir)+os.pathsep+os.environ.get("PYTHONPATH","")\n'
              'import sys\n'
              'sys.path.insert(0,str(profile_dir))\n'
              'import sitecustomize\n')
    assert runtime['args'][0] == '-c'
    runtime['args'][1] = prefix + runtime['args'][1]
metadata = source['metadata']
metadata['compatibility']['runtime_launch_configuration'] = fingerprint(runtime)
metadata['attributes']['source_cache_id'] = source['id']
metadata['attributes']['derivation'] = 'original cache entries unchanged; native vLLM 0.31 AOT controls'
if a.worker_method:
    metadata['attributes']['worker_method'] = a.worker_method + '; native CUDA initialization safety guard retained'
if a.profile:
    metadata['attributes']['diagnostic_profile'] = 'timing wrappers only; same native function arguments and results'
if a.sampling_provider_shim:
    metadata['attributes']['sampling_provider'] = 'native FlashInfer 0.7 provider API; baked sm89 sampling binary; JIT disabled'
    metadata['attributes']['flashinfer_shim_sha256'] = '314d97ea8c3d288d0afce09e9f3dc823a83d4a219c02b85097de71ee143cf0fb'
a.out.mkdir(parents=True)
tree = a.out / 'tree'
tree.mkdir()
# The source artifact is pinned and was verified during the original bake.
# Extract only these local, previously verified shards, then recheck every
# file, mode, and link against the original manifest before repacking.
for shard in source['chunks']:
    subprocess.run(['tar', '--zstd', '-xf', str(a.bake / 'cache-artifact' / shard['name']),
                    '-C', str(tree)], check=True)
subprocess.run(['target/release/switchyard', 'artifacts', 'verify', '--manifest',
                str(a.bake / 'cache-artifact/manifest.json'), '--directory', str(tree)], check=True)
if a.sampling_provider_shim:
    # Keep the compiled module in its original tree; the provider only registers
    # its pinned architecture, version, and location with the upstream shim.
    assert source['metadata']['compatibility']['gpu_compute_capability'] == '8.9'
    assert (tree / 'flashinfer/.cache/flashinfer/0.7.0.post1/89/cached_ops/sampling/sampling.so').is_file()
    provider = tree / 'sampling-provider'
    provider.mkdir()
    with zipfile.ZipFile(a.sampling_provider_shim) as wheel:
        for entry in wheel.namelist():
            if entry.endswith('/'):
                continue
            relative = entry.split('.data/purelib/', 1)[-1]
            assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
            dest = provider / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(wheel.read(entry))
    (provider / 'switchyard_flashinfer_sampling.py').write_text(
        'from pathlib import Path\n'
        'def get_provider():\n'
        '    root=Path(__file__).resolve().parent.parent\n'
        '    return {"schema_version":1,"distribution":"switchyard-flashinfer-sampling",'
        '"version":"0.7.0.post1+cu129","jit_cache_dir":'
        'root/"flashinfer/.cache/flashinfer/0.7.0.post1/89/cached_ops",'
        '"cuda_architectures":["sm89"],"modules":["sampling"]}\n')
    info = provider / 'switchyard_flashinfer_sampling-0.7.0.post1+cu129.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: switchyard-flashinfer-sampling\nVersion: 0.7.0.post1+cu129\n')
    (info / 'entry_points.txt').write_text(
        '[flashinfer.jit_cache.providers]\n'
        'switchyard-sm89-sampling = switchyard_flashinfer_sampling:get_provider\n')
meta = a.out / 'metadata.json'
meta.write_text(json.dumps(metadata, indent=2) + '\n')
subprocess.run(['target/release/switchyard', 'artifacts', 'pack', '--source', str(tree),
                '--out', str(a.out / 'artifact'), '--metadata', str(meta)], check=True)
derived = json.loads((a.out / 'artifact/manifest.json').read_text())
original = {entry['path']: entry for entry in source['entries']}
actual = {entry['path']: entry for entry in derived['entries']}
assert all(actual[path] == entry for path, entry in original.items()), 'derivative changed original file content or modes'
cache['compatibility'] = derived['metadata']['compatibility']
cache['artifact'] = {'id': derived['id'], 'prefix': 'switchyard/serving-comparison/cache/' + derived['id']}
(a.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
print(json.dumps({'source_cache': source['id'], 'strict_cache': derived['id'],
                  'original_entries_unchanged': True, 'config': str(a.out / 'config.json')}))
