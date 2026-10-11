"""Prepare a provenance-linked cache for vLLM's fail-on-AOT-miss experiment.

The cache's bytes are unchanged. Only its launch compatibility label changes:
VLLM_FORCE_AOT_LOAD is explicitly ignored by vLLM 0.31 compile_factors().
The normal immutable bake artifact is preserved. This script neither provisions
a GPU nor publishes; use the artifact CLI to publish the verified derivative.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', required=True, type=Path)
p.add_argument('--bake', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
p.add_argument('--profile', action='store_true', help='Time native calls with a diagnostic sitecustomize')
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
if a.profile:
    source_code = Path('scripts/serving-comparison/startup-profile.py').read_text()
    prefix = ('import os,pathlib\n'
              'profile_dir=pathlib.Path(os.environ["VLLM_CACHE_ROOT"]).parent/"startup-profile"\n'
              'profile_dir.mkdir(exist_ok=True)\n'
              f'(profile_dir/"sitecustomize.py").write_text({source_code!r})\n'
              'os.environ["PYTHONPATH"]=str(profile_dir)\n'
              'import sys\n'
              'sys.path.insert(0,str(profile_dir))\n'
              'import sitecustomize\n')
    assert runtime['args'][0] == '-c'
    runtime['args'][1] = prefix + runtime['args'][1]
metadata = source['metadata']
metadata['compatibility']['runtime_launch_configuration'] = fingerprint(runtime)
metadata['attributes']['source_cache_id'] = source['id']
metadata['attributes']['derivation'] = 'identical files; vLLM 0.31 ignores VLLM_FORCE_AOT_LOAD in compiler hash'
if a.profile:
    metadata['attributes']['diagnostic_profile'] = 'timing wrappers only; same native function arguments and results'
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
meta = a.out / 'metadata.json'
meta.write_text(json.dumps(metadata, indent=2) + '\n')
subprocess.run(['target/release/switchyard', 'artifacts', 'pack', '--source', str(tree),
                '--out', str(a.out / 'artifact'), '--metadata', str(meta)], check=True)
derived = json.loads((a.out / 'artifact/manifest.json').read_text())
assert derived['entries'] == source['entries'], 'derivative changed file content or modes'
cache['compatibility'] = derived['metadata']['compatibility']
cache['artifact'] = {'id': derived['id'], 'prefix': 'switchyard/serving-comparison/cache/' + derived['id']}
(a.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
print(json.dumps({'source_cache': source['id'], 'strict_cache': derived['id'],
                  'entries_unchanged': True, 'config': str(a.out / 'config.json')}))
