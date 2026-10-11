"""Add a bounded hardware observer while preserving native compiler entries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',required=True,type=Path)
p.add_argument('--cache-artifact',required=True,type=Path)
p.add_argument('--out',required=True,type=Path)
p.add_argument('--cli',default='target/release/switchyard')
a=p.parse_args()
assert not a.out.exists()
cfg=json.loads(a.config.read_text());assert len(cfg['models'])==1
runtime=cfg['bootstrap']['runtime'];cache=cfg['models'][0]['caches'][0]
source=json.loads((a.cache_artifact/'manifest.json').read_text())
def fingerprint():
    return hashlib.sha256(json.dumps([runtime['program'],runtime['args'],dict(sorted(runtime['environment'].items()))],separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
assert source['id']==cache['artifact']['id']
assert source['metadata']['compatibility']['runtime_id']==runtime['artifact']['id']
assert source['metadata']['compatibility']['runtime_launch_configuration']==fingerprint()
assert runtime['environment']['VLLM_FORCE_AOT_LOAD']==runtime['environment']['FLASHINFER_DISABLE_JIT']=='1'
assert runtime['args'][0]=='-c'
code=Path('scripts/serving-comparison/hardware-observer.py').read_text();digest=hashlib.sha256(code.encode()).hexdigest()
prefix=('import os,pathlib,subprocess,sys\n'
        'observer_file=pathlib.Path("/tmp")/("switchyard-hardware-observer-%d.py"%os.getpid())\n'
        f'observer_file.write_text({code!r})\n'
        'observer=subprocess.Popen([sys.executable,"-S","-u",str(observer_file),str(os.getpid()),"2.0","180"],'
        'stdin=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)\n')
runtime['args'][1]=prefix+runtime['args'][1]
meta=copy.deepcopy(source['metadata']);meta['compatibility']['runtime_launch_configuration']=fingerprint()
meta['attributes'].update({'source_cache_id':source['id'],'hardware_observer_sha256':digest,
                          'hardware_observer':'separate stdlib process; 2-second GPU and owned process observations; original compiler entries unchanged'})
a.out.mkdir();tree=a.out/'tree';tree.mkdir()
for chunk in source['chunks']:
    subprocess.run(['tar','--zstd','-xf',str(a.cache_artifact/chunk['name']),'-C',str(tree)],check=True)
subprocess.run([a.cli,'artifacts','verify','--manifest',str(a.cache_artifact/'manifest.json'),'--directory',str(tree)],check=True)
(a.out/'cache-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
subprocess.run([a.cli,'artifacts','pack','--source',str(tree),'--out',str(a.out/'cache-artifact'),'--metadata',str(a.out/'cache-metadata.json')],check=True)
derived=json.loads((a.out/'cache-artifact/manifest.json').read_text());assert derived['entries']==source['entries']
cache.update({'compatibility':derived['metadata']['compatibility'],'artifact':{'id':derived['id'],'prefix':'switchyard/serving-comparison/cache/'+derived['id']}})
(a.out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps({'cache_id':derived['id'],'runtime_id':runtime['artifact']['id'],'launch_fingerprint':fingerprint(),'observer_sha256':digest,'compiler_entries_unchanged':True}))
