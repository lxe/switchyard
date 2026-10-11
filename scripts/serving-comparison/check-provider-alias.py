"""Reproduce native FlashInfer provider discovery across a late alias and fork."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime-tree',required=True,type=Path)
p.add_argument('--cache-tree',required=True,type=Path)
p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
code='''import importlib.util, json, multiprocessing, os, pathlib, subprocess, sys
provider_path='/opt/switchyard/cache/late/sampling-provider'
negative_before=provider_path in sys.path_importer_cache and sys.path_importer_cache[provider_path] is None
pathlib.Path('/opt/switchyard/cache/late').symlink_to('/cache-source')
visible_before=importlib.util.find_spec('flashinfer_jit_cache') is not None
if os.environ['ALIAS_INVALIDATE']=='1':importlib.invalidate_caches()
visible_after=importlib.util.find_spec('flashinfer_jit_cache') is not None
from flashinfer.jit import env
import torch
parent_count=len(env.FLASHINFER_AOT_PROVIDERS)
context=multiprocessing.get_context('fork')
receiver,sender=context.Pipe(duplex=False)
def child():sender.send(len(env.FLASHINFER_AOT_PROVIDERS));sender.close()
process=context.Process(target=child);process.start();fork_count=receiver.recv();process.join();assert process.exitcode==0
fresh_code="from flashinfer.jit import env; print(len(env.FLASHINFER_AOT_PROVIDERS))"
fresh_count=int(subprocess.check_output([sys.executable,'-c',fresh_code],text=True).strip())
print(json.dumps({'importer_cached_missing_before_alias':negative_before,'shim_visible_before_invalidation':visible_before,
'shim_visible_after_optional_invalidation':visible_after,'native_parent_provider_count':parent_count,
'native_fork_provider_count':fork_count,'native_fresh_interpreter_provider_count':fresh_count,
'native_aot_sampling_artifact_count':len(env.get_aot_artifacts('sampling')),
'cuda_initialized':torch.cuda.is_initialized(),'invalidation_enabled':os.environ['ALIAS_INVALIDATE']=='1'}))
'''
rows=[]
for invalidate in ['0','1']:
    cmd=['docker','run','--rm','--network','none','--read-only','--cpus','2','--memory','3g',
         '--tmpfs','/tmp:rw,exec,size=1g','--tmpfs','/opt/switchyard/cache:rw,size=1m',
         '--mount','type=bind,source='+str(a.runtime_tree.resolve())+',target=/opt/switchyard/runtime,readonly',
         '--mount','type=bind,source='+str(a.cache_tree.resolve())+',target=/cache-source,readonly',
         '--env','PYTHONPATH=/opt/switchyard/cache/late/sampling-provider',
         '--env','FLASHINFER_NO_DOWNLOAD=1','--env','FLASHINFER_DISABLE_JIT=1',
         '--env','FLASHINFER_CUDA_ARCH_LIST=8.9','--env','ALIAS_INVALIDATE='+invalidate,
         'switchyard-hydration-smoke:local','/bin/bash','-c',
         'source /opt/switchyard/runtime/runtime-env.sh && export PYTHONPATH=/opt/switchyard/cache/late/sampling-provider && exec python -c '+shlex.quote(code)]
    result=subprocess.run(cmd,capture_output=True,text=True,check=True)
    row=json.loads(result.stdout.splitlines()[-1]);count=int(invalidate)
    assert row['importer_cached_missing_before_alias'] and not row['shim_visible_before_invalidation']
    assert row['native_parent_provider_count']==row['native_fork_provider_count']==row['native_aot_sampling_artifact_count']==count
    assert row['native_fresh_interpreter_provider_count']==1 and not row['cuda_initialized']
    rows.append(row)
a.out.write_text(json.dumps({'gpu_execution_tested':False,'native_flashinfer_version':'0.7.0.post1','cases':rows},indent=2)+'\n')
print(json.dumps({'cases_passed':len(rows),'out':str(a.out)}))
