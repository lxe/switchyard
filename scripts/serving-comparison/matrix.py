"""Run remaining matched experiments sequentially, stopping on any failed guard."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--provider-env',required=True)
p.add_argument('--r2-env',required=True)
p.add_argument('--ledger',default='results/comparison-budget.json')
p.add_argument('--vllm-bake',required=True,type=Path)
p.add_argument('--sglang-bake',default='results/cloud-sglang-0.8b-bake',type=Path)
p.add_argument('--suffix',default='mo',help='Fresh placement/run suffix; failed outputs remain immutable')
a=p.parse_args()
base=Path('configs/runpod/serving-comparison')

def guard(kind,out,operation,extra):
 cfg=base/(kind+'.json')
 command=[sys.executable,'scripts/serving-comparison/run.py','--out',str(out),'--config',str(cfg),
          '--provider-env',a.provider_env,'--ledger',a.ledger,'--reservation','1.0','--',
          'target/release/switchyard','live',operation,'--config',str(cfg),
          '--env-file',a.provider_env,'--env-file',a.r2_env,'--out',str(out),*extra]
 print(json.dumps({'starting':str(out)}),flush=True)
 subprocess.run(command,check=True)

def ready(out):
 receipt=json.loads((out/'independent-receipt.json').read_text())
 assert receipt['controller_exit']==0 and receipt['all_absent'] and not receipt['reason']

ready(a.vllm_bake)
# Actual SGLang GPU startup is required before this script can provision it.
assert any('"engine-passed"' in line for line in Path('results/local-sglang-gpu.log').read_text().splitlines())
if a.sglang_bake.exists():
 ready(a.sglang_bake)
else:
 guard('sglang',a.sglang_bake,'bake',['--plan',str(base/'sglang-bake.json'),
                                 '--namespace','switchyard/serving-comparison/cache'])
for kind,out in [('vllm',a.vllm_bake),('sglang',a.sglang_bake)]:
 ready(out)
 path=base/(kind+'.json');cfg=json.loads(path.read_text())
 cfg['models'][0]['caches']=[json.loads((out/'cache-configuration.json').read_text())]
 path.write_text(json.dumps(cfg,indent=2)+'\n')
for model in ['0.8b','2b','4b']:
 for kind in ['vllm','sglang']:
  out=Path('results')/f'cloud-{kind}-{model}-full-{a.suffix}'
  if out.exists():
   ready(out)
   continue
  guard(kind,out,'gauntlet',['--workloads',str(base/'workloads.json'),'--model','qwen3.5-'+model])
