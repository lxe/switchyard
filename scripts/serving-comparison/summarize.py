"""Derive safe case metrics, startup milestones, literal checks, and rental totals."""
import argparse
import hashlib
import json
import math
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--ledger',type=Path,default=Path('results/comparison-budget.json'))
p.add_argument('--out',required=True,type=Path)
a=p.parse_args()

def percentile(values,q):
 values=sorted(values)
 return values[max(0,min(len(values)-1,math.ceil(len(values)*q)-1))] if values else None

ledger=json.loads(a.ledger.read_text());rows=[]
expected=hashlib.sha256(b'POCKET_MODEL_OK').hexdigest()
for run in ledger['runs']:
 root=Path(run['run']);row={k:run[k] for k in ['run','controller_exit','estimated_rental_usd','all_absent','cleanup','reason']}
 row['cases']=[]
 states=[]
 statefile=root/'agent-state.ndjson'
 if statefile.exists():states=[json.loads(s)['state'] for s in statefile.read_text().splitlines()]
 if states:
  state=states[-1];row['startup_milestones']=state['startup_milestones'];row['hydration_mode']=state['hydration_mode'];row['artifact_bytes_total']=state['artifact_bytes_total'];row['model_bytes_total']=state['bytes_total']
 for file in sorted(root.rglob('workloads/*.json')):
  report=json.loads(file.read_text())
  if not isinstance(report,dict) or 'completed' not in report:continue
  item={k:report[k] for k in ['name','requests','completed','failed','unfinished','budget_exceeded','invalid_actions','output_tokens','elapsed_s','output_tokens_per_s','ttft_p50_s','ttft_p95_s','response_p95_s']}
  completions=report['completions'];ttft=[c['first_content_s']-c['started_s'] for c in completions]
  item['service_ttft_p50_s']=percentile(ttft,.5);item['service_ttft_p95_s']=percentile(ttft,.95)
  cached=[c['cached_input_tokens'] for c in completions if c.get('cached_input_tokens') is not None]
  item['cached_input_tokens']=sum(cached) if cached else None
  item['input_tokens']=sum(c['input_tokens'] for c in completions)
  if report['name'] in ['cold-single','hot-single']:
   item['literal_correct']=sum(c.get('content_trimmed_sha256')==expected for c in completions)
  row['cases'].append(item)
 rows.append(row)
out={'estimated_total_rental_usd':sum(r['estimated_rental_usd'] for r in rows),'all_owned_pods_absent':all(r['all_absent'] for r in rows),'runs':rows}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['estimated_total_rental_usd','all_owned_pods_absent']}))
