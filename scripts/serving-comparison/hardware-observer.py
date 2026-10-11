"""Read GPU and owned process statistics in a separate stdlib-only process.

This observer never imports Torch, initializes CUDA, changes clocks, or starts
threads in the API process. Its bounded measurements share the owned engine's
stdout. The serving launcher is the only process-tree root inspected.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

owner=int(sys.argv[1]);interval=float(sys.argv[2]);maximum_s=float(sys.argv[3])
started=time.monotonic();observer=os.getpid()
GPU_FIELDS=['uuid','driver_version','pstate','clocks.current.sm','clocks.current.memory',
            'power.draw','enforced.power.limit','utilization.gpu','temperature.gpu','memory.used']


def text_file(path):
    try:return Path(path).read_text().strip()
    except OSError:return None


def process_snapshot(pid):
    try:
        raw=Path('/proc',str(pid),'stat').read_text();end=raw.rfind(')')
        fields=raw[end+2:].split();status={}
        for line in Path('/proc',str(pid),'status').read_text().splitlines():
            if ':' in line:
                key,value=line.split(':',1)
                if key in {'Threads','Cpus_allowed_list','Mems_allowed_list'}:status[key]=value.strip()
        return {'pid':pid,'name':raw[raw.find('(')+1:end],'state':fields[0],
                'user_ticks':int(fields[11]),'system_ticks':int(fields[12]),
                'affinity_cpus':len(os.sched_getaffinity(pid)),**status}
    except (OSError,ValueError,IndexError):return None


def process_tree():
    queue=[owner];seen=set();rows=[]
    while queue and len(seen)<32:
        pid=queue.pop(0)
        if pid in seen or pid==observer:continue
        seen.add(pid);row=process_snapshot(pid)
        if row is None:continue
        rows.append(row)
        children=text_file('/proc/'+str(pid)+'/task/'+str(pid)+'/children')
        if children:
            queue.extend(int(x) for x in children.split())
    return rows


def emit(**values):
    print('SWITCHYARD_HARDWARE_OBSERVER '+json.dumps({'observer_pid':observer,
          'owner_pid':owner,'monotonic_s':time.monotonic(),'unix_s':time.time(),**values},sort_keys=True),flush=True)


cpu_model=None
for line in (text_file('/proc/cpuinfo') or '').splitlines():
    if line.startswith('model name'):
        cpu_model=line.split(':',1)[1].strip();break
emit(event='inventory',cpu_model=cpu_model,logical_cpu_count=os.cpu_count(),
     affinity_cpus=len(os.sched_getaffinity(0)),clock_ticks_per_second=os.sysconf('SC_CLK_TCK'),
     cpu_quota=text_file('/sys/fs/cgroup/cpu.max'),
     effective_cpuset=text_file('/sys/fs/cgroup/cpuset.cpus.effective'),
     thread_environment={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'] if k in os.environ},
     interval_s=interval,maximum_s=maximum_s)
previous={}
while time.monotonic()-started<maximum_s:
    rows=process_tree()
    root=next((r for r in rows if r['pid']==owner),None)
    if root is None or root['state']=='Z':break
    at=time.monotonic()
    for row in rows:
        ticks=row['user_ticks']+row['system_ticks'];prior=previous.get(row['pid'])
        if prior and at>prior[0]:
            row['cpu_core_equivalents']=(ticks-prior[1])/os.sysconf('SC_CLK_TCK')/(at-prior[0])
        previous[row['pid']]=(at,ticks)
    began=time.monotonic()
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu='+','.join(GPU_FIELDS),
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        gpu=[dict(zip(GPU_FIELDS,(v.strip() for v in row)))
             for row in csv.reader(result.stdout.splitlines())] if result.returncode==0 else []
        emit(event='sample',processes=rows,gpu=gpu,gpu_query_returncode=result.returncode,
             gpu_query_duration_s=time.monotonic()-began)
    except (OSError,subprocess.TimeoutExpired) as error:
        emit(event='sample',processes=rows,gpu=[],gpu_query_error_class=type(error).__name__,
             gpu_query_duration_s=time.monotonic()-began)
    time.sleep(max(0,interval-(time.monotonic()-began)))
emit(event='finished')
