"""Exercise pinned FLA functions and Torch's native NVML visibility parser."""
import ast
import ctypes
import functools
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

spec_path = Path('deploy/hydration/patches/vllm-0.31.0-fla-nvml.json')
spec = json.loads(spec_path.read_text())
base = Path('results/vllm-export/runtime')
source = (base / spec['target']).read_text()
assert hashlib.sha256(source.encode()).hexdigest() == spec['before_sha256']
patched = source
for replacement in spec['replacements']:
    assert patched.count(replacement['before']) == 1
    patched = patched.replace(replacement['before'], replacement['after'])
assert hashlib.sha256(patched.encode()).hexdigest() == spec['after_sha256']
torch_source = (base / 'venv/lib/python3.12/site-packages/torch/cuda/__init__.py').read_text()
torch_functions = {'_parse_visible_devices', '_transform_uuid_to_ordinals',
                   '_device_count_nvml', '_get_nvml_device_index'}

class NvmlError(Exception): pass

class Probe:
    def __init__(self, *, cuda=True, visible=1, initialized=False, name='NVIDIA L40S', major=8,
                 nvml=True, physical=None, visibility=None, current_device=0,
                 driver_status=3, driver_missing=False, query_failure=False, cuda_order=None):
        self.cuda=cuda; self.initialized=initialized; self.nvml=nvml
        self.current_device=current_device; self.driver_status=driver_status
        self.driver_missing=driver_missing; self.query_failure=query_failure
        self.physical=physical or {0:(name,major)}
        uuids=['GPU-'+str(i)+'-full-uuid' for i in range(max(self.physical)+1)]
        self.target_calls=0; self.torch_calls=0; self.nvml_calls=0; self.nvml_contexts=0
        self.driver_calls=0
        namespace={'os':SimpleNamespace(getenv=lambda key: visibility if key=='CUDA_VISIBLE_DEVICES' else None),
                   'torch':SimpleNamespace(version=SimpleNamespace(hip=False)), 'cast':cast,
                   'Device':object, '_raw_device_count_nvml':lambda:len(uuids) if visible else 0,
                   '_raw_device_uuid_nvml':lambda:uuids,
                   '_get_device_index':lambda device,optional:device}
        definitions=[n for n in ast.parse(torch_source).body
                     if isinstance(n,ast.FunctionDef) and n.name in torch_functions]
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'pinned-torch-cuda-functions','exec'),namespace)
        self.parser=namespace
        self.visible_ids=namespace['_parse_visible_devices']()
        if self.visible_ids and isinstance(self.visible_ids[0],str):
            self.visible_ids=namespace['_transform_uuid_to_ordinals'](self.visible_ids,uuids)
        self.visible_ids=[i for i in self.visible_ids if i < (len(uuids) if visible else 0)]
        if cuda_order and (not visibility or not visibility.startswith('GPU-')):
            self.visible_ids=[cuda_order[i] for i in self.visible_ids]
    def is_initialized(self):return self.initialized
    def _device_count_nvml(self):return self.parser['_device_count_nvml']() if self.nvml else -1
    def _get_nvml_device_index(self,index):return self.parser['_get_nvml_device_index'](index)
    def _parse_visible_devices(self):return self.parser['_parse_visible_devices']()
    def _raw_device_count_nvml(self):return self.parser['_raw_device_count_nvml']()
    def get_device_name(self,index=0):
        self.torch_calls+=1; self.initialized=True
        return self.physical[self.visible_ids[index]][0]
    def get_device_capability(self):
        self.torch_calls+=1; self.initialized=True
        return self.physical[self.visible_ids[self.current_device]][1],0
    def get_current_target(self):
        self.target_calls+=1
        if not self.visible_ids:raise RuntimeError('no active target')
        if self.cuda:self.initialized=True
        return SimpleNamespace(backend='cuda' if self.cuda else 'hip')
    def cuCtxGetCurrent(self,pointer):
        self.driver_calls+=1
        return self.driver_status
    def library(self,name):
        assert name=='libcuda.so.1'
        if self.driver_missing:raise OSError('driver absent')
        return self
    def handle(self,index):
        assert self.nvml_contexts==1
        self.nvml_calls+=1
        if self.query_failure:raise NvmlError('query failed')
        return index
    def context(self,function):
        def wrapped(*args,**kwargs):
            self.nvml_contexts+=1
            try:return function(*args,**kwargs)
            finally:self.nvml_contexts-=1
        return wrapped

def evaluate(code,probe,is_patched):
    saved={name:sys.modules.get(name) for name in ['ctypes','vllm.platforms.cuda']}
    sys.modules['ctypes']=SimpleNamespace(CDLL=probe.library,c_void_p=ctypes.c_void_p,byref=ctypes.byref)
    sys.modules['vllm.platforms.cuda']=SimpleNamespace(with_nvml_context=probe.context,
        pynvml=SimpleNamespace(NVMLError=NvmlError,nvmlDeviceGetHandleByIndex=probe.handle,
            nvmlDeviceGetName=lambda index:probe.physical[index][0],
            nvmlDeviceGetCudaComputeCapability=lambda index:(probe.physical[index][1],0)))
    namespace={'functools':functools,'torch':SimpleNamespace(cuda=probe),
               'current_platform':SimpleNamespace(is_cuda=lambda:probe.cuda),
               'triton':SimpleNamespace(runtime=SimpleNamespace(driver=SimpleNamespace(active=probe))),
               'Literal':__import__('typing').Literal}
    try:
        tree=ast.parse(code)
        definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                     and n.name in {'get_available_device','_check_platform','_is_nvidia_hopper','_can_use_nvml_platform_query'}]
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'pinned-fla-functions','exec'),namespace)
        platform=namespace['_check_platform']();namespace['is_nvidia']=platform=='nvidia'
        if is_patched:hopper=namespace['_is_nvidia_hopper']()
        else:
            assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
                            and any(isinstance(t,ast.Name) and t.id=='is_nvidia_hopper' for t in n.targets))
            hopper=eval(compile(ast.Expression(assignment.value),'pinned-fla-expression','eval'),namespace)
        assert probe.nvml_contexts==0
        return platform,hopper
    finally:
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value

cases=[('nvml_sm89',{},True),('nvml_h100',{'name':'NVIDIA H100','major':9},True),
       ('nvml_blackwell',{'name':'NVIDIA RTX 5090','major':12},True),
       ('already_initialized',{'initialized':True,'major':9},False),
       ('no_visible_gpu',{'visible':0},False),('hip_backend',{'cuda':False},False),
       ('non_nvml_native_fallback',{'nvml':False},False),
       ('multiple_physical_numeric_fallback',{'physical':{0:('NVIDIA H100',9),1:('NVIDIA L40S',8)},'visibility':'1'},False),
       ('different_cuda_nvml_order',{'physical':{0:('NVIDIA L40S',8),1:('NVIDIA H100',9)},'visibility':'0','cuda_order':[1,0]},False),
       ('full_uuid',{'physical':{0:('NVIDIA H100',9),1:('NVIDIA L40S',8)},'visibility':'GPU-1-full-uuid'},True),
       ('abbreviated_uuid',{'physical':{0:('NVIDIA H100',9),1:('NVIDIA L40S',8)},'visibility':'GPU-1'},True),
       ('ambiguous_uuid',{'physical':{0:('NVIDIA L40S',8),1:('NVIDIA H100',9)},'visibility':'GPU-'},False),
       ('initialized_current_device',{'initialized':True,'physical':{0:('NVIDIA L40S',8),1:('NVIDIA H100',9)},'current_device':1},False),
       ('external_current_device',{'physical':{0:('NVIDIA L40S',8),1:('NVIDIA H100',9)},'current_device':1,'driver_status':0},False),
       ('external_single_driver_initialized',{'driver_status':0},False),
       ('driver_error',{'driver_status':999},False),('driver_unavailable',{'driver_missing':True},False),
       ('nvml_query_failure',{'query_failure':True},False)]
rows=[]
for name,options,fast in cases:
    original_probe=Probe(**options);new_probe=Probe(**options)
    old=evaluate(source,original_probe,False);new=evaluate(patched,new_probe,True)
    assert old==new,(name,old,new)
    if fast:
        assert not new_probe.initialized and not new_probe.target_calls and not new_probe.torch_calls
        assert original_probe.initialized
    elif new[0]=='nvidia':assert new_probe.initialized
    if name=='external_current_device':assert new==('nvidia',True)
    rows.append({'case':name,'original_result':old,'patched_result':new,
                 'original_cuda_initialized':original_probe.initialized,'patched_cuda_initialized':new_probe.initialized,
                 'patched_torch_calls':new_probe.torch_calls,'patched_target_calls':new_probe.target_calls,
                 'patched_driver_queries':new_probe.driver_calls,'nvml_fast_path_expected':fast,'passed':True})
result={'patch_spec_sha256':hashlib.file_digest(spec_path.open('rb'),'sha256').hexdigest(),
        'before_sha256':spec['before_sha256'],'after_sha256':spec['after_sha256'],
        'native_torch_parser_sha256':hashlib.sha256(torch_source.encode()).hexdigest(),
        'native_cuda_guard_changed':False,'gpu_execution_tested':False,'cases':rows}
Path('docs/serving-comparison-data/fla-nvml-semantic-checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'cases_passed':len(rows),'descriptor_sha256':result['patch_spec_sha256'],'after_sha256':spec['after_sha256']}))
