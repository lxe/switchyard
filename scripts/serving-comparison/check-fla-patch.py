"""Exercise the pinned native FLA patch's platform semantics and CUDA side effects."""
import ast
import copy
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

spec_path = Path('deploy/hydration/patches/vllm-0.31.0-fla-nvml.json')
spec = json.loads(spec_path.read_text())
source_path = Path('results/vllm-export/runtime') / spec['target']
source = source_path.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == spec['before_sha256']
patched = source
for replacement in spec['replacements']:
    assert patched.count(replacement['before']) == 1
    patched = patched.replace(replacement['before'], replacement['after'])
assert hashlib.sha256(patched.encode()).hexdigest() == spec['after_sha256']

class Probe:
    def __init__(self, *, cuda=True, visible=1, initialized=False, name='NVIDIA L40S', major=8,
                 nvml=True, physical=None, visible_ids=None, assigned_ids=None, current_device=0):
        self.cuda=cuda; self.visible=visible; self.initialized=initialized
        self.name=name; self.major=major; self.nvml=nvml; self.current_device=current_device
        self.physical=physical or {0:(name,major)}
        self.visible_ids=visible_ids or ([0] if visible else [])
        self.assigned_ids=assigned_ids or self.visible_ids
        self.target_calls=0; self.torch_calls=0; self.nvml_calls=0
    def is_initialized(self): return self.initialized
    def _device_count_nvml(self): return len(self.visible_ids) if self.nvml else -1
    def get_device_name(self,index=0):
        self.torch_calls+=1; self.initialized=True
        return self.physical[self.visible_ids[index]][0]
    def get_device_capability(self):
        self.torch_calls+=1; self.initialized=True
        return self.physical[self.visible_ids[self.current_device]][1],0
    def get_current_target(self):
        self.target_calls+=1
        if not self.visible_ids: raise RuntimeError('no active target')
        if self.cuda:self.initialized=True
        return SimpleNamespace(backend='cuda' if self.cuda else 'hip')

class NvmlPlatform:
    def __init__(self,probe):self.probe=probe
    def is_cuda(self):return self.probe.cuda
    def visible_device_id_to_physical_device_id(self,index):return self.probe.visible_ids[index]
    def _get_physical_device_name(self,physical_id):
        self.probe.nvml_calls+=1
        return self.probe.physical[physical_id][0]
    def get_device_capability(self,index=0):
        self.probe.nvml_calls+=1
        return SimpleNamespace(major=self.probe.physical[self.probe.visible_ids[index]][1])
    def get_device_name(self,logical_id=0):
        # Deliberately different from a visible ordinal in the mapping fixture.
        return self.probe.physical[self.probe.assigned_ids[logical_id]][0]

class NonNvmlPlatform:
    def __init__(self,probe):self.probe=probe
    def is_cuda(self):return self.probe.cuda

def evaluate(code, probe, patched):
    import sys
    sys.modules['vllm.platforms.cuda']=SimpleNamespace(NvmlCudaPlatform=NvmlPlatform)
    namespace = {'functools':functools,
                 'torch':SimpleNamespace(cuda=probe),
                 'current_platform':(NvmlPlatform(probe) if probe.nvml else NonNvmlPlatform(probe)),
                 'triton':SimpleNamespace(runtime=SimpleNamespace(driver=SimpleNamespace(active=probe))),
                 'Literal':__import__('typing').Literal}
    tree = ast.parse(code)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {'get_available_device','_check_platform','_is_nvidia_hopper'}]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'native-fla-functions', 'exec'), namespace)
    platform = namespace['_check_platform']()
    namespace['is_nvidia'] = platform == 'nvidia'
    if patched:
        hopper = namespace['_is_nvidia_hopper']()
    else:
        assignment = next(n for n in tree.body if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=='is_nvidia_hopper' for t in n.targets))
        hopper = eval(compile(ast.Expression(assignment.value),'native-fla-expression','eval'),namespace)
    return platform, hopper

cases = [('nvml_sm89',{}),('nvml_h100',{'name':'NVIDIA H100','major':9}),
         ('nvml_blackwell',{'name':'NVIDIA RTX 5090','major':12}),
         ('already_initialized',{'initialized':True,'major':9}),
         ('no_visible_gpu',{'visible':0}),('hip_backend',{'cuda':False}),
         ('non_nvml_native_fallback',{'nvml':False}),
         ('visible_vs_assigned_logical',{'physical':{3:('NVIDIA L40S',8),7:('NVIDIA H100',9)},'visible_ids':[3,7],'assigned_ids':[7]}),
         ('uuid_visible_mapping',{'physical':{'GPU-mapped':('NVIDIA L40S',8)},'visible_ids':['GPU-mapped']}),
         ('initialized_current_device',{'initialized':True,'physical':{3:('NVIDIA L40S',8),7:('NVIDIA H100',9)},'visible_ids':[3,7],'assigned_ids':[3],'current_device':1})]
rows=[]
for name,options in cases:
    original_probe=Probe(**options);new_probe=Probe(**options)
    old=evaluate(source,original_probe,False);new=evaluate(patched,new_probe,True)
    assert old == new,(name,old,new)
    if name in {'nvml_sm89','nvml_h100','nvml_blackwell','visible_vs_assigned_logical','uuid_visible_mapping'}:
        assert not new_probe.initialized and not new_probe.target_calls and not new_probe.torch_calls
        assert original_probe.initialized
    if name == 'non_nvml_native_fallback': assert new_probe.initialized
    if name == 'already_initialized': assert new_probe.target_calls == original_probe.target_calls
    rows.append({'case':name,'original_result':old,'patched_result':new,
                 'original_cuda_initialized':original_probe.initialized,'patched_cuda_initialized':new_probe.initialized,
                 'patched_torch_calls':new_probe.torch_calls,'patched_target_calls':new_probe.target_calls,'passed':True})
result={'patch_spec_sha256':hashlib.file_digest(spec_path.open('rb'),'sha256').hexdigest(),
        'before_sha256':spec['before_sha256'],'after_sha256':spec['after_sha256'],
        'native_cuda_guard_changed':False,'gpu_execution_tested':False,'cases':rows}
Path('docs/serving-comparison-data/fla-nvml-semantic-checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
