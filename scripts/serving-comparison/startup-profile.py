"""Diagnostic sitecustomize: time native startup calls without changing results.

The launcher writes this file outside the runtime and adds its directory to
PYTHONPATH for the diagnostic process and its children. It logs no arguments,
prompts, environment values, or credentials. Durations may overlap through
nesting; they must not be added together.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
import traceback

FUNCTIONS = {
    'torch.cuda': ['_lazy_init'],
    'vllm.v1.worker.gpu.warmup': ['warmup_kernels', '_warmup_kernels', 'run_mixed_prefill_decode_warmup'],
    'vllm.model_executor.warmup.kernel_warmup': [
        'kernel_warmup', 'cutedsl_warmup', 'qwen_triton_warmup',
        'qwen_vl_triton_warmup', 'mamba_triton_warmup', 'is_deep_gemm_supported',
        'deep_gemm_warmup', 'b12x_warmup', 'minimax_m3_msa_warmup',
        'flashinfer_sparse_mla_decode_autotune_warmup',
        'deepseek_v4_sparse_mla_attention_warmup', '_warmup_gemm_rs_ar',
        'kimi_k3_triton_warmup', 'watermark_sample_warmup', 'qwen4_exp_qsa_triton_warmup',
    ],
}
CLASSES = {
    'vllm.v1.worker.gpu_worker': {'Worker': [
        'init_device', 'load_model', 'determine_available_memory',
        'initialize_from_config', 'compile_or_warm_up_model',
        'execute_model', 'sample_tokens',
    ]},
    'vllm.v1.worker.gpu.model_runner': {'GPUModelRunner': [
        'load_model', 'profile_run', 'capture_model',
    ]},
    'vllm.v1.worker.gpu_model_runner': {'GPUModelRunner': [
        'load_model', 'profile_run', 'capture_model',
    ]},
    'flashinfer.jit.core': {
        'JitSpec': ['build_and_load'],
        'JitSpecNvcc': ['try_load', 'build', 'load'],
    },
}
ACTIVE = True
ASSET_PATHS = set()

def audit_asset_access(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    path = os.fsdecode(args[0])
    marker = '/site-packages/flashinfer_cubin/'
    if marker in path and path not in ASSET_PATHS:
        ASSET_PATHS.add(path)
        print('SWITCHYARD_RUNTIME_ASSET_ACCESS ' + json.dumps({
            'pid': os.getpid(), 'path': 'flashinfer_cubin/' + path.split(marker, 1)[1],
        }, sort_keys=True), flush=True)

sys.addaudithook(audit_asset_access)

def emit(**values):
    print('SWITCHYARD_STARTUP_PROFILE ' + json.dumps({
        'pid': os.getpid(), 'monotonic_s': time.monotonic(), **values,
    }, sort_keys=True), flush=True)

def timed(function, name):
    @functools.wraps(function)
    def call(*args, **kwargs):
        global ACTIVE
        if not ACTIVE:
            return function(*args, **kwargs)
        if name == 'torch.cuda._lazy_init' and sys.modules['torch.cuda'].is_initialized():
            return function(*args, **kwargs)
        began = time.monotonic()
        observation = {}
        if name == 'torch.cuda._lazy_init':
            observation['callers'] = [
                {'file': os.path.basename(frame.filename), 'function': frame.name, 'line': frame.lineno}
                for frame in traceback.extract_stack(limit=14)[:-1]
            ]
        if name.endswith('determine_available_memory') and args:
            snapshot = getattr(args[0], 'init_snapshot', None)
            if snapshot is not None:
                observation = {'free_memory_bytes': snapshot.free_memory,
                               'total_memory_bytes': snapshot.total_memory}
        emit(event='begin', name=name, **observation)
        try:
            return function(*args, **kwargs)
        finally:
            emit(event='end', name=name, duration_s=time.monotonic() - began)
            if name.endswith('compile_or_warm_up_model'):
                ACTIVE = False
    return call

class Loader:
    def __init__(self, inner): self.inner = inner
    def create_module(self, spec): return self.inner.create_module(spec)
    def exec_module(self, module):
        self.inner.exec_module(module)
        for name in FUNCTIONS.get(module.__name__, []):
            function = getattr(module, name, None)
            if callable(function):
                setattr(module, name, timed(function, module.__name__ + '.' + name))
        for classname, methods in CLASSES.get(module.__name__, {}).items():
            cls = getattr(module, classname, None)
            if cls is not None:
                for name in methods:
                    function = getattr(cls, name, None)
                    if callable(function):
                        setattr(cls, name, timed(function, module.__name__ + '.' + classname + '.' + name))

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in FUNCTIONS and fullname not in CLASSES: return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader: spec.loader = Loader(spec.loader)
        return spec

sys.meta_path.insert(0, Finder())
emit(event='process_started')
