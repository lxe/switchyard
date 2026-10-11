"""Offline CPU checks for a hydrated CUDA serving environment, without a GPU."""

import importlib
import importlib.metadata
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

ROOT = Path("/opt/switchyard/runtime")
assert sys.version_info[:2] == (3, 12), sys.version
assert Path(sys.executable).resolve().is_relative_to(ROOT)
assert Path(sys.base_prefix).resolve().is_relative_to(ROOT)
assert sys.dont_write_bytecode

bytecode_counts = {}
for name, library in (("stdlib", ROOT / "python/lib/python3.12"),
                      ("site_packages", ROOT / "venv/lib/python3.12/site-packages")):
    count = 0
    for cache in library.rglob(f"*.{sys.implementation.cache_tag}.pyc"):
        with cache.open("rb") as stream:
            header = stream.read(16)
        assert header[:4] == importlib.util.MAGIC_NUMBER, cache
        assert int.from_bytes(header[4:8], "little") == 1, cache
        count += 1
    assert count > 0, library
    bytecode_counts[name] = count


class BytecodeOnlyLoader(importlib.machinery.SourceFileLoader):
    def get_data(self, path):
        if path.endswith(".py"):
            raise AssertionError("accepted bytecode must not read source contents")
        return super().get_data(path)

    def source_to_code(self, *args, **kwargs):
        raise AssertionError("accepted bytecode must not recompile source")


# Copy actual stdlib and serving-module caches into scratch. Alter only source
# timestamps to mirror hydration, and reject source reads or compilation.
with tempfile.TemporaryDirectory(prefix="switchyard-bytecode-") as temporary:
    for index, original in enumerate((ROOT / "python/lib/python3.12/json/__init__.py",
                                      ROOT / "venv/lib/python3.12/site-packages/vllm/__init__.py")):
        source = Path(temporary) / str(index) / "module.py"
        source.parent.mkdir()
        shutil.copy2(original, source)
        cache = Path(importlib.util.cache_from_source(str(source)))
        cache.parent.mkdir()
        original_cache = Path(importlib.util.cache_from_source(str(original)))
        shutil.copy2(original_cache, cache)
        assert cache.read_bytes()[8:16] == importlib.util.source_hash(source.read_bytes())
        changed_timestamp = source.stat().st_mtime + 86400
        os.utime(source, (changed_timestamp, changed_timestamp))
        assert BytecodeOnlyLoader(f"bytecode_smoke_{index}", str(source)).get_code(
            f"bytecode_smoke_{index}") is not None

versions = {}
for module in ("torch", "torchvision", "torchaudio", "torchcodec", "vllm", "transformers", "tokenizers", "triton"):
    importlib.import_module(module)
    versions[module] = importlib.metadata.version(module)

import torch

assert torch.version.cuda == "12.9", torch.version.cuda
assert torch.arange(4).sum().item() == 6
with tempfile.TemporaryDirectory(prefix="switchyard-smoke-") as temporary:
    source = Path(temporary) / "smoke.cpp"
    binary = Path(temporary) / "smoke"
    source.write_text('#include <vector>\nint main(){std::vector<int> v{1,2};return v.size()!=2;}\n')
    subprocess.run([os.environ["CXX"], str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    cuda_source = Path(temporary) / "smoke.cu"
    cubin = Path(temporary) / "smoke.cubin"
    cuda_source.write_text('#include <curand.h>\n#include <curand_kernel.h>\nextern "C" __global__ void smoke(float* x){curandState_t s;curand_init(1234,threadIdx.x,0,&s);x[threadIdx.x] += curand_uniform(&s);}\n')
    subprocess.run([os.environ["CUDACXX"], "--cubin", "-arch=sm_89", "-ccbin", os.environ["CXX"],
                    str(cuda_source), "-o", str(cubin)], check=True)
    assert cubin.read_bytes().startswith(b"\x7fELF")
    audio = Path(temporary) / "smoke.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 160)
    from torchcodec.decoders import AudioDecoder
    samples = AudioDecoder(str(audio)).get_all_samples()
    assert samples.sample_rate == 16000
    assert samples.data.numel() == 160

# CUDA vLLM's argument parser instantiates a GPU DeviceConfig, even for --version.
# Import its real serving modules here; argument parsing/model execution is a GPU check.
for module in ("vllm.entrypoints.cli.serve", "vllm.entrypoints.cli.benchmark.main",
               "vllm.entrypoints.openai.api_server"):
    importlib.import_module(module)
print(json.dumps({"status": "passed", "python": sys.version.split()[0], "packages": versions,
                  "cuda_build": torch.version.cuda, "cuda_compilation_tested": True,
                  "gpu_execution_tested": False, "bytecode_invalidation": "unchecked-hash",
                  "bytecode_caches_checked": bytecode_counts,
                  "timestamp_independent_bytecode_tested": True}, sort_keys=True))
