"""Offline CPU checks for a hydrated CUDA serving environment, without a GPU."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

ROOT = Path("/opt/switchyard/runtime")
assert sys.version_info[:2] == (3, 12), sys.version
assert Path(sys.executable).resolve().is_relative_to(ROOT)
assert Path(sys.base_prefix).resolve().is_relative_to(ROOT)

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
    cuda_source.write_text('extern "C" __global__ void smoke(float* x){x[threadIdx.x] += 1.0f;}\n')
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
                  "gpu_execution_tested": False}, sort_keys=True))
