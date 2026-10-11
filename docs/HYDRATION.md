# Serving runtime hydration

Switchyard can start an inference engine from a verified filesystem artifact instead of installing the serving stack in the GPU container image. A small Ubuntu image starts the Rust agent. The agent downloads and verifies the runtime, compatible optional compiler caches, and model weights before launching the engine.

The runtime occupies `/opt/switchyard/runtime`. That fixed prefix preserves Python virtual-environment discovery, executable shebangs, and native-library paths. The artifact contains its own Python interpreter, virtual environment, CUDA wheel libraries, selected native dependencies, and relocated C/C++ and CUDA compilers. It does not replace the worker's `/usr`, `/lib`, or package database. The host still supplies the NVIDIA driver and GPU devices.

## Component boundaries

| Component | Responsibility |
| --- | --- |
| `switchyard-runtime-build` | Assemble the locked serving environment inside an isolated Docker builder; record build provenance. |
| `switchyard-artifacts` | Pack filesystem trees, validate manifests, transfer chunks, safely extract entries, and verify complete installations. |
| `switchyard-artifact-tools` | Expose standalone build, pack, publish, hydrate, and verify commands. |
| `switchyard-provider` | Implement object-storage signing and GPU-provider lifecycle operations. |
| `switchyard-live` | Resolve configured runtime/cache references and enforce engine, model, and GPU compatibility. |
| `switchyard-agent` | Hydrate the selected artifacts, launch the engine, verify inference readiness, and expose progress. |
| Core, scheduler, and simulator | Use generic capacity, startup, transfer, readiness, and failure observations. They have no serving-package or storage-provider dependencies. |

Docker is the build environment and the source of a reproducible filesystem export. The worker image contains Ubuntu, certificates, and curl for downloading the verified static agent. Runtime packages and models arrive through hydration rather than image layers. Docker build caches are local preparation aids; they are not required on a provisioned GPU.

## Locked builder inputs

`deploy/hydration/sources.lock.json` pins the Ubuntu and Rust builder images by digest, the Ubuntu package snapshot, and the Python, uv, vLLM, and CUDA compiler downloads by SHA-256. `requirements.in` specifies the intended serving packages. `requirements.lock` supplies exact transitive versions and download hashes. Installation requires hashes and binary distributions; source-package builds are disabled.

The checked-in recipe targets Linux x86_64 and Ubuntu 24.04 with Python 3.12, vLLM 0.31.0, and CUDA 12.9 Torch wheels. Explicit vLLM, Torch, torchvision, torchaudio, and TorchCodec URLs prevent an unrelated default CUDA wheel variant from entering the runtime. TorchCodec uses the matching build from the [official CUDA 12.9 wheel index](https://download.pytorch.org/whl/cu129/torchcodec/). FlashInfer's precompiled cubin distribution is resolved through its official additional wheel index. Native compiler and FFmpeg 6 packages come from the pinned Ubuntu snapshot, and required shared libraries are discovered from the assembled ELF files.

The `nvidia-cuda-nvcc-cu12` Python wheel supplies compiler support files but omits the `nvcc` driver required by CUDA extension builders. The artifact therefore includes the `cuda_nvcc`, `cuda_cudart`, and `cuda_cccl` components from [NVIDIA's CUDA 12.9.1 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json). These provide the compiler, runtime headers and libraries, and CUB/Thrust headers under `runtime/cuda`, with their license files. `CUDA_HOME`, `CUDA_PATH`, and `CUDACXX` point into that tree. The driver stub supplied for linking is excluded from the runtime library search path.

The Rust builder separates downloading, filesystem normalization, native-library/toolchain assembly, and command execution into modules. It verifies downloaded source hashes, checks installed Python dependency consistency, compiles Python bytecode, and converts internal absolute symlinks to relative links. A link that resolves outside the runtime is rejected. Native binaries copied from the build system are dereferenced and retain executable permissions.

The export includes `build-info.json` with installed package versions, Ubuntu package versions, native dependency origins, and source/builder hashes. `artifact-metadata.json` supplies compatibility labels and provenance for the generic packer. A CPU build records `gpu_bake_performed=false`; it does not claim to contain GPU-specific compiled caches.

## Build and verify a runtime

Run these commands from the repository root. Docker must support BuildKit and buildx; output directories must be scratch locations outside the build context's included source files.

```sh
cargo build --release --features artifacts
target/release/switchyard artifacts build-runtime \
  --out results/runtime-export

docker build --file deploy/hydration/Dockerfile.smoke \
  --tag switchyard-hydration-smoke:local deploy/hydration
docker run --rm --network none \
  --label io.switchyard.component=hydration-smoke \
  --mount type=bind,source="$PWD/results/runtime-export/runtime",target=/opt/switchyard/runtime,readonly \
  switchyard-hydration-smoke:local

target/release/switchyard artifacts pack \
  --source results/runtime-export/runtime \
  --metadata results/runtime-export/runtime/artifact-metadata.json \
  --out results/runtime-artifact
```

The smoke container uses the pinned, otherwise untouched Ubuntu baseline. With its network disabled and the runtime mounted read-only, it imports the installed serving packages and vLLM serving modules, performs a Torch CPU operation, decodes a generated WAV through TorchCodec, and compiles and runs a small C++ program using the artifact's compiler. It also compiles a CUDA kernel into an `sm_89` cubin without executing it. This proves runtime dependency completeness for those operations. CUDA vLLM's argument parser constructs a device configuration even for `--version`, so CLI parsing, GPU execution, model loading, and endpoint performance require the separately measured GPU experiment.

The packer records each file's size, permissions, and SHA-256 alongside directory and internal-symlink entries. It normalizes archive ownership and timestamps, partitions files into independently compressed zstd tar streams, and computes an identity covering the metadata, entries, and compressed chunks. An individual large file remains a streamed entry rather than being loaded into memory.

## Publish and configure an artifact

Publishing uses the storage block in a fleet configuration and locally supplied credential variables. The command uploads immutable chunks first and publishes the manifest only after all chunks have been verified. An existing object must match the expected content. The receipt supplies the exact prefix and artifact identity needed by the serving configuration.

```sh
target/release/switchyard artifacts publish \
  --config configs/runpod/qwen-small.json \
  --env-file .env \
  --directory results/runtime-artifact \
  --prefix switchyard/artifacts/runtime \
  --receipt results/runtime-publication.json
```

Set `bootstrap.runtime.artifact` to the published reference, `program` to `venv/bin/python`, and `args` to `["-m", "vllm.entrypoints.cli.main"]`. Runtime environment entries can use `{runtime_root}` and `{cache_root}` placeholders. Use the native-library path, compiler paths, certificate path, and offline/cache variables from the build's `runtime-env.sh`. The live runner checks the executable and compatibility labels before provisioning. A credential file, publication scratch, and temporary signed URLs belong outside version control.

`FLASHINFER_WORKSPACE_BASE` places FlashInfer's `.cache/flashinfer` directory and version/architecture-specific compilation workspace beneath the configured cache root. Triton, TorchInductor, vLLM, CUDA, and Hugging Face paths are configured separately in `runtime-env.sh`. A GPU cache-baking job must use the same environment as the later serving process so the exported cache tree includes every engine cache.

## Worker startup and verification

The agent's observation endpoint is available during preparation. It verifies each runtime or cache manifest before accepting downloads, then transfers chunks with bounded concurrency and retries. Chunk hashes and lengths are checked before extraction. Extraction accepts explicit regular files, directories, and validated internal symlinks; archive traversal, escaping links, unexpected entries, unsafe modes, and decompression-size violations fail hydration.

Hydration uses a staging directory. Files are verified against the manifest before the installation becomes available through an atomic directory rename. Reusing an installed artifact requires verification of the complete tree, including its contents and links. A marker or directory name alone is insufficient evidence of a cache hit. Verified compressed chunks may also be reused independently after an interrupted installation.

The agent prepares the runtime and configured compiler-cache artifacts, then hydrates the model and starts the engine. The engine receives a controlled environment without inherited cloud or storage credentials. The agent reports ready only after the engine's health check and an actual inference readiness check succeed. Artifact observations expose identity, phase, transferred and verified bytes, extraction progress, and cache hits without exposing download URLs.

## GPU compiler caches

Compiler caches are independent artifacts selected by serving integration. Required compatibility includes the runtime artifact identity, model revision, GPU, full engine-configuration fingerprint, and runtime launch-configuration fingerprint. Arguments or environment flags that affect compilation therefore invalidate a previously baked cache. The engine writes to a separate copy of the verified cache seed, preserving the immutable source artifact.

A CPU runtime build precompiles Python bytecode and installs available upstream precompiled kernels. It cannot bake GPU-specific Triton or TorchInductor kernels, establish model KV capacity, or persist process-local CUDA graph captures. These require a separately owned GPU preparation job using the exact serving configuration. The eager baseline measures serving without claiming optimized compilation-cache reuse; an optimized profile must be baked and measured independently. Changes to the engine, model, GPU, or compilation settings require a new compatible artifact.

Startup reports should separate container availability, network hydration, extraction and verification, engine initialization, inference readiness, and first-token latency. Runtime and model payloads remain substantial even when the base image is small. Parallel transfer, verified local reuse, and compatible compiler caches must demonstrate their benefit through measured results.
