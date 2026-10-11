# Serving runtime hydration

Switchyard starts an inference engine from separately published filesystem artifacts. A small Ubuntu image starts the Rust agent, which hydrates the serving runtime, model weights, and optional compiler caches. The checked fleet streams trusted artifacts directly into staging files, overlaps runtime and model hydration, and places the model on a RAM-backed filesystem.

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

The builder forces `compileall -f --invalidation-mode unchecked-hash` for both the standalone interpreter's `python/lib/python3.12` tree and the virtual environment's `venv/lib/python3.12/site-packages` tree. These caches remain usable when extraction changes source-file timestamps. Python accepts them without repeating source-content hashing. Preparation records source and bytecode hashes in the artifact manifest; the trusted fleet startup path uses the pinned publication without scanning those contents again. This mode assumes the installed runtime remains immutable. Changes to Python sources require rebuilding the artifact. `PYTHONDONTWRITEBYTECODE=1` prevents serving processes from creating or rewriting runtime bytecode.

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

The smoke check also verifies unchecked-hash headers in emitted bytecode across both Python trees. It copies actual stdlib and vLLM source/cache pairs into temporary storage, changes the source timestamps, and loads the existing caches with a loader that rejects source reads and recompilation. The read-only runtime is untouched during this check.

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

## Worker startup policy

`bootstrap.startup` configures the endpoint agent independently of the provider. `mode: "trusted_streaming"` streams HTTP response bodies through zstd and TAR extraction into a private installation directory. `limits.hydration_concurrency: 20` bounds the combined runtime and model transfers to twenty active HTTP streams. Each independent runtime shard begins extraction as its bytes arrive; the worker does not first download the complete archive or retain compressed copies on disk.

Trusted startup omits content hashing, full-tree scans, and per-file synchronization. It retains path containment, supported entry types, declared size limits, and normal TLS/decompression framing. The runtime and model must come from trusted, pinned publications. Failures discard the private staging tree, and successful completion publishes it with an atomic rename. This mode reports wire and extraction progress without labeling those bytes as verified.

`mode: "verified"` remains the generic library default. It validates chunk and file hashes, verifies complete installations, and supports reuse of verified compressed chunks. Standalone callers can choose the trusted path with `switchyard artifacts hydrate --trusted-streaming --concurrency 20 --extraction-concurrency 20`.

`model_root: "/dev/shm/switchyard"` places the model directory at `/dev/shm/switchyard/model`. `require_model_ram: true` requires RAM backing and sufficient filesystem, host-memory, and container-memory capacity before transfer. The agent reports the observed filesystem and mount source. An overlay mount does not establish whether the underlying disk is NVMe. RunPod's system-RAM placement floor is configured separately; it does not establish the size of `/dev/shm`.

Runtime and model hydration run concurrently. With `early_launch: true`, the agent starts the configured engine prelude as soon as the runtime and compiler-cache dependencies are ready. The prelude may import engine modules while the model continues hydrating. It must wait for the exact stdin line `switchyard-model-ready-v1\n` before accessing model files. The agent releases that gate only after the current model installation succeeds. The checked vLLM prelude imports Torch and the serving CLI before waiting, then invokes the CLI with the managed model arguments. A failed or cancelled hydration terminates the owned engine process group.

The engine receives a controlled environment without inherited cloud or storage credentials. Serving readiness requires health, tokenization, and a successful decoded-token request. State observations include transfer progress, storage evidence, and startup milestones; they exclude signed download URLs.

## GPU compiler caches

Compiler caches are independent artifacts selected by serving integration. Required compatibility includes the runtime artifact identity, model revision, GPU, full engine-configuration fingerprint, and runtime launch-configuration fingerprint. Arguments or environment flags that affect compilation therefore invalidate a previously baked cache. The engine writes to a separate copy of the selected cache seed, preserving the immutable source artifact.

A CPU runtime build precompiles Python bytecode and installs available upstream precompiled kernels. A separately owned GPU preparation job runs the actual model with the intended serving configuration to compile GPU-specific Triton and TorchInductor kernels, populate other engine caches, and measure model KV capacity. Ordinary filesystem cache export does not preserve process-local GPU allocations or CUDA graph captures; the serving process recreates that state. The eager baseline measures serving without claiming optimized compilation-cache reuse; an optimized profile must be baked and measured independently. Changes to the engine, model, GPU, or compilation settings require a new compatible artifact.

`VLLM_ENABLE_STARTUP_PLAN=1` persists the upstream engine's measured KV-memory plan under `VLLM_CACHE_ROOT`. A compatible subsequent boot can reuse that plan when the hardware/configuration fingerprint and free-memory conditions match. vLLM still performs its model profile run, so plan reuse does not eliminate every initialization operation. The preparation artifact also captures Triton, FlashInfer, CUDA driver, XDG, and Torch-extension caches beneath the selected writable cache subtree. CUDA driver cache capacity is set to 4 GiB. The trusted experiment also sets `FLASHINFER_CUBIN_CHECKSUM_DISABLED=1` to avoid hashing prepublished kernel files during serving initialization, and `FLASHINFER_NO_DOWNLOAD=1` to reject a missing kernel rather than fetch it during the experiment. These flags belong to the serving configuration and participate in its cache fingerprint.

Startup reports separate container availability, overlapping network/extraction, engine imports, model readiness, engine initialization, and first-token latency. Each optimization must demonstrate its benefit on a fresh allocation with a compatible prepared cache; a CPU-only test cannot establish GPU startup performance.

## Cache bake acceptance

A cache bake runs the selected engine, model, GPU, and serving configuration through startup and representative inference. The requests cover the batch sizes, input lengths, output lengths, and sampling operations supported by the deployment. Kernels for unexercised shapes or operations may still require compilation. The job exports the engine's persistable compiler outputs and tuning data with the runtime and model compatibility labels.

A second, fresh allocation hydrates the exported artifact and runs the same configuration. Acceptance requires observations of cache consumption, compilation misses or rebuilds, and the time to the first useful response token. A populated cache directory alone does not establish reuse. For vLLM, `VLLM_FORCE_AOT_LOAD=1` makes a missing Torch compilation artifact fail explicitly; it does not prove that separate FlashInfer or Triton code paths avoided compilation. Those paths require their own observations. See the [vLLM faster-startup guidance](https://docs.vllm.ai/en/stable/configuration/optimization/#faster-startup).

Engine memory profiling measures temporary GPU memory demand to size the KV cache that holds attention state. A compatible startup plan can reuse its recorded sizing result, subject to the engine's hardware and free-memory checks. Warm-up may compile kernels, perform tuning, or establish process-local execution state. Persistable results belong in the bake artifact; repeated process-local setup must have a separately measured duration. Diagnostic profiling times these operations during an investigation and is optional instrumentation, not part of the normal serving contract.

Fresh-host measurements report provisioning, agent startup, runtime reception and extraction, model reception and loading, engine initialization, and the first useful token. Concurrent stages overlap, so their durations must not be summed as though they were sequential. A cold-start improvement must reduce the end-to-end measurement without invalidating inference correctness or the required steady-state throughput.
