# Serving runtime and compiler-cache preparation

A serving runtime contains the pinned Python environment, CUDA compiler and headers, native libraries, and package bytecode. A compiler cache contains the model-specific Torch AOT and Triton outputs plus FlashInfer sampling binaries. A fresh worker hydrates both artifacts. Its configuration binds the runtime, model content, engine settings, launch program, and GPU to the cache metadata.

## Runtime source and patches

The complete corrected vLLM runtime is `cb2c15aea8acc7494f5240a3b162d21be83fa828650d2e5ef02592b9fd29ce14`. It contains vLLM 0.31.0+cu129, Python 3.12.14, Torch 2.13.0+cu129, Triton 3.7.1, and FlashInfer 0.7.0.post1. Its CUDA 12.9 compiler closure includes CURAND and supports the native sampling compilation used during a cache bake.

The FLA descriptor is `deploy/hydration/patches/vllm-0.31.0-fla-nvml.json`, SHA-256 `5b935fc5f4eacaff29876d2c9304df0b54ebe69828c6c3da8916ba36e54135f5`. It transforms the pinned module from `d6c90d36505e02086954c23ca4dd2fea82a69416d68ff1d0501e16b3148b5b0c` to `3f19edd2c7aba0cb6a3251715199c3fe82ad1b20b7946b24b192d23bec82b828`. The fast path requires one visible GPU, an uninitialized CUDA driver, and either UUID visibility or one physical GPU. Query failures, initialized drivers, and ambiguous numeric ordering retain the original Torch queries. The upstream CUDA and NUMA multiprocessing guards remain unchanged. The driver check uses `cuCtxGetCurrent` without calling `cuInit`; the [CUDA Driver API initialization contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__INITIALIZE.html) specifies the uninitialized status returned before initialization.

The canonical runtime builder can apply a descriptor after verifying its path, descriptor digest, installed package version, pristine source digest, replacement counts, and result digest. It must rebuild bytecode after patching and record the patch receipt in `build-info.json`. Neither runtime source lock enables this patch automatically. The independent lazy-Inductor experiment is excluded from this FLA experiment.

The private SM89 derivative is `808e0a7b84d878c32de76f5da6a4b05894e1d61a013c9ccb000b1ea7f59bf9bd`. Relative to the unpatched scoped runtime, only the declared source module, its Python 3.12 bytecode, and provenance metadata change. Its 0.8B configuration removes universal cubin payloads while preserving compiler files, native libraries, headers, package metadata, and directory paths. The patched 0.8B GPU test completed 222 requests and nine literal checks, retained native fork, and loaded Torch AOT plus the native sampling provider. Its accepted cache is `ccc33cfaec8d0e669950f4b61b229fb5062ba46e8e17be1b76a613c10e349c69`. That asset profile has only been accepted for this 0.8B engine configuration; broader model selection requires GPU acceptance. The measured process-start interval improved, while total first-token time did not improve in the matched pair; the GPU receipt records both outcomes.

## Model identities

The three Qwen revisions used for the comparison and parent raw-extents configurations are:

| Model | Repository | Revision | Raw header-first extent manifest |
| --- | --- | --- | --- |
| 0.8B | Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | `0222a31a9a1b9c24724718965056f13d3196aff4c9175fe58c281ef4677f839a` |
| 2B | Qwen/Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` | `5f69d906cb0eed9c402cb7c8da8999bc6377a94f9b071d3b2e549f54549f0158` |
| 4B | Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `a51d79850d8a018e39961356ca80de6d6b31a3f729d57f32edaa8a42779780d6` |

The raw content IDs are `ba047becd40d1a7b6c6dfb5939e91f93b2bfc3cf139d570ea88b58a2a32c5e1e` for 0.8B, `91aac048ec06b7d94b2fb6893a3c197d1f6b6842bcdf5a2139ffd955b2fb0b38` for 2B, and `d7545c853741122e1554a189fc6867d0b6277d79a058ca8bb5864126a747c1a6` for 4B. The parent `configs/runpod/qwen-extents-l40s.json` pins these manifests and content IDs. Its loader uses `venv/bin/python`, CUDA staging, and a 16 MiB pinned buffer. Preserve the loader configuration and model content identity when preparing the final fleet. The original gzip model archives have different archive identities; do not substitute those labels for raw-extent manifests.

## Per-model native cache bake

For each model, use the final runtime and launch configuration with an empty model cache list. Enable native compilation for the bake by omitting `VLLM_FORCE_AOT_LOAD` and `FLASHINFER_DISABLE_JIT`. Use that model's bake plan, GPU architecture 8.9, exact engine settings, and fixed RC2 agent. The parent plans are `configs/runpod/cache-bake-l40s-0.8b.json`, `cache-bake-l40s-2b.json`, and `cache-bake-l40s-4b.json`.

Run `switchyard live bake --config <fleet> --plan <model-plan> --env-file <provider-env> --env-file <storage-env> --out <fresh-output> --namespace switchyard/artifacts/cache`. This warms native inference, exports the cache, releases the owned GPU, and publishes the compiler artifact. Require a successful receipt and a confirming provider GET 404 before using its exported `cache-configuration.json`. Bind that exact model configuration to the export; the 0.8B Torch cache cannot serve as the 2B or 4B cache.

The original six-case engine matrix did not export reusable 2B or 4B caches. The parent's later 2B raw-extent bake exported `e005a45a52d975987e1ccb214632e4cae6c74af41c6fdfa98c2632158c77c8c3` from the complete unpatched runtime. The later 4B raw-extent bake exported `4dcca475c3e4adea7cf329ec546b783890c4fc1855528d31c41b4dc2ea81ea8f`. The parent owns their strict derivatives and fresh reuse acceptance. The 0.8B FLA acceptance remains based on the original complete archive, SHA-256 `bc98fc9c5e6c9f00289e7fb63882f7a34e1c003e05da071d38a0adfc16a09178`; its raw-extent cache still requires preparation or acceptance under the final fleet configuration. A library patch or loader change still requires native AOT acceptance under the final configuration.

## Native sampling provider and fresh reuse

Use `scripts/serving-comparison/strict-cache.py` on a single-model configuration whose cache reference is the bake export. Pass `--bake <bake-output> --config <matching-config> --out <fresh-derivative> --sampling-provider-shim <shim-wheel>`. The pinned official shim is FlashInfer 0.7.0.post1+cu129, SHA-256 `314d97ea8c3d288d0afce09e9f3dc823a83d4a219c02b85097de71ee143cf0fb`. The script registers the baked `sampling.so` through `flashinfer.jit_cache.providers`, preserves every original compiler entry, and sets `VLLM_FORCE_AOT_LOAD=1` and `FLASHINFER_DISABLE_JIT=1`.

Keep compiled paths at the original bake prefix, or create a checked alias before serving imports. Python can cache a nonexistent `PYTHONPATH` directory before the launcher creates that alias. Call `importlib.invalidate_caches()` after the alias exists and before imports that discover native providers. A fork inherits both the importer cache and FlashInfer's provider inventory. The native reproduction in `check-provider-alias.py` shows zero providers in the parent and fork without invalidation, one in a fresh interpreter, and one in all three after invalidation.

Publish the verified derivative with the artifact CLI, then run a fresh native gauntlet under the final runtime, model loader, launch program, and GPU. Require explicit Torch AOT load, native FlashInfer provider load with JIT disabled, completed correctness and concurrency cases, and confirmed cleanup. Cache fingerprints establish the declared identity; native acceptance establishes that the exported binaries actually execute.
