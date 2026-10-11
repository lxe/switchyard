# Serving runtime and compiler-cache preparation

A serving runtime contains the pinned Python environment, CUDA compiler and headers, native libraries, and package bytecode. A compiler cache contains the model-specific Torch AOT and Triton outputs plus FlashInfer sampling binaries. Each fresh worker hydrates these artifacts, and its configuration binds the cache to the runtime, model content, weight loader, engine settings, launch program, and GPU architecture.

## Runtime identity

The complete vLLM runtime is `cb2c15aea8acc7494f5240a3b162d21be83fa828650d2e5ef02592b9fd29ce14`. It contains vLLM 0.31.0+cu129, Python 3.12.14, Torch 2.13.0+cu129, Triton 3.7.1, and FlashInfer 0.7.0.post1. Its CUDA 12.9 compiler closure includes CURAND and supports native sampling compilation during a cache bake. The strict raw-extent fleet in `configs/runpod/qwen-extents-l40s-strict.json` uses this complete runtime with no library patches.

The source locks are `deploy/hydration/sources.lock.json` and `deploy/hydration/sglang/sources.lock.json`. Neither enables a source patch. `switchyard artifacts build-runtime` builds the complete artifact through the locked Docker/Rust recipe; [Runtime hydration](HYDRATION.md) documents the offline compiler, header, audio, bytecode, and dependency checks. These CPU checks establish the runtime closure. Model execution and native compiler-cache reuse require separate GPU acceptance.

## Model identities and weight loading

The raw-extent fleet pins the following repositories, revisions, and header-first manifests. Each manifest records the extent layout; its content ID identifies the exact model content independently of that layout.

| Model | Repository | Revision | Extent manifest |
| --- | --- | --- | --- |
| 0.8B | Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | `0222a31a9a1b9c24724718965056f13d3196aff4c9175fe58c281ef4677f839a` |
| 2B | Qwen/Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` | `5f69d906cb0eed9c402cb7c8da8999bc6377a94f9b071d3b2e549f54549f0158` |
| 4B | Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `a51d79850d8a018e39961356ca80de6d6b31a3f729d57f32edaa8a42779780d6` |

The content IDs are `ba047becd40d1a7b6c6dfb5939e91f93b2bfc3cf139d570ea88b58a2a32c5e1e` for 0.8B, `91aac048ec06b7d94b2fb6893a3c197d1f6b6842bcdf5a2139ffd955b2fb0b38` for 2B, and `d7545c853741122e1554a189fc6867d0b6277d79a058ca8bb5864126a747c1a6` for 4B. The fleet's loader uses `venv/bin/python`, CUDA staging, and a 16 MiB pinned buffer. Its loader compatibility digest is `2cc44ced9e0f841bc0644b3a3ac6d8274c311eb9e15a2f121deeefebbe88fb4a`. Preserve this binding when deriving a compiler cache. Archive-model caches instead bind an archive SHA-256 and cannot be relabeled as raw-extent caches.

## Native cache identities and acceptance

Each model has its own native bake and strict reuse derivative. The strict derivative adds the official sampling provider while preserving every original compiler entry and memory-plan file.

| Model | Native bake cache | Strict reuse cache |
| --- | --- | --- |
| 0.8B | `92cfb6239dea59dc41b3421c5d64dc98dbd15ab4b0f94dbdbb9e57bbe5a4511b` | `792688c5aab3f534f5cf9f4f36f8cb250a7e0a5b22863fea1cd05d0b9e746d17` |
| 2B | `e005a45a52d975987e1ccb214632e4cae6c74af41c6fdfa98c2632158c77c8c3` | `9c96f77edd1caf601673c0d594ef98e3456b0e71739536c97932a2f972269475` |
| 4B | `4dcca475c3e4adea7cf329ec546b783890c4fc1855528d31c41b4dc2ea81ea8f` | `94bf504907877d2d90b1cea80719e8531a8fc8081d220fe5eb495f28a68666b6` |

The bake receipts are [0.8B preparation](../experiments/qwen-0.8b-extents-cache-bake.json), [2B preparation](../experiments/qwen-2b-extents-cache-bake.json), and [4B preparation](../experiments/qwen-4b-extents-cache-bake.json). Fresh strict reuse passed 34 calls per model: one cold call, one hot call, and a 32-call burst. The [0.8B reuse](../experiments/qwen-0.8b-extents-cache-reuse.json), [2B reuse](../experiments/qwen-2b-extents-cache-reuse.json), and [4B reuse](../experiments/qwen-4b-extents-cache-reuse.json) receipts record 102 completed calls in total, native AOT loading required, FlashInfer JIT disabled, and independent GET 404 cleanup. These smoke tests establish execution of the exported binaries under the recorded configurations. They do not replace the full scheduler gauntlet or establish acceptance of the persisted memory plan.

## Per-model cache preparation

Start with a single-model configuration that binds the final runtime, weight loader, engine settings, and launch program. Set its cache list to empty. Omit `VLLM_FORCE_AOT_LOAD` and `FLASHINFER_DISABLE_JIT` during the bake so native compilation can generate that model's cache. The preparation configurations are `configs/runpod/preparation/qwen-0.8b-extents.json`, `qwen-2b-extents.json`, and `qwen-4b-extents.json`; their corresponding plans are `configs/runpod/cache-bake-l40s-0.8b.json`, `cache-bake-l40s-2b.json`, and `cache-bake-l40s-4b.json`.

```sh
switchyard live bake \
  --config <single-model-fleet> --plan <matching-model-plan> \
  --env-file <provider-env> --env-file <storage-env> \
  --out <fresh-bake-output> --namespace switchyard/artifacts/cache

python scripts/serving-comparison/strict-cache.py \
  --bake <fresh-bake-output> --config <matching-cache-config> \
  --out <fresh-derivative> --sampling-provider-shim <pinned-shim-wheel>
```

The bake warms native inference, exports the cache, releases its GPU, and publishes the compiler artifact. Require successful export and a confirming provider GET 404 before deriving from its `cache-configuration.json`. The strict-cache script verifies the source artifact and registers its baked `sampling.so` through `flashinfer.jit_cache.providers`. It uses the official FlashInfer 0.7.0.post1+cu129 shim, SHA-256 `314d97ea8c3d288d0afce09e9f3dc823a83d4a219c02b85097de71ee143cf0fb`, and sets `VLLM_FORCE_AOT_LOAD=1`, `FLASHINFER_DISABLE_JIT=1`, and `FLASHINFER_NO_DOWNLOAD=1`.

Publish the verified derivative with `switchyard artifacts publish`, then run fresh reuse under the matching runtime, model, loader, launch program, and GPU. Require native Torch AOT load, native FlashInfer sampling load with JIT disabled, completed correctness and concurrency cases, and confirmed cleanup. Cache fingerprints establish the declared identity; the fresh GPU run establishes that the exported binaries execute.

## Provider discovery and compiled paths

Compiled modules can retain paths from their original bake. Preserve that prefix or create a checked alias before serving imports. Python can cache a nonexistent `PYTHONPATH` directory before the launcher creates the alias, so call `importlib.invalidate_caches()` after the alias exists and before discovering native providers. A fork inherits both the importer cache and FlashInfer's provider inventory.

The native CPU reproduction in `scripts/serving-comparison/check-provider-alias.py` finds zero providers in the parent process and its fork without invalidation, while a fresh interpreter finds one. After invalidation, all three find the provider. [Provider alias checks](serving-comparison-data/fla-native-provider-alias-checks.json) retain this evidence. The GPU receipts independently record the negative importer entry, shim visibility after invalidation, and successful native provider loading.

## Optional FLA platform-query patch

The FLA descriptor is `deploy/hydration/patches/vllm-0.31.0-fla-nvml.json`, SHA-256 `5b935fc5f4eacaff29876d2c9304df0b54ebe69828c6c3da8916ba36e54135f5`. It transforms the pinned source module from `d6c90d36505e02086954c23ca4dd2fea82a69416d68ff1d0501e16b3148b5b0c` to `3f19edd2c7aba0cb6a3251715199c3fe82ad1b20b7946b24b192d23bec82b828`. The runtime builder verifies the descriptor and target paths, descriptor digest, installed package version, source digest, exact replacement matches, and result digest before writing the patch set. It then rebuilds bytecode and records the applied patches in `build-info.json`.

The patch uses NVML for platform queries only when Torch CUDA and the driver are uninitialized, one GPU is visible, and either UUID visibility resolves the selected device or only one physical GPU exists. Initialized drivers, query failures, and ambiguous numeric ordering use the original Torch queries. The upstream CUDA and NUMA multiprocessing guards remain active. The driver check calls `cuCtxGetCurrent` without calling `cuInit`; the [CUDA Driver API initialization contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__INITIALIZE.html) defines the uninitialized status.

The patched SM89 runtime is `808e0a7b84d878c32de76f5da6a4b05894e1d61a013c9ccb000b1ea7f59bf9bd`. Relative to the unpatched scoped runtime `df7b2f985cb03bda92fbb33572969a85e963b9d2c3c2b7f8543b3fd75d39bc4e`, only the declared source module, its Python 3.12 bytecode, and provenance metadata change. The scope removes universal cubin payloads while preserving compiler files, native libraries, headers, package metadata, and directory paths. This asset profile is accepted only for the specified 0.8B BF16 FlashAttention workload on SM89.

Both the [initial paired GPU test](serving-comparison-data/fla-gpu-acceptance.json) and [reverse-order repeat](serving-comparison-data/fla-reverse-order-acceptance.json) completed 222 calls and nine literal checks per variant, retained native fork for the patched runtime, and loaded Torch AOT plus native sampling. The original patched cache is `ccc33cfaec8d0e669950f4b61b229fb5062ba46e8e17be1b76a613c10e349c69`; the observer-enabled repeat derivative is `dc5a08f614fae7856f4b0f2461c689cda2b0f21acfd8990f13e57576cb4ef883`. The observer runs in a separate stdlib process and preserves all compiler entries.

The process-start improvement repeated, but aggregate rates were lower in both pairs and some native aggregate latency increased. [Native timing evidence](serving-comparison-data/fla-native-decode-comparison.json) separates those residuals from serial client dispatch delays. The patch remains unselected. Its GPU acceptance binds the 0.8B archive SHA-256 `bc98fc9c5e6c9f00289e7fb63882f7a34e1c003e05da071d38a0adfc16a09178`; raw-extent loading and larger models need separate patch acceptance. The strict raw-extent fleet therefore retains the complete unpatched runtime.
