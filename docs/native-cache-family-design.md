# Shared native runtime and driver cache publication

The final native fleet needs one runtime artifact for the accepted engine binary and a separate CUDA cache artifact for each model and GPU. The current baseline receipts use model-specific runtime configurations and direct-controller driver selection. Those artifacts remain historical measurements of source `600d6dd`; they do not satisfy the final shared scheduler configuration.

## Shared runtime

The runtime contains the native engine, launch wrapper, and its Ubuntu CUDA/C++ library closure. Model configurations belong in preparation artifacts. The live native adapter passes each selected model's fixed configuration through `--config-json`, so no model-specific configuration file belongs in the shared runtime. The runtime manifest binds the accepted source, Ubuntu binary SHA-256, toolkit/compiler versions, CPU architecture, and GPU compiler target. Offline diagnostic binaries stay in preparation outputs.

Publication waits for the final serving-source freeze and a new Ubuntu build. Baseline model archives can be reused without repacking when their native format and pinned revision remain accepted. CUDA caches must be baked again against the new binary and exact model configuration; baseline caches cannot be relabeled as final caches.

## Driver cache family

A driver cache family is one immutable artifact containing independently baked CUDA driver-cache seeds for one model, runtime, configuration, and GPU. Its proposed layout is:

```text
driver-family.json
drivers/580.126.20/bake.json
drivers/580.126.20/cuda/...
drivers/580.159.04/bake.json
drivers/580.159.04/cuda/...
```

`driver-family.json` records the shared identities and every included component artifact ID, exact driver, bake receipt, and cache-file inventory. Each component retains its original bytes and driver identity. A family containing only one validated driver is valid; it accepts only that driver. The family manifest describes the layout and supported driver set rather than claiming a single exact `driver_version` for all contents.

The shared launch wrapper observes GPU name, compute capability, and driver through `nvidia-smi`. It rejects an unsupported GPU, architecture, or missing driver directory. It then sets `CUDA_CACHE_PATH` to the matching directory inside the private hydrated cache seed. It does not merge driver caches, infer compatibility between drivers, hash complete runtime/model files during serving, or run a readiness inference. Its behavior and environment must be finalized before deriving live cache labels.

## Live cache bindings and acceptance

The existing scheduler selects serving caches by GPU name and uses one shared runtime environment. Each model therefore refers to one driver cache family named `gpu-kernels` for that GPU. The artifact must contain the generic labels produced by Rust `artifacts::cache_requirements`, including `runtime_id`, `model_revision`, `model_repository`, the model archive identity, `engine_configuration`, and `runtime_launch_configuration`. Native labels add binary, native configuration, compiler/toolkit, architecture, and family-layout identity. Component receipts retain the exact driver labels.

After pinning the shared runtime and model archive references in a cache-free fleet file, derive the generic labels with `switchyard live cache-bindings --config FLEET.json --gpu 'NVIDIA GeForce RTX 4090' --out LABELS.json`. This command uses the live Rust serialization and fingerprint code without credentials, network access, or allocation. Publication must merge additional native labels without overriding any generated identity.

Review the wrapper and family validation before final family publication. Acceptance requires offline component identity checks, missing-driver rejection, observed selection of the correct private directory, and a real fresh scheduler launch with streamed content and confirmed owned-pod cleanup. The final workload gauntlet remains a separate scheduler test. This design does not implement direct incoming GPU tensor loading or a hard GPU allocator ceiling.
