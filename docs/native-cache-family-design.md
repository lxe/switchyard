# Shared native runtime and driver cache publication

The final native fleet uses one runtime artifact for the accepted engine binary and a separate CUDA cache artifact for each model and GPU. Baseline receipts use model-specific runtime configurations and direct-controller driver selection. Those artifacts remain historical measurements of source `600d6dd`; the final families are independently baked against the accepted Ubuntu binary from source `1c966bb`.

## Shared runtime

The runtime contains the native engine, launch wrapper, and its Ubuntu CUDA/C++ library closure. Model configurations belong in preparation artifacts. The live native adapter passes each selected model's fixed configuration through `--config-json`, so no model-specific configuration file belongs in the shared runtime. The runtime manifest binds the accepted source, Ubuntu binary SHA-256, toolkit/compiler versions, CPU architecture, and GPU compiler target. Offline diagnostic binaries stay in preparation outputs.

The shared runtime is published from the accepted Ubuntu build after its file inventory is verified offline. Baseline model archives are reused without repacking because their native format and pinned revisions remain accepted. CUDA caches are baked again against the new binary and exact model configuration; each published family records those new component identities.

## Driver cache family

A driver cache family is one immutable artifact containing independently baked CUDA driver-cache seeds for one model, runtime, configuration, and GPU. Its layout is:

```text
driver-family.json
drivers/<exact-driver>/bake.json
drivers/<exact-driver>/identity.tsv
drivers/<exact-driver>/component-manifest.json
drivers/<exact-driver>/cuda/...
```

Each observed driver has its own directory. The published final families currently contain the directory for 580.159.04.

`driver-family.json` records the shared identities and every included component artifact ID, exact driver, bake receipt, and cache-file inventory. Each component retains its original bytes and driver identity. A family containing only one validated driver is valid; it accepts only that driver. The family manifest describes the layout and supported driver set rather than claiming a single exact `driver_version` for all contents.

The shared launch wrapper observes GPU name, compute capability, and driver through `nvidia-smi`. It rejects an unsupported GPU, architecture, or missing driver directory. It then sets `CUDA_CACHE_PATH` to the matching directory inside the private hydrated cache seed. It does not merge driver caches, infer compatibility between drivers, hash complete runtime/model files during serving, or run a readiness inference. Its behavior and environment must be finalized before deriving live cache labels.

## Live cache bindings and acceptance

The existing scheduler selects serving caches by GPU name and uses one shared runtime environment. Each model therefore refers to one driver cache family named `gpu-kernels` for that GPU. The artifact must contain the generic labels produced by Rust `artifacts::cache_requirements`, including `runtime_id`, `model_revision`, `model_repository`, the model archive identity, `engine_configuration`, and `runtime_launch_configuration`. Native labels add binary, native configuration, compiler/toolkit, architecture, and family-layout identity. Component receipts retain the exact driver labels.

After pinning the shared runtime and model archive references in a cache-free fleet file, derive the generic labels with `switchyard live cache-bindings --config FLEET.json --gpu 'NVIDIA GeForce RTX 4090' --out LABELS.json`. This command uses the live Rust serialization and fingerprint code without credentials, network access, or allocation. Publication must merge additional native labels without overriding any generated identity.

Family publication requires offline component identity checks and confirmed bake-pod cleanup. Direct cloud acceptance records selection of the correct private directory, streamed content on a fresh pod, and confirmed serving-pod cleanup. Wrapper fixtures establish missing-driver rejection. The generic scheduler preflight and workload gauntlet are separate acceptance gates; a successful direct launch does not establish their results. This format does not implement direct incoming GPU tensor loading or a hard GPU allocator ceiling.

## Preparation tools

`prepare-shared-runtime.py` inventories the accepted Ubuntu export and publishes only its serving binary, wrapper, and libraries. It emits a common launch configuration with the private family-cache root, expected GPU and capability, library path, enabled driver cache, fixed cache size, and zero hydration retries. `prepare_artifacts.py --shared-runtime` reuses that runtime ID and places each model's native configuration beside its offline bake executable and requests. `run-bake.py` then reads that preparation configuration while retaining the maximum equal-length cohort and varied prompts. The [native bake platform controller](native-bake-platform.md) can run several fixed models sequentially on one owned driver host using its own allocation profile.

`prepare-family.py --fleet-config FLEET.json` invokes the Rust `live cache-bindings` command itself. It checks the component binary and physical capability against the fleet's shared runtime, compares every baked native bound with the selected live model, verifies the complete component tree using Rust, and rejects duplicate driver variants or unconfirmed bake-pod cleanup. Packing has no network access. Supplying `--publish-config` publishes the packed family only after these checks; the publication receipt becomes complete after the manifest upload succeeds.

The generic live model reference reads a legacy archive manifest, while the native artifact publisher stores a different artifact manifest at the original prefix. `publish-live-model.py` therefore copies the already published archive bytes within R2 to a separate immutable live prefix and publishes the converted legacy manifest last. It hashes local source archives before copying and checks copied object lengths; it does not claim a full remote hash readback for a new copy. Existing-object reuse requires the existing complete-byte hash check. R2 supports the required server-copy operations in its [S3 API](https://developers.cloudflare.com/r2/api/s3/api/). The AWS client uses bounded multipart copies for large objects and explicit small-object copies that omit unsupported tagging headers.

The narrow wrapper fixtures use a fake `nvidia-smi` and stub engine. They verify observed-driver selection, rejection of missing driver/GPU/capability or changed component identity before execution, and exact forwarding of literal inline JSON arguments. Run them with `python -m unittest discover -s deploy/native/tests -p test_driver_family.py`. They establish launch control behavior; hardware and scheduler acceptance remain separate.
