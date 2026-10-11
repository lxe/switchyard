# Runtime component packaging

A runtime composition consists of one boot artifact and named component artifacts. Boot contains the Python interpreter, installed package code, package metadata, and native libraries needed to import the serving API. A component contains complete directories needed after import, such as CUDA, the C++ toolchain, and packaged FlashInfer cubins.

Each tree uses the existing `switchyard-artifact-v1` format. Boot is published at `/opt/switchyard/runtime`; components are published as complete sibling directories under `/opt/switchyard/components/<name>`. The roots must be disjoint, so the boot verifier sees the same immutable tree before and after component publication.

## Partition recipe

The [vLLM 0.31 recipe](../deploy/hydration/runtime-components/vllm-0.31.json) moves complete `cuda`, `toolchain`, and `flashinfer_cubin/cubins` directories into `build`. FlashInfer package code remains in boot. Its official `FLASHINFER_CUBIN_DIR` setting names the sibling resource directory; `FLASHINFER_NO_DOWNLOAD=1` prevents missing resources from triggering downloads.

| Field | Definition |
| --- | --- |
| `format`, `schema_version` | `switchyard-runtime-partition-v1`, version `1`. Unknown fields are rejected. |
| `runtime_root`, `components_root` | Pinned disjoint absolute roots. Nested roots and shell metacharacters are rejected. |
| `requires` | Compatibility values that must match the complete source manifest. |
| `components[].name` | Unique name using letters, digits, underscores, or hyphens; `boot` is reserved. |
| `components[].moves` | Complete source directories and component-relative destinations. Sources and destinations cannot overlap. |
| `environment` | Launch values with `{runtime_root}` and `{components_root}` expansion. |
| `library_paths` | Boot-relative native loader directories; `*` matches one complete path component. |
| `external_libraries` | Explicit base-image or NVIDIA-driver dependencies, reported separately from packaged libraries. |
| `toolchain_wrappers` | Relocated toolchain roots and GCC major versions. Original wrappers are preserved; generated wrappers derive their own prefix. |
| `boot_imports`, `resource_checks`, `compiler_checks` | Imports before publication, official resource-loader checks after publication, and CPU compiler acceptance commands. |

Every source entry appears exactly once in the inventory with its component, destination, mode, and original digest or link target. No header, native library, compiler file, cubin, checksum, or other source payload is pruned. Original boot metadata and environment are preserved verbatim under `partition-source`; original compiler wrappers are preserved under the component's `partition-source`. Generated metadata, exports, and relocated wrappers are additional files. Symlinks are rewritten for relocation and remain inside one artifact.

## Preparation command

Composition runs in Rust through `switchyard-runtime-build` and `switchyard-artifact-tools`. Python is used only for package imports and resource smoke checks inside the serving interpreter. Preparation requires Linux, `cp`, `readelf`, and Docker with the acceptance image already present locally.

```sh
cargo build --release --features artifacts
target/release/switchyard artifacts partition-runtime \
  --source /path/to/complete/runtime \
  --manifest /path/to/complete/artifact/manifest.json \
  --recipe deploy/hydration/runtime-components/vllm-0.31.json \
  --patch-lock deploy/hydration/fla-source-patches.lock.json \
  --out /path/to/fresh/composition
```

The optional patch lock pins the existing [FLA descriptor](../deploy/hydration/patches/vllm-0.31.0-fla-nvml.json). The runtime builder applies its exact source and result hashes in a private copied full runtime, recompiles the changed module using unchecked-hash bytecode at the pinned prefix, and packs that complete source before partitioning. Package versions must match the installed inventory. Original build metadata and source lock are retained under `derivation-source`, along with patch input and application receipts. Omit `--patch-lock` for an already patched source; applying it twice fails the source hash check.

`--plan-only` produces the inventory and native report without component packing or serving imports. With a patch lock, it still derives and packs the complete source so the plan identifies that immutable source.

The normal build verifies the original source, copies it without mutable hardlink aliases, relocates directories in private staging, and audits every preserved byte and permission. It packs and verifies components with the ordinary archive library. After import and compiler acceptance, it packs and verifies boot, writes the descriptor, and publishes the whole local output with an exclusive directory rename. Existing destinations are never replaced. Failed builds retain diagnostic logs in a failure receipt and do not publish a composition directory.

## Native and Python acceptance

The native audit reads actual x86-64 ELF dependency tables. It resolves packaged libraries using declared directories, `$ORIGIN` paths, and local symlinks, then compares boot with the complete source. Newly lost dependencies fail preparation. Existing optional plugin dependencies remain recorded; the result does not promise every RDMA, MPI, or media backend.

Acceptance has no network or GPU devices, a read-only boot mount, and an empty component parent. One Python process imports Torch, the vLLM CLI and OpenAI API modules, and FlashInfer's environment and cubin loader while components are absent. It records loaded native libraries and confirms CUDA has not initialized. A loaded boot ELF with unresolved declared dependencies fails acceptance.

The Rust builder renames each complete component into the visible sibling parent and opens the gate. The same Python process retains its imported modules and resource path, loads an actual cubin through the official loader, and checks its size and SHA-256. The recipe also compiles and executes C++ code using standard library headers and compiles an SM89 cubin using relocated `nvcc`. These checks prove imports, deferred resource discovery, headers, and compiler execution; they do not execute GPU kernels or measure serving startup.

## Composition descriptor and cache preparation

`runtime-set.json` binds boot and component IDs, recipe, roots, environment, parent and patched source IDs, and the preparation tool hash. Local evidence includes `source-inventory.json`, `native-closure.json`, `recipe.json`, and `acceptance/receipt.json`. Packed outputs are `boot/artifact`, `components/<name>/artifact`, and, for derivation, `source/artifact`.

Copy only each component's `name`, `artifact`, and `compatibility` into a runtime ticket. Its `destination`, `manifest`, and `tree` fields describe local preparation output. Select the descriptor's boot artifact, compatibility, and environment together. The agent expands both root placeholders and gates serving work until every required component is published completely.

This changes runtime identity. Existing cache artifacts cannot be relabeled as compatible: a fresh model cache bake must bind the new boot ID, every component ID, and the complete launch environment. The descriptor marks `fresh_cache_bake_required`; fleet integration and GPU bake/serving acceptance are subsequent steps.

## Local acceptance receipt

The [local receipt](runtime-component-acceptance.json) records the FLA-patched complete source, boot and build IDs, exact packed-manifest coverage of all 185,583 source entries, and CPU acceptance. Boot has 4,562,590,833 compressed bytes; build has 682,116,732. The source and boot each retain 67 unresolved optional dependency edges; partitioning adds none. The serving import process loaded 161 packaged native libraries and verified deferred FlashInfer resource discovery, C++ execution, and SM89 cubin compilation with CUDA uninitialized. These are local packaging results; cache bake and GPU serving acceptance require the selected composition and fresh cache identities.
