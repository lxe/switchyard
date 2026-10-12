# Native GPU build artifacts

`scripts/build-native-gpu.py` compiles the native Candle backend and Switchyard paged-attention kernel for an explicit list of CUDA architectures in one invocation. Each target has its own Cargo build directory, PTX, static CUDA archive, optional standalone cubins, and compatibility receipt. The default job also exports the three native engine binaries. Compilation and inspection use CPU tools; the job never launches an engine, executes a GPU kernel, discovers the attached GPU, or runs a model bake.

## Build commands

Run from a Linux GNU builder with Rust, the CUDA toolkit, a host C++ compiler, and `readelf`. Release binaries should be built inside the same pinned base-image family used by the serving runtime. A local Arch executable does not establish Ubuntu ABI compatibility.

```sh
python scripts/build-native-gpu.py \
  --arch 89 --arch 120a \
  --cuda-root /usr/local/cuda \
  --builder-id 'nvidia/cuda:<version>-devel-ubuntu24.04@sha256:<digest>' \
  --out results/native-gpu-builds/release-01 --jobs 2
```

`--arch` is repeatable and has no default. Values identify compiler targets, rather than marketing names or a request to inspect the current GPU. Targets at or above 90 require an explicit suffix because the pinned Candle compiler helper otherwise adds one. For example, select `90a` for the helper's Hopper target and `120a` for its RTX 5090 target. The CUDA toolkit must support the requested base architecture, and the emitted PTX and device objects must report the exact suffixed target.

Use `--plan` to resolve sources, dependencies, compiler identities, and targets without compiling or creating output. `--kernels-only` builds the engine library to enable its Candle CUDA dependency and exports only kernel files. `--no-cubins` omits standalone PTX assembly. `--offline` requires the locked Cargo dependencies to be cached already. `--build-root` selects a separate reusable compiler cache; the default is `target/native-gpu-builds/<compiler-cache-key>` in this checkout. Architecture builds run sequentially, with `--jobs` bounding Cargo and Cudaforge CPU workers.

The output directory must be fresh. Each target's `build.json` appears only after all its files pass inspection. The top-level `build.json` appears only when every selected target succeeds. A failed job retains logs and `failure.json` but has no complete matrix receipt. Earlier successful target receipts remain usable individually; consumers must not treat an incomplete matrix as a completed release.

## Ubuntu CUDA container build

`scripts/build-native-gpu-container.py` runs the matrix job inside the digest-pinned CUDA 13.0.0 Ubuntu 24.04 builder. It builds both targets sequentially in one container invocation, then creates a runtime directory for each target using the existing native packager's receipt writer. Run this command after the final engine source is merged and committed:

```sh
python /home/lxe/switchyard/scripts/build-native-gpu-container.py \
  --workspace /home/lxe/switchyard \
  --arch 89 --arch 120a \
  --image 'nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d' \
  --out /home/lxe/switchyard/results/native-gpu-builds/ubuntu13-final-01 \
  --cache-root /home/lxe/switchyard/target/native-gpu-container \
  --jobs 2 --memory 8g
```

The image above is also the default. Docker must be available, and the builder needs network access to install Ubuntu build tools and Rust 1.98.1 and fetch locked Cargo dependencies. The image digest pins the CUDA base; Ubuntu package versions and compiler identities are recorded because package installation can change independently. CPU and memory limits apply to the container. Add `--plan` to inspect the source commit, image, target arguments, and Docker command without creating outputs or starting a container.

The wrapper rejects uncommitted changes or untracked files under the Cargo manifests, `.cargo`, and `crates`. It archives the selected Git commit into a temporary source directory and mounts that snapshot read-only. The snapshot contains no `.git` directory, so the inner job receives the original commit through `--source-commit`; its receipt still hashes the actual engine, native-kernel and Candle input contents. The separately mounted build scripts are retained in `container-inputs`, and `producer.py` retains the exact inner matrix producer.

The container receives no GPU devices and does not execute an engine or a bake. It retains `ubuntu-os-release.txt`, `ubuntu-packages.txt`, `ubuntu-libc.txt`, `ubuntu-rustc.txt`, and `ubuntu-nvcc.txt` at the output root. Its `container-evidence` directory contains the receipt helper's six required inputs: `nvcc-gpu-arch.txt`, `cuda-environment-version.txt`, `cuda-packages.txt`, `nvcc.txt`, `os-release.txt`, and `architecture.txt`. These come from the container's compiler, `CUDA_VERSION`, installed CUDA packages, OS release, and machine architecture. The runtime receipt validates agreement among them and records the toolkit release, compiler version, and installed package versions. Each target also keeps the matrix job's ELF library and symbol-version requirements. The wrapper validates the matrix and target identities, every exported executable and kernel file, and the retained producer hash before assembling that target's runtime.

For the command above, pass `targets/sm_89/runtime` or `targets/sm_120a/runtime` under the output directory to the existing packager's `--runtime` option. Each contains `build.json` in the existing `switchyard-native-runtime-build-v1` format, `tree/bin/switchyard-engine`, `tree/bin/switchyard-engine-bake`, and `tree/lib` with the resolved CUDA toolkit and C++ runtime libraries. The wrapper records each executable's `ldd` output beside the runtime receipt. Ubuntu system libc libraries remain supplied by the serving base image, while the host NVIDIA driver supplies `libcuda.so.1`; neither driver libraries nor link-time driver stubs are bundled. An unresolved non-driver dependency fails assembly.

The physical GPU capability in the runtime receipt is `8.9` or `12.0`, as required by the packager. Its separate `cuda_compiler_target` field records `compute_89` or `compute_120a`, supplied through the receipt helper's optional `--compiler-target` argument. Existing callers that omit this argument retain their numeric compiler-target field. The helper checks that the compiler target's base agrees with the numeric physical capability and appears in `nvcc --list-gpu-arch`; that listing reports base architectures. The matrix's PTX and device-object inspections establish the exact suffixed output target and retain `sm_89` or `sm_120a` independently.

The packager requires the runtime source commit to match `engine_source_commit` in its configuration. The wrapper publishes the original matrix `build.json` at the output root only after both runtime directories are assembled, preserving the matrix identity. Failed container jobs retain `container.log`, partial output under `matrix`, and `container-failure.json`, without publishing a completed root matrix. Runtime directory creation does not establish GPU, numerical, serving, driver-cache, or model-bake acceptance.

## Artifact contents

For `--arch 89`, the output paths are:

| Path | Contents and consumer |
| --- | --- |
| `targets/sm_89/bin/switchyard-engine` | Native serving executable with Candle and native paged-attention PTX embedded. |
| `targets/sm_89/bin/switchyard-engine-bake` | Separate GPU-dependent bake executable; building it does not run a bake. |
| `targets/sm_89/bin/switchyard-engine-bench` | Benchmark executable; the build job never runs it. |
| `targets/sm_89/kernels/ptx/*.ptx` | Candle and native paged-attention PTX, checked for target and PTX ISA version. |
| `targets/sm_89/kernels/libmoe.a` | Candle's static CUDA archive, inspected with `cuobjdump`. |
| `targets/sm_89/kernels/cubin/*.cubin` | Optional offline `ptxas` outputs, inspected as CUDA ELF device code. |
| `targets/sm_89/build.json` | Compatibility inputs, file hashes, artifact identity, runtime library requirements, and validation state. |
| `build.json` | Completed target list and matrix identity. |

The exporter independently inspects the `switchyard-engine-kernels` build output and includes `paged_decode.ptx` and its optional cubin. Native CUDA source and build-script changes invalidate compiler identities. A missing native build output or a mismatched native target prevents publication, including in `--kernels-only` mode.

The matrix exporter requires the three binaries listed above and exports only those names. Additional Cargo binary targets, including the offline `switchyard-engine-merged-probe` and `switchyard-engine-mixed-bench` diagnostics, are recorded in `excluded_binary_targets` and omitted from the artifact. They may still be compiled by Cargo's `--bins` command. The Ubuntu wrapper's runtime directory contains the serving and bake binaries; the benchmark remains available only in the matrix target's `bin` directory.

The native RunPod packager uses the serving and bake binaries and verifies their toolkit-library closure. The container wrapper supplies its existing runtime input layout without modifying deployment scripts or configuration. Driver libraries and link-time `libcuda` stubs are not copied into these artifacts.

Standalone cubins are build outputs for inspection and later backend integration. The current pinned Candle loader reads embedded PTX; merely shipping sibling cubins does not make that loader consume them or eliminate its driver JIT. A loader change needs its own modular backend implementation and hardware proof.

## Compatibility keys

Each target's `compatibility_key` hashes the following inputs. `artifact_id` additionally binds the emitted file hashes and validation fields. A compatibility key identifies compatible build inputs; it does not promise bit-for-bit deterministic compiler output. The retained `producer.py` has the exact build-job hash in each receipt.

| Input | Why it is bound |
| --- | --- |
| Source commit and engine/native-kernel/Candle input-content hash | Prevents source changes, including untracked engine sources, from sharing a build cache. |
| Cargo.lock digest and Candle/Cudaforge/Cudarc versions and sources | Pins the backend and dependency resolution. |
| Exact `sm_XX` or suffixed target | Keeps CUDA machine-code and PTX targets separate. |
| Rust target, Rust compiler, host libc, and host C++ identity | Records host ABI and compiler dependencies. Engine receipts also list ELF library and symbol-version requirements. |
| CUDA compiler, assembler, and inspection-tool hashes and versions | Prevents toolkit updates from silently reusing build directories. |
| Builder identity, Cargo configuration hashes, and relevant compiler environment | Binds the release image pin, source configuration, and inherited Rust/compiler flags. |
| Build-job hash and output mode | Separates changes to preparation and cubin or executable selection. |

Production builds should provide an immutable `--builder-id` and retain the corresponding image. The job records the supplied identity; it does not verify that the invocation is running inside that image. The builder's system libraries remain part of that controlled builder environment.

`compiler_cache_key` omits the source-commit label, export mode, and build-job hash while retaining actual source content, architecture, compiler inputs, and an explicit Cargo command-policy version. This permits a documentation commit or receipt-only change to reuse identical compiler outputs. Cargo still decides which executable or library targets need rebuilding. The full artifact compatibility key continues to bind provenance and export policy.

Driver JIT caches have a separate lifetime and compatibility boundary: their generated code depends on the driver, GPU target, PTX, and JIT options. This job does not capture them. Model bake artifacts likewise remain separate and must retain model/configuration, engine binary, GPU and driver identities, plus their existing acceptance gates. CUDA contexts, allocations, graphs, and memory sizing are never part of a compiled-kernel artifact.

NVIDIA documents ordinary PTX forward compatibility and stricter limits for architecture-specific targets such as `compute_90a` and `compute_100a`. This job conservatively selects an exact target; it does not relabel a suffixed artifact as a universal fallback. See the [Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).

## Upstream build controls

The pinned [Candle build script](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-kernels/build.rs) uses Cudaforge to generate PTX and `libmoe.a`. It resolves one architecture per build through `CUDA_COMPUTE_CAP`. This job preserves that upstream build and runs a separate target build for each selected architecture; it does not patch Candle or introduce a multi-target runtime dispatcher.

[llama.cpp's CUDA CMake configuration](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt) accepts `CMAKE_CUDA_ARCHITECTURES` and distinguishes real device-code targets from virtual PTX targets. Its explicit target list is the useful build pattern here. Candle's single-target build contract makes separate target artifacts the smaller implementation for the current backend.

## Validation

Run the CPU fixtures with:

```sh
python -m unittest discover -s scripts/tests -p 'test_native_gpu*.py'
```

The fixtures verify target isolation, rejection of mismatched PTX and device archives, missing completion receipts on compiler failure, source-key invalidation, and exclusion of both extra diagnostics. Container fixtures check the pinned CPU-only command, rejection of uncommitted Rust sources, driver exclusion, corruption checks for every export category and producer source, and publication of the unchanged matrix identity. They execute the checked-in runtime receipt script with a complete fixture evidence set, verify physical capability separately from the suffixed compiler target, and reject missing evidence or inconsistent targets. These fixtures do not launch Docker or compile CUDA.

Hardware acceptance remains separate: load the selected engine on each intended GPU and driver combination, run numerical and serving gates, measure cold/hot startup, and verify any GPU-dependent bake before publishing it as accepted. Compile success and device-object inspection do not establish those results.

The [CPU acceptance receipt](receipts/native-gpu-builds-cpu.json) records a real local build from source commit `600d6dd`, using CUDA 13.4 and a Linux GNU host. One invocation produced `sm_89` and `sm_120a` sets, each containing three executables, 11 PTX files, 11 standalone cubins, and one static CUDA archive. All 52 exported file hashes and the retained producer hash were verified. The serving executable in each set contains only its selected embedded PTX target; the bake and benchmark helpers do not embed PTX.

That export reused compiler outputs from the same sources and toolchain; it is not a cold-build timing result. PTX ISA 9.4 is recorded explicitly, so an older driver must not be assumed to load those local CUDA 13.4 outputs. Compilation and inspection alone do not establish GPU execution or cache portability.

The [Ubuntu matrix receipt](receipts/native-gpu-builds-ubuntu.json) records the accepted serving source built in the digest-pinned Ubuntu 24.04 CUDA 13.0 image. One CPU-only invocation produced separate SM89 and SM120a targets and runtime library closures. All 52 exported target hashes and both runtime inventories were independently checked after publication of the completion marker. Physical capability and compiler target remain separate fields; no GPU was exposed to the builder, and its container was removed.

| Target | Serving binary SHA-256 |
| --- | --- |
| SM89 | `410e210f203803846772ccc6a031bbbbb9ce1e1a0e420ea82f81b738d5163355` |
| SM120a | `28873160d343e377e74792abe8510cfae44c8a2ec2c09f55c18a03e20379f9d8` |

These identities differ from the earlier local CUDA 13.4 engine. Each requires its own exact-binary driver-cache bake and hardware acceptance; earlier caches and serving measurements cannot be relabeled for this build.

The [SM120a local fleet acceptance](receipts/native-fleet-ubuntu-rtx5090.json) separately records fresh driver-cache bakes, fresh-process HTTP first-token measurements and 51 useful live-client streams across 0.8B, 2B and 4B on an RTX 5090 with driver 615.71.09. The executable used its exported CUDA 13.0 libraries. These BF16 serving measurements do not repeat the F32 reference gate or establish SM89 cloud acceptance.

## Paged engine matrix

The [paged engine Ubuntu receipt](receipts/native-paged-gpu-builds-ubuntu.json) records source `aa67c22` built with the same digest-pinned CUDA 13.0 Ubuntu builder. Both targets now export 12 PTX files, 12 standalone cubins, one static archive and three executables. All 56 exported file hashes, both assembled runtime inventories and the producer hash were independently verified; the owned CPU builder was removed.

| Target | Serving binary SHA-256 | Runtime bytes including offline bake helper |
| --- | --- | ---: |
| SM89 | `deddcde553ba5308e38c201c1933e90c6cbf16a57ac09894348c8f8d6dc3f26e` | 832,088,032 |
| SM120a | `37774358815f7fa83b495912655676a238cdaa89a090ab0aa302e659ddff94f8` | 856,977,504 |

These binaries include native paged decode. Their identities supersede earlier binaries only for deployments that explicitly select the new runtime and compatible model/cache configuration. The completed matrix does not establish GPU numerical parity, driver-cache coverage, R2 publication or RunPod serving acceptance. Old runtime families remain pinned to their original binaries.

The [SM120a generation receipt](receipts/native-paged-ubuntu-generation-rtx5090.json) separately checks the new Ubuntu CUDA 13.0 executable on RTX 5090 against the previously accepted local Arch CUDA 13.4 F32 binary. Both execute identical offline requests with the same verified native weight artifacts and fixed prefix-cache configuration. All 48 rows and 1,536 output token IDs match across the three models, including varied lengths and repeated prefixes. This checks complete generated sequences on one GPU; it does not replace the teacher-forced published-model gate, establish RTX 4090 parity, or measure serving cold-start performance.
