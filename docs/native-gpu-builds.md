# Native GPU build artifacts

`scripts/build-native-gpu.py` compiles the native Candle backend for an explicit list of CUDA architectures in one invocation. Each target has its own Cargo build directory, PTX, static CUDA archive, optional standalone cubins, and compatibility receipt. The default job also exports the three native engine binaries. Compilation and inspection use CPU tools; the job never launches an engine, executes a GPU kernel, discovers the attached GPU, or runs a model bake.

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

## Artifact contents

For `--arch 89`, the output paths are:

| Path | Contents and consumer |
| --- | --- |
| `targets/sm_89/bin/switchyard-engine` | Native serving executable with Candle's PTX embedded. |
| `targets/sm_89/bin/switchyard-engine-bake` | Separate GPU-dependent bake executable; building it does not run a bake. |
| `targets/sm_89/bin/switchyard-engine-bench` | Benchmark executable; the build job never runs it. |
| `targets/sm_89/kernels/ptx/*.ptx` | Candle-generated PTX, checked for target and PTX ISA version. |
| `targets/sm_89/kernels/libmoe.a` | Candle's static CUDA archive, inspected with `cuobjdump`. |
| `targets/sm_89/kernels/cubin/*.cubin` | Optional offline `ptxas` outputs, inspected as CUDA ELF device code. |
| `targets/sm_89/build.json` | Compatibility inputs, file hashes, artifact identity, runtime library requirements, and validation state. |
| `build.json` | Completed target list and matrix identity. |

The native RunPod packager uses `bin/switchyard-engine` and `bin/switchyard-engine-bake` from the selected target, then assembles and verifies its toolkit-library closure. This build job does not modify RunPod configuration, runtime packaging, or its output directories. Driver libraries and link-time `libcuda` stubs are not copied into these artifacts.

Standalone cubins are build outputs for inspection and later backend integration. The current pinned Candle loader reads embedded PTX; merely shipping sibling cubins does not make that loader consume them or eliminate its driver JIT. A loader change needs its own modular backend implementation and hardware proof.

## Compatibility keys

Each target's `compatibility_key` hashes the following inputs. `artifact_id` additionally binds the emitted file hashes and validation fields. A compatibility key identifies compatible build inputs; it does not promise bit-for-bit deterministic compiler output. The retained `producer.py` has the exact build-job hash in each receipt.

| Input | Why it is bound |
| --- | --- |
| Source commit and engine/Candle input-content hash | Prevents source changes, including untracked engine sources, from sharing a build cache. |
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
python -m unittest discover -s scripts/tests -p test_native_gpu_builds.py
```

The fixtures verify target isolation, rejection of mismatched PTX and device archives, missing completion receipts on compiler failure, and source-key invalidation. Hardware acceptance remains separate: load the selected engine on each intended GPU and driver combination, run numerical and serving gates, measure cold/hot startup, and verify any GPU-dependent bake before publishing it as accepted. Compile success and device-object inspection do not establish those results.

The [CPU acceptance receipt](receipts/native-gpu-builds-cpu.json) records a real local build from source commit `600d6dd`, using CUDA 13.4 and a Linux GNU host. One invocation produced `sm_89` and `sm_120a` sets, each containing three executables, 11 PTX files, 11 standalone cubins, and one static CUDA archive. All 52 exported file hashes and the retained producer hash were verified. The serving executable in each set contains only its selected embedded PTX target; the bake and benchmark helpers do not embed PTX.

The final export reused compiler outputs from the same sources and toolchain; it is not a cold-build timing result. PTX ISA 9.4 is recorded explicitly, so an older driver must not be assumed to load these particular local outputs. The receipt establishes compilation and inspection, with GPU execution, numerical acceptance, driver-cache portability, and model-bake acceptance still unvalidated. A production Ubuntu release remains a separate invocation in the pinned serving-compatible builder.
