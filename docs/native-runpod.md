# Native Rust RunPod deployment

The native RunPod path serves the Rust `switchyard-engine` binary from independently published runtime, model, and CUDA driver-cache artifacts. A small Ubuntu container downloads the static Rust agent. The agent hydrates all three artifacts concurrently, launches the native engine, and proxies its chat completion endpoint. The serving container requires no Python interpreter, Torch, vLLM, or CRIU.

The profiles under `configs/runpod/native/` select the corrected engine source, a pinned Qwen 3.5 model revision, an RTX 4090 allocation, CUDA 13.0, and fixed request limits. The measured deployments use BF16. Numerical accuracy follows the source commit's [accuracy receipt](native-engine-accuracy.md); cloud startup and throughput measurements do not establish additional numerical parity.

## Artifact contents and compatibility

Each artifact has its own immutable R2 prefix and manifest ID. Packing and hashing happen before a paid serving allocation. The fresh serving path uses trusted streaming and does not hash complete model or runtime files during startup.

| Artifact | Contents | Used during serving |
| --- | --- | --- |
| Runtime | Native engine, launch wrapper, native configuration, CUDA and C++ libraries | Yes |
| Model | Prepared native text weights, tokenizer, model configuration and chat template | Yes |
| GPU cache | Writable seed of CUDA driver compilation cache, native bake receipt | Yes |
| Preparation | Native bake executable, representative requests, build inventory | No |

The runtime labels identify the source commit, engine SHA-256, native configuration SHA-256, Ubuntu version, CUDA build version, CPU architecture, and GPU compute capability. The cache adds the selected `runtime_id`, `model_revision`, model artifact ID, GPU name, and exact driver version. The launch controller checks the published cache manifest before allocation. Optional `--cache-alternative PATH/prepared-baked.json` inputs add independently baked driver variants for the same runtime, configuration, and model. Bootstrap observes the actual host driver and selects its matching signed ticket before downloading the agent or hydrating artifacts; an unsupported driver stops the allocation without transferring model weights. The agent checks the runtime and model bindings before hydration, and the launch wrapper checks driver version and compute capability before executing the engine.

The cache seed is mounted under `{cache_root}/gpu-kernels`; its `cuda/` subdirectory becomes `CUDA_CACHE_PATH`. Cache files remain a separate artifact so a new engine build or configuration requires a new bake while unchanged model weights can be reused. The driver cache contains compiled GPU code, not a saved GPU process or initialized model.

The Ubuntu build bundles `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libcurand.so.10`, `libstdc++.so.6`, and `libgcc_s.so.1`. RunPod supplies `libcuda.so.1` from the host driver; the container supplies its ordinary Ubuntu C library and loader. The build output retains `ldd`, ELF version, compiler, and CUDA toolkit evidence. CUDA 13.0 requires a compatible host driver; see NVIDIA's [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

## Fixed resource limits

All three profiles bound context length at 128 tokens, batch size at eight, and queued requests at 64. The CUDA cache is capped at 512 MiB. Artifact transfers share twenty download slots; the Rust model receiver passes 64 KiB pieces through a four-piece bounded channel. Runtime and cache hydration retain their own manifest and transfer accounting.

| Model | Maximum native weight bytes | GPU admission bytes | Minimum host RAM |
| --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B | 1,610,612,736 | 8,589,934,592 | 32 GiB |
| Qwen 3.5 2B | 8,589,934,592 | 12,884,901,888 | 32 GiB |
| Qwen 3.5 4B | 12,884,901,888 | 21,474,836,480 | 64 GiB |

Each profile declares a fixed GPU admission budget and verifies that the offline bake GPU has that capacity. The native engine currently has no allocator that enforces a hard GPU-memory ceiling. The native weight reader also assembles its capped host buffer before copying tensors to CUDA; direct GPU consumption of incoming weight chunks remains unimplemented. These are engine integration limits, and the receipt records them explicitly.

The controller requests the model's listed host RAM, eight vCPUs, and a 30 GiB container disk. It allows one owned pod per controller and rejects hourly prices above $1.00. The serving deadline is 600 seconds. Startup waits for the engine's `/health` endpoint at a fixed 100 ms cadence; it does not run inference for readiness or automatic memory sizing.

## Local build and publication

Run the commands from an isolated checkout of the selected engine commit. The build script requires a fresh output directory, mounts the source read-only, and uses a digest-pinned CUDA 13.0 Ubuntu builder. It builds without allocating or exposing a local GPU. The receipt validates the builder's toolkit release against installed compiler packages and `nvcc`, and records the numeric compiler target separately from GPU capability. Suffixed targets require the separate architecture-aware packaging workflow. Python and compilers exist only in the build or local control environment.

```bash
cargo build --locked --release --features artifacts --bin switchyard
bash scripts/build-agent.sh
bash deploy/native/build-runtime.sh results/native-runpod/runtime-sm89 89
python deploy/native/prepare_artifacts.py \
  --config configs/runpod/native/qwen3.5-0.8b-sm89.json \
  --runtime results/native-runpod/runtime-sm89 \
  --model results/rust-engine/prepared-text-01 \
  --requests results/rust-engine/gauntlet-02/requests.json \
  --agent target/artifacts/switchyard-agent-x86_64-linux-musl \
  --tool target/release/switchyard \
  --output results/native-runpod/prepared-sm89
```

The model and requests arguments accept existing prepared native weights and representative request files. Add `--reuse-prepared PATH/prepared.json` to reuse the model archive from an earlier publication. Reuse requires the model name to match. A cache is carried forward only when the runtime ID, full runtime compatibility labels, and model revision still match.

The profile points to local files containing the RunPod and R2 credentials. The control scripts parse these files without sourcing shell code. Credentials stay in the local controller; pods receive only an agent token and expiring object URLs. Publication verifies archive hashes offline, checks any existing object before reuse, uploads through a multipart-aware local client, and sends the manifest last. This native controller owns one writer for each selected artifact prefix. Multipart upload supports the 4B model archive, which exceeds R2's [single-request upload limit](https://developers.cloudflare.com/r2/objects/upload-objects/). Signed tickets are uploaded directly from memory and deleted after launch cleanup.

## Offline cloud bake and fresh launch

The bake allocates one owned GPU pod, hydrates the runtime, model, and preparation artifacts, and executes the native bake tool. Representative generation includes an eight-row equal-length cohort and varied prompt lengths. This generation prepares the compiler cache using the declared request configuration. The tool saves the native binary identity, configuration, GPU, driver, CUDA cache inventory, and generation output. The controller publishes the cache and terminates the bake pod.

```bash
python deploy/native/run-bake.py \
  --config configs/runpod/native/qwen3.5-0.8b-sm89.json \
  --prepared results/native-runpod/prepared-sm89/prepared.json \
  --tool target/release/switchyard \
  --output results/native-runpod/bake-sm89
python deploy/native/run-launch.py \
  --config configs/runpod/native/qwen3.5-0.8b-sm89.json \
  --prepared results/native-runpod/bake-sm89/prepared-baked.json \
  --output results/native-runpod/launch-sm89
```

Every output directory must be fresh. The launch uses a new paid pod, empty ephemeral model/runtime storage, and the previously baked cache artifact. It requests one streamed completion after health becomes ready and records the first nonempty `delta.content`. The 2B and 4B profiles then measure one hot single call and two eight-call bursts with equal and varied prompt lengths. Each request is capped at 32 generated tokens. Workload receipts retain emitted token IDs, elapsed time, first-content distributions, and native first-token elapsed time. Native batch telemetry records the cohorts that actually formed through the external proxy; per-request queue wait is unavailable in the frozen engine. Provider logs and native JSON telemetry distinguish container startup, hydration, CUDA initialization, weight loading, listener readiness, and client receipt. Agent state records byte counts, hydration milestones, and cache residency. An image cache hit is reported as unknown unless the provider establishes it.

Both controllers retain the pod ID and exact allocated name before checking the hourly price, and terminate only that owned pod in `finally`. Cleanup requires a subsequent provider GET to return 404. The scripts use the RunPod [v2 pod API](https://docs.runpod.io/api-reference-v2/pods/create-a-pod) and retain controller events, sanitized container logs, native bake output, artifact publication receipts, completion frames, and cleanup confirmation.

## Corrected fleet measurements

The frozen source `600d6dd55f0d8afd771d8ba01331e4fdfdd2bda2` passed a real fresh RunPod launch for each listed model. All three used the same Ubuntu engine binary, SHA-256 `06a83dc1153adb4161b4a2b0b168925d5f493e7e7498ab4db155df7544f83c2a`, an RTX 4090, and driver 580.159.04. Each launch hydrated an empty ephemeral model directory and a prepared driver-cache seed. The bake and serving pods are confirmed absent.

| Model and receipt | Allocation to health ready | Allocation to first content | Request to first content |
| --- | ---: | ---: | ---: |
| [0.8B](receipts/native-runpod-qwen0.8b.json) | 35.20 s | 36.23 s | 1.026 s |
| [2B](receipts/native-runpod-qwen2b.json) | 59.71 s | 60.26 s | 0.555 s |
| [4B](receipts/native-runpod-qwen4b.json) | 172.17 s | 173.21 s | 1.037 s |

The 2B and 4B launches also completed the three hot workloads below. Every call emitted its full 32-token budget; the burst rows contain eight simultaneous client calls and 256 emitted tokens. Throughput divides actual emitted token IDs by the interval from the first client start to the last client finish. First-content latency includes the external RunPod proxy. With eight samples, the nearest-rank p95 is the maximum observed latency.

| Model | Workload | Aggregate tokens/s | First content p50 | First content p95 |
| --- | --- | ---: | ---: | ---: |
| 2B | Single | 43.7 | 0.396 s | 0.396 s |
| 2B | Equal-length burst | 163.8 | 0.840 s | 1.058 s |
| 2B | Varied-length burst | 82.7 | 1.508 s | 2.758 s |
| 4B | Single | 26.8 | 0.371 s | 0.371 s |
| 4B | Equal-length burst | 115.3 | 1.234 s | 1.235 s |
| 4B | Varied-length burst | 33.5 | 3.513 s | 6.790 s |

These measurements apply to the frozen engine, listed model revisions, fixed bounds, architecture, and driver. The frozen engine groups equal prompt lengths, and external proxy arrival times can split a burst into multiple native batches. The receipts retain the actual `batch_started` and `batch_finished` events. They do not establish results for the later mixed-length engine or the generic scheduler gauntlet.

The final scheduler fleet requires a new accepted binary and cache bakes. Its [shared runtime and driver-cache family design](native-cache-family-design.md) removes per-model configuration from the runtime artifact while preserving each exact driver-cache component.
