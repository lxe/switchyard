# Native Rust RunPod deployment

The native RunPod path serves the Rust `switchyard-engine` binary from independently published runtime, model, and CUDA driver-cache artifacts. A small Ubuntu container downloads the static Rust agent. The agent hydrates all three artifacts concurrently, launches the native engine, and proxies its chat completion endpoint. The serving container requires no Python interpreter, Torch, vLLM, or CRIU.

`configs/runpod/native/qwen3.5-0.8b-sm89.json` selects the corrected engine source, the model revision, an RTX 4090 allocation, CUDA 13.0, and fixed request limits. The measured deployment uses BF16. Its numerical accuracy follows the source commit's [accuracy receipt](native-engine-accuracy.md); the cloud startup measurement does not establish additional numerical parity.

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

The profile bounds context length at 128 tokens, aggregate native weight bytes at 1,610,612,736, batch size at eight, and queued requests at 64. The CUDA cache is capped at 512 MiB. Artifact transfers share twenty download slots; the Rust model receiver passes 64 KiB pieces through a four-piece bounded channel. Runtime and cache hydration retain their own manifest and transfer accounting.

The profile declares an initial GPU admission budget of 8 GiB and verifies that the offline bake GPU has that capacity. The native engine currently has no allocator that enforces a hard GPU-memory ceiling. The native weight reader also assembles its capped host buffer before copying tensors to CUDA; direct GPU consumption of incoming weight chunks remains unimplemented. These are engine integration limits, and the receipt records them explicitly.

The controller requests at least 32 GiB host RAM, eight vCPUs, and a 30 GiB container disk. It allows one owned pod per controller and rejects hourly prices above $1.00. The serving deadline is 600 seconds. Startup waits for the engine's `/health` endpoint at a fixed 100 ms cadence; it does not run inference for readiness or automatic memory sizing.

## Local build and publication

Run the commands from an isolated checkout of the selected engine commit. The build script mounts the source read-only and uses a digest-pinned CUDA 13.0 Ubuntu builder. It builds without allocating or exposing a local GPU. Python and compilers exist only in the build or local control environment.

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

Both controllers retain the pod ID and exact allocated name and terminate only that owned pod in `finally`. Cleanup requires a subsequent provider GET to return 404. The scripts use the RunPod [v2 pod API](https://docs.runpod.io/api-reference-v2/pods/create-a-pod) and retain controller events, sanitized container logs, native bake output, artifact publication receipts, completion frames, and cleanup confirmation. The committed [cloud receipt](receipts/native-runpod-qwen0.8b.json) records the measured corrected build and fresh launch.

The runtime and model inputs are modular: a new prepared model and native configuration can use the same workflow. The receipt proves only the listed 0.8B model, architecture, driver, and request limits; 2B, 4B, and other GPU targets require their own bake and fresh launch evidence.
