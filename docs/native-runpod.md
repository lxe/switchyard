# Native Rust RunPod deployment

The native RunPod path serves the Rust `switchyard-engine` binary from independently published runtime, model, and CUDA driver-cache artifacts. A small Ubuntu container downloads the static Rust agent. The agent hydrates all three artifacts concurrently, launches the native engine, and proxies its chat completion endpoint. The serving container requires no Python interpreter, Torch, vLLM, or CRIU.

The `*-sm89-final.json` profiles under `configs/runpod/native/` select the accepted engine source, a pinned Qwen 3.5 model revision, an RTX 4090 allocation, CUDA 13.0, and fixed request limits. Earlier profiles and receipts retain the frozen baseline. The measured deployments use BF16. Numerical accuracy follows the source commit's [accuracy receipt](native-engine-accuracy.md); cloud startup and throughput measurements do not establish additional numerical parity.

## Artifact contents and compatibility

Each artifact has its own immutable R2 prefix and manifest ID. Packing and hashing happen before a paid serving allocation. The fresh serving path uses trusted streaming and does not hash complete model or runtime files during startup.

| Artifact | Contents | Used during serving |
| --- | --- | --- |
| Runtime | Native engine, launch wrapper, CUDA and C++ libraries shared by all models | Yes |
| Model | Prepared native text weights, tokenizer, model configuration and chat template | Yes |
| GPU cache | Family of exact-driver CUDA compilation seeds and their bake receipts | Yes |
| Preparation | Native configuration, bake executable, representative requests, build inventory | No |

The runtime labels identify the source commit, engine SHA-256, Ubuntu version, CUDA build and compiler versions, CPU architecture, and GPU compute capability. A model's cache family adds its native configuration hash, revision, model archive identity, GPU name, and the Rust scheduler's configuration fingerprints. Each component retains its exact driver identity. The live native adapter passes the model's configuration through `--config-json`, which allows all three models to use the same runtime artifact.

The direct launch controller checks the published cache family manifest before allocation. With `--family PATH/family.json`, bootstrap observes the actual host driver and selects its matching signed ticket before downloading the agent or hydrating artifacts; an unsupported driver stops the allocation without transferring model weights. The shared launch wrapper independently checks the observed GPU, compute capability, and selected component marker before executing the engine. The generic scheduler hydrates the family as one cache artifact and relies on that wrapper to reject an unsupported driver. Earlier direct launches used `--cache-alternative` with separate artifacts.

The cache family is hydrated under `{cache_root}/gpu-kernels`; `drivers/<observed-driver>/cuda/` becomes `CUDA_CACHE_PATH`. Cache files remain a separate artifact so a new engine build or configuration requires a new bake while unchanged model weights can be reused. The driver cache stores compiled GPU code and metadata.

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

For a new publication, supply a verified Ubuntu runtime export whose `build.json` matches the profile's source commit. The accepted SM89 export used for these final artifacts is `results/native-gpu-builds/ubuntu13-final-01/targets/sm_89/runtime`. The following commands show preparation of one model; use fresh output paths for each publication.

```bash
cargo build --locked --release --features live --bin switchyard
python deploy/native/prepare-shared-runtime.py \
  --config configs/runpod/native/qwen3.5-0.8b-sm89-final.json \
  --runtime results/native-gpu-builds/ubuntu13-final-01/targets/sm_89/runtime \
  --tool target/release/switchyard \
  --output results/native-runpod/shared-sm89
python deploy/native/prepare_artifacts.py \
  --config configs/runpod/native/qwen3.5-0.8b-sm89-final.json \
  --runtime results/native-gpu-builds/ubuntu13-final-01/targets/sm_89/runtime \
  --model results/rust-engine/prepared-text-01 \
  --requests results/rust-engine/gauntlet-02/requests.json \
  --agent target/artifacts/switchyard-agent-x86_64-linux-musl \
  --tool target/release/switchyard \
  --shared-runtime results/native-runpod/shared-sm89/shared-runtime.json \
  --output results/native-runpod/prepared-sm89
```

The model and requests arguments accept existing prepared native weights and representative request files. Add `--reuse-prepared PATH/prepared.json` to reuse the model archive from an earlier publication. Reuse requires the model name to match. A cache is carried forward only when the runtime ID, full runtime compatibility labels, and model revision still match.

The profile points to local files containing the RunPod and R2 credentials. The control scripts parse these files without sourcing shell code. Credentials stay in the local controller; pods receive only an agent token and expiring object URLs. Publication verifies archive hashes offline, checks any existing object before reuse, uploads through a multipart-aware local client, and sends the manifest last. This native controller owns one writer for each selected artifact prefix. Multipart upload supports the 4B model archive, which exceeds R2's [single-request upload limit](https://developers.cloudflare.com/r2/objects/upload-objects/). Signed tickets are uploaded directly from memory and deleted after launch cleanup.

## Offline cloud bake and fresh launch

The bake allocates one owned GPU pod, hydrates the runtime, model, and preparation artifacts, and executes the native bake tool. Representative generation includes an eight-row equal-length cohort and varied prompt lengths. This generation prepares the compiler cache using the declared request configuration. The tool saves the native binary identity, configuration, GPU, driver, CUDA cache inventory, and generation output. The controller publishes the cache and terminates the bake pod.

```bash
python deploy/native/run-bake.py \
  --config configs/runpod/native/qwen3.5-0.8b-sm89-final.json \
  --prepared results/native-runpod/prepared-sm89/prepared.json \
  --tool target/release/switchyard \
  --output results/native-runpod/bake-sm89
python deploy/native/prepare-family.py \
  --fleet-config configs/runpod/native-fleet.json \
  --model qwen3.5-0.8b \
  --prepared results/native-runpod/bake-sm89/prepared-baked.json \
  --tool target/release/switchyard \
  --publish-config configs/runpod/native/qwen3.5-0.8b-sm89-final.json \
  --output results/native-runpod/family-sm89
python deploy/native/run-launch.py \
  --config configs/runpod/native/qwen3.5-0.8b-sm89-final.json \
  --prepared results/native-runpod/bake-sm89/prepared-baked.json \
  --family results/native-runpod/family-sm89/family.json \
  --live-model results/native-runpod/live-model-0.8b-03/live-model.json \
  --output results/native-runpod/launch-sm89
```

The family command derives labels from the selected fleet configuration, so that file must already name the shared runtime and model archive used by the bake. The live-model argument points to the separately published legacy archive reference described in the [family publication workflow](native-cache-family-design.md). Reusing the final checked-in fleet requires no additional build or publication.

Every output directory must be fresh. The launch uses a new paid pod, empty ephemeral model/runtime storage, and the previously baked cache artifact. It requests one streamed completion after health becomes ready and records the first nonempty `delta.content`. The 2B and 4B profiles then measure one hot single call and two eight-call bursts with equal and varied prompt lengths. Each request is capped at 32 generated tokens. Workload receipts retain emitted token IDs, elapsed time, first-content distributions, and native first-token elapsed time. Native batch telemetry records the cohorts that actually formed through the external proxy; per-request queue wait is unavailable in the frozen engine. Provider logs and native JSON telemetry distinguish container startup, hydration, CUDA initialization, weight loading, listener readiness, and client receipt. Agent state records byte counts, hydration milestones, and cache residency. An image cache hit is reported as unknown unless the provider establishes it.

Receipt field `agent_listener_allocation_s` records the controller's first successful agent HTTP observation through the RunPod proxy. It does not establish when the agent bound its socket, and proxy availability can overlap hydration. Native event `http_listening` measures the engine's listener on its own process clock. Client first-content measurements use the controller's monotonic clock; agent hydration milestones use the agent clock. These clocks describe different boundaries and should not be subtracted from one another without an explicit timestamp alignment.

Both controllers retain the pod ID and exact allocated name before checking the hourly price, and terminate only that owned pod in `finally`. Cleanup requires a subsequent provider GET to return 404. The scripts use the RunPod [v2 pod API](https://docs.runpod.io/api-reference-v2/pods/create-a-pod) and retain controller events, sanitized container logs, native bake output, artifact publication receipts, completion frames, and cleanup confirmation.

## Final fleet configuration

`configs/runpod/native-fleet.json` selects the accepted Ubuntu SM89 binary from source `1c966bba5ff60412ceb33b1fdb6745eb78943e57`, SHA-256 `410e210f203803846772ccc6a031bbbbb9ce1e1a0e420ea82f81b738d5163355`. All three pinned models use runtime artifact `a48e3d2af7cd81fc7a860d011418b44a99fe3f30e4595f2d1ab2d86e724c0a53`, containing 452,739,512 compressed bytes. Model configurations arrive as inline native arguments, and bake executables stay outside this serving artifact.

Each model has a separately published live archive reference and a cache family named `gpu-kernels`. The current families contain only driver 580.159.04 for the RTX 4090 at compute capability 8.9. Romania and both listed Iceland pools are eligible allocation locations, but location alone does not establish driver compatibility. An allocation with a different driver stops at the wrapper; additional driver coverage requires an independent bake against the same binary and configuration.

The fleet uses the public static agent from [release agent-v0.5.0-rc.3](https://github.com/lxe/switchyard/releases/tag/agent-v0.5.0-rc.3), pinned to SHA-256 `8c32c2bc05ca76485a8208af288fa63cbf22bcaec2b63f623b61b7196e8f2e4f`. Its configuration requests at least 64 GiB host RAM, eight vCPUs, and 30 GiB disk per GPU for the complete model set. It permits three instances and $2.80 per hour, with $1.00 per run and $3.00 per experiment. Startup, request, run, and experiment deadlines are 600, 120, 1,200, and 3,600 seconds respectively. Hydration concurrency is twenty, hydration retries are zero, and health polling remains 100 ms.

The capacity priors remain planning assumptions. Their startup values of 35.20, 59.71, and 172.17 seconds come from frozen baseline readiness observations; the single/aggregate throughput assumptions are 40/160, 40/160, and 25/100 tokens per second. These values are not final-binary gauntlet results. Input-work and KV priors also remain assumptions until matching scheduler measurements replace them.

Offline `live cache-bindings` validation confirms that every published family matches the final runtime, model archive, native engine configuration, and runtime launch fingerprint. The full acceptance workflow also requires a read-only cloud preflight and a 1,470-call workload gauntlet per model. Direct controller receipts establish fresh native serving and cleanup; the scheduler and elasticity gates require their own results.

The portable [final fleet publication receipt](receipts/native-runpod-final-fleet.json) records the complete runtime, agent, model archive, and cache references, the exact fleet-file digest, observed driver coverage, and a fresh provider 404 audit of all six final bake and serving pods.

## Final runtime cloud measurements

The accepted Ubuntu binary passed a fresh direct-controller launch for each pinned model using the shared runtime and its independently baked driver-cache family. All launches observed an RTX 4090, driver 580.159.04, and compute capability 8.9. Every model directory was uncached, and all three bake pods and all three serving pods are confirmed absent. These receipts measure BF16 serving; F32 parity is established by the separate local numerical gates.

| Model and receipt | Allocation to health ready | Allocation to first content | Request to first content |
| --- | ---: | ---: | ---: |
| [0.8B](receipts/native-runpod-final-qwen0.8b.json) | 22.97 s | 24.07 s | 1.102 s |
| [2B](receipts/native-runpod-final-qwen2b.json) | 60.85 s | 61.85 s | 0.997 s |
| [4B](receipts/native-runpod-final-qwen4b.json) | 111.55 s | 112.59 s | 1.047 s |

Hydration durations use the agent clock. The CUDA, weight-reader, and engine-listener timestamps below use elapsed time from native process start; the weight-read column is the interval between its start and finish events. They do not include allocation, container startup, or model transfer.

| Model | Agent hydration duration | CUDA initialized | Weight-read interval | Engine listening |
| --- | ---: | ---: | ---: | ---: |
| 0.8B | 14.26 s | 0.216 s | 1.032 s | 1.592 s |
| 2B | 48.57 s | 0.227 s | 2.856 s | 3.896 s |
| 4B | 94.95 s | 0.215 s | 6.011 s | 8.144 s |

The 2B and 4B launches each completed seventeen hot calls: one single call, eight equal-length calls, and eight varied-length calls. Every call emitted its full 32-token budget. Aggregate throughput divides the emitted token count by the interval from first client start to last client finish; first-content latency includes the external proxy. With eight samples, the nearest-rank p95 is the maximum observed latency.

| Model | Workload | Aggregate tokens/s | First content p50 | First content p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2B | Single | 36.4 | 0.385 s | 0.385 s |
| 2B | Equal-length burst | 159.7 | 0.942 s | 0.943 s |
| 2B | Varied-length burst | 110.6 | 1.199 s | 1.628 s |
| 4B | Single | 23.0 | 0.573 s | 0.573 s |
| 4B | Equal-length burst | 116.7 | 1.230 s | 1.231 s |
| 4B | Varied-length burst | 80.5 | 1.672 s | 2.208 s |

Both equal-length bursts formed native batches of one and seven rows as calls arrived through the proxy. The 2B varied burst formed one single row followed by a seven-row mixed batch. The 4B varied burst formed mixed batches of three and five rows. The receipts retain every observed batch shape and prompt length. These direct workloads establish mixed-length serving for the final cloud binary; the full scheduler gauntlet remains a separate measurement.

## Frozen baseline cloud measurements

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

The final fleet uses a new accepted binary and independent cache bakes. Its [shared runtime and driver-cache family format](native-cache-family-design.md) removes per-model configuration from the runtime artifact while preserving each exact driver-cache component.
