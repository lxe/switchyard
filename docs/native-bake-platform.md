# Native bake platform controller

The native bake platform controller allocates one RunPod GPU directly, observes its hardware, and runs fixed native cache-preparation commands. It reads an independent platform configuration and a list of model configurations and prepared artifacts. It does not use inference scheduling to select or operate the bake pod.

## Platform configuration

`configs/runpod/native/bake-platform-sm89-32g.json` permits the 0.8B and 2B models with a 32 GB host-RAM minimum and a 12 GiB GPU-admission floor. `bake-platform-sm89-64g.json` permits all three models with a 64 GB RAM minimum and a 20 GiB GPU-admission floor. Both request an RTX 4090, compute capability 8.9, eight vCPUs, 30 GB disk, and a minimum compatible host CUDA capability of 13.0. Their price ceiling is $1.00 per hour, their owned-pod limit is one, and their pod deadline is 600 seconds.

The provider's minimum host CUDA capability is an allocation requirement. The serving runtime retains its own toolkit, compiler, binary, and source identities. A newer compatible host driver does not change those artifact identities; its cache component records the actual observed driver separately.

`driver_target` is an exact observed driver version, or `null` for discovery. `excluded_drivers` prevents selecting unwanted versions before weights transfer. These fields select a bake host; they do not establish cache validity. A new engine or model configuration still requires new components for every desired driver. The 32 GB discovery profile permits compatible hosts reporting CUDA 13.0 or newer. The 64 GB profiles now request the reported host CUDA version 13.0 and target 580.173.02, observed on a Romania RTX 4090 during the [driver discovery attempt](receipts/native-bake-driver-discovery-01.json). The observed driver still determines whether generation can begin.

`bake-platform-sm89-64g-driver159.json` requests the same fixed 64 GB host and 20 GiB GPU floor for all three models while targeting 580.159.04. It admits the listed Romania and Iceland pools. Its driver must still be observed before any weights transfer; pool names do not establish driver identity. Use the original 64 GB profile for a separate 580.173.02 bake. Both drivers need fresh F32 components for the new binary.

`bake-platform-sm89-64g-discovery.json` admits the same three pools and fixed resources with a minimum reported CUDA capability of 13.0 and no exact driver target. Its completed bake observed 580.126.20 on an Iceland RTX 4090. `bake-platform-sm89-64g-driver595.json` targets the separately observed 595.91.07 host with reported CUDA 13.2. Discovery permits an offline bake to record a new driver; serving still requires a published component for the exact observed driver.

Model configurations supply the model revision, native request bounds, and binary provenance. Before allocation, the controller checks that the platform admits every requested model and meets its fixed RAM, CPU, disk, and GPU-admission requirements. Every input must refer to the same shared runtime. After generation, the controller compares the bake receipt's complete native configuration, exact binary hash, and hardware identity before packing a component.

## Offline control and cleanup

The HTTP bake control surface authenticates every request with a temporary token. It exposes status, a bounded diagnostic log, and bundles for known model names. Its only write operation releases the fixed bake program with an empty `POST /start`; callers cannot submit commands. The controller observes and records hardware before releasing that gate, so unsupported or unwanted hardware causes no weights transfer.

Python and zstd are installed only inside the temporary offline bake container. Its token and signed download script stay in memory and are removed from the environment before native subprocesses run. These dependencies are not published into the serving runtime or serving image. Native generation uses the existing Rust bake executable and engine binary.

The bake controller polls the authenticated HTTP surface, retains sanitized provider logs, and downloads each cache bundle with a 32 MiB compressed-byte limit. The diagnostic tail is capped at 64 KiB and redacts URLs and the control token. It downloads the finished components, then terminates the exact journaled pod before local packing and object-store publication. Each component is verified and published independently after cleanup succeeds. A provider GET returning 404 confirms cleanup. Transient 404/502/503/504 start responses receive at most ten retries of the same empty operation; sanitized response diagnostics are retained, and authentication or request errors stop immediately. Generation failure, control failure, and timeout follow the same cleanup path. Publication failure occurs after GPU ownership has ended. The 600-second deadline bounds the remote-work loop; bounded diagnostic reads and provider cleanup calls run afterward.

## Controller invocation

Run this example after the engine source and prepared artifacts are frozen. Replace the configuration and preparation paths together when using a newer engine freeze; the paths below refer to the recorded binary from source `1c966bb`. Every output directory must be fresh.

```bash
python deploy/native/run-model-bakes.py \
  --platform-config configs/runpod/native/bake-platform-sm89-64g.json \
  --config configs/runpod/native/qwen3.5-0.8b-sm89-final.json \
  --prepared results/native-runpod/prepared-0.8b-sm89-final-01/prepared.json \
  --config configs/runpod/native/qwen3.5-2b-sm89-final.json \
  --prepared results/native-runpod/prepared-2b-sm89-final-01/prepared.json \
  --config configs/runpod/native/qwen3.5-4b-sm89-final.json \
  --prepared results/native-runpod/prepared-4b-sm89-final-01/prepared.json \
  --tool target/release/switchyard \
  --output results/native-runpod/bake-models-sm89-driver-target
```

Artifact assembly happens after baking. `prepare-family.py` verifies the finished components and derives serving compatibility labels with the offline Rust `cache-bindings` command. This operation performs no cloud allocation, inference scheduling, or generation. Existing family components retain their original bytes and identities when a new component is added.

## F32 prefix-cache candidate

`configs/runpod/native-fleet-f32-prefix-candidate.json` supplies separate F32 serving inputs for all three pinned models. Each native model uses 16-token pages, 32-token checkpoints, a 256 MiB retention budget and 16 entries, with context 128, batch size eight, queue capacity 64 and a 2 ms batch wait. The three matching `qwen3.5-*-sm89-f32-prefix-candidate.json` preparation profiles request a 64 GB host and a fixed 20 GiB GPU-admission floor on the RTX 4090. These limits are fixed admission requirements; cloud peak memory remains to be measured.

The candidate now names the independently inventoried and published Ubuntu SM89 runtime `05f47556f16e7983ae5b056dd3834e9540e87170a1fc850d40a5eb41dc7d0a82` and three new cache families covering 580.126.20 and 580.159.04. Its shared wrapper requires a validated exact-driver seed before executing the engine. Its engine freeze is source `5e9850e`, and its preparation profiles name build source `aa67c22`, which includes the matrix follow-up. The new `switchyard-engine-kernels` crate embeds its paged-decode PTX in that binary. The local compute-120 binary and existing BF16 runtime cannot supply its SM89 identity.

The already published native weights and live model archives retain their pinned revisions and bytes. New preparation artifacts must bind those weights to the new runtime and complete F32 cache configuration. Independently bake all three models for each desired driver, then assemble new families using the candidate's native bounds. Populate the fleet with its published shared runtime before deriving family labels, and add only verified new family references afterward. The [candidate receipt](receipts/native-f32-prefix-candidate.json) records these pending identities and fixed limits.

Local RTX 5090 gates establish F32 teacher/reference agreement and complete cached/uncached sequence equality. The [580.126.20 cloud bake](receipts/native-f32-bake-driver126-01.json) and [580.159.04 cloud bake](receipts/native-f32-bake-driver159-01.json) each matched all 48 rows and 1,536 output token IDs across the three models, using the identical native artifacts, configuration and requests. Each model reused 960 input-token positions within the fixed retention budget. Both owned pods were confirmed absent before component publication. Required 580.173.02 coverage, cloud peak memory and fresh serving measurements remain pending; the [capacity receipt](receipts/native-f32-capacity-rejections-01.json) retains rejected requests that created no pod. The copied scheduling priors remain planning assumptions until matching F32 measurements replace them. Existing uncached BF16 deployment files retain their original artifact references; cache-enabled BF16 quality acceptance remains unestablished.

## Cross-GPU generation comparison

`f32-prefix-bake-requests.json` supplies eight prompts with a shared 64-token prefix and lengths 64, 80, 88 and 96, each with 32 output tokens. Preparation appends eight copies of the first prompt, producing the exact 16-row bake input within context 128 and batch size eight. These rows exercise varied lengths, shared prefixes and repeated cache hits.

`prepare-generation-reference.py` restores the already published native model archives locally, verifies their complete trees with the Rust artifact tool, and runs the selected reference-tested F32 engine against the expanded bake requests. It changes only the physical model-directory argument. The retained references contain the fixed native configuration, pinned model artifact, expected cloud build source, local binary and hardware identities, complete requests, and every generated token ID. All three RTX 5090 executions produced 16 complete rows and 512 output token IDs per model; the [candidate receipt](receipts/native-f32-prefix-candidate.json) links their portable references.

Cache-enabled F32 controller invocations require one `--generation-reference` argument per paired model input, using the paths recorded in that receipt. The controller validates these references before allocation. After downloading each cloud bake, it compares the bake configuration and requests, then requires exact equality of every output token ID, row count and finish reason. It saves `generation-parity.json` beside `generated.json` before publication. A difference retains the full expected and actual sequences and prevents publishing the component until the cause is diagnosed. Latency is not compared across GPUs. This offline hardware comparison remains separate from teacher-forced mathematical gates and serving readiness.

The [Ubuntu paged-engine matrix](native-gpu-builds.md#paged-engine-matrix) supplies reviewed SM89 and SM120a runtime inventories.

## Separate 32 GB host profiles

The `qwen3.5-*-sm89-f32-prefix-32g.json` inputs request 32 GB host RAM while preserving each model's complete F32 engine configuration, 20 GiB GPU-admission floor, eight vCPUs, request bounds and price ceiling. The original 64 GB profiles and fleet input retain their admission requirements. The aggregate native weight-buffer caps remain 1.5 GiB, 8 GiB and 12 GiB for 0.8B, 2B and 4B. These caps govern safetensors file buffers; other process allocations and file cache require separate measurements.

The [0.8B fresh serving receipt](receipts/native-f32-runpod-qwen0.8b-01.json) records an Iceland allocation using the 32 GB profile and driver 580.159.04. Allocation-to-first-content was 28.35 seconds. All 544 requested hot-workload tokens completed, with observed HTTP throughput of 23.01 tokens/sec for one call, 164.18 for the equal-length burst and 136.11 for the varied-length burst. These figures include the RunPod proxy and client request boundary. The native read finished with 1,504,832,320 bytes of aggregate source buffers; total cloud host memory was not sampled in this launch.

Two fresh 2B attempts encountered uncovered drivers: [595.91.07](receipts/native-f32-serving-driver-rejection-01.json) and [580.173.02](receipts/native-f32-serving-driver-rejection-02.json). Both failed before engine startup and their pods were confirmed deleted. The separate `*-32g-cuda13.json` inputs constrain the reported host CUDA version to 13.0 for direct controller experiments. This metadata filter still requires an observed exact driver and its validated cache component. The two `bake-platform-sm89-32g-f32-driver*.json` profiles permit only 0.8B and 2B offline bakes, retaining the 20 GiB GPU floor. Neither driver has a new published F32 component yet.

The [local 4B host-memory experiment](receipts/native-f32-4b-local-host-memory-01.json) ran the frozen Ubuntu compute-120a engine under a 32 GiB cgroup limit with swap disabled. Cgroup usage peaked at 15,555,022,848 bytes, child peak RSS was 9,023,557,632 bytes, and all 512 output token IDs matched. It loaded 8,411,565,456 bytes of source buffers within the unchanged 12 GiB cap, with no OOM events. This execution used existing files and pagecache; fresh cloud hydration and 4B serving at 32 GB remain pending.

The separately named 4B 32 GB profile enables `measure_host_memory`. Its direct launch controller starts the fixed Bash sampler in `host-memory.sh`, which records cgroup current, peak and limit bytes plus available swap and OOM counters once per second, for at most 600 samples. Missing counters are recorded as null, and a mount-root fallback is identified separately from the process cgroup. These measurements appear in the portable launch receipt alongside the observed source-buffer bytes. The sampler changes no runtime artifact, cache identity, engine configuration or serving dependency.

## L40 fleet candidate

`configs/runpod/native-fleet-f32-l40-candidate.json` binds the same three native models to newly validated NVIDIA L40 cache families. Its matching `qwen3.5-*-l40-f32-prefix-candidate.json` profiles request 64 GB host RAM, eight vCPUs, 30 GB disk and a 20 GiB GPU-admission floor in EUR-IS-2 or US-KS-2. Engine configuration, request limits, retention limits and the $1/hour direct-run ceiling remain fixed. The runtime artifact and native model archives retain their existing bytes and IDs. Changing `NATIVE_ENGINE_EXPECTED_GPU` to `NVIDIA L40` produces a new runtime-launch fingerprint, and each family records the actual L40 hardware and driver.

The [L40 bake receipt](receipts/native-f32-l40-bake-01.json) records an allocation in US-KS-2 at $0.82/hour, reporting compute capability 8.9, driver 580.159.04 and 46,068 MiB GPU memory. All 48 rows and 1,536 output token IDs matched the retained local F32 executions, including prompt counts and finish reasons. Each model reused 960 input-token positions. The controller confirmed pod deletion before publishing the new components and assembling the three L40-bound families. The rebuilt Rust CLI independently derives the same fingerprints, and the [candidate receipt](receipts/native-f32-l40-candidate.json) contains their complete artifact references and compatibility labels.

The first [fresh serving attempt](receipts/native-f32-l40-serving-driver-rejection-01.json) reported 580.178.04 in US-KS-2. Its exact-driver guard failed before engine startup, and cleanup was confirmed. Two [targeted bake requests](receipts/native-f32-l40-capacity-rejection-01.json) for that driver returned no capacity and created no pod. The published families therefore admit only 580.159.04. Fresh L40 serving and cgroup-memory acceptance remain incomplete for all three models; copied scheduling priors remain planning assumptions. Both owned L40 pods are confirmed absent in the [cleanup audit](receipts/native-f32-l40-owned-cleanup-audit-01.json).

## Verification scope

The CPU HTTP fixtures establish authentication, hardware checks before program execution, exact driver-target rejection, fixed-operation gating, known-bundle access, and retained sanitized stderr on failure. Three lifecycle fixtures verify that confirmed cleanup and receipt copying precede publication, and that generation failure or unconfirmed cleanup prevents publication. Five generation fixtures verify complete-ID equality, retained mismatch diagnostics, row-count rejection, configuration/request rejection and complete reference outputs. Two HTTP gate fixtures also establish bounded idempotent start retries and sanitized response retention. Run all 17 offline controller fixtures with `python -m unittest discover -s deploy/native/tests -p 'test_*.py'`; these also include the three existing driver-family wrapper fixtures.

The discovery attempt established driver 580.173.02 and confirmed owned-pod cleanup. It published no additional cache component. The HTTP request headers and output-parent creation have been repaired, and diagnostics now persist before cleanup. A successful cloud bake remains required to establish component coverage. The F32 preparations bind the reviewed Ubuntu runtime. Exact-driver bakes, cross-GPU comparisons and fresh cloud serving receipts remain pending.
