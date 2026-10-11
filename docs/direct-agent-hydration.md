# Agent hydration and serving startup

The selected Qwen3.5 0.8B endpoint produced its first streamed content, `Paris`, **26.997 seconds after local container launch**. Runtime hydration completed in **8.177 seconds** from that artifact's start. The main repair makes the compiler/toolchain artifact a bake dependency rather than a launch dependency for this prepared eager profile. Fixed GPU memory allocation, final tensor layout, native caches and CPU process restore remain enabled.

The [measurement receipt](receipts/direct-agent-hydration.json) preserves the controlled cases, shard counters, kernel wait samples, memory/pressure counters, disk deltas and final native timeline. The checked-in [selected runtime configuration](../configs/serving-profiles/qwen3.5-0.8b-rtx5090.runtime.json) includes the exact runtime, snapshot and cache identities. It is a local privileged-CRIU fixture; these results do not establish cloud-provider restore permissions or cross-host snapshot portability.

## Direct streaming configuration

Runtime, component and cache shards pass directly from HTTP responses through bounded readers into independent zstd/tar decoders. Each worker writes disjoint files in private staging. Finalization creates symlinks, applies modes and writes the trusted receipt, then publishes the directory with an atomic rename. No whole-artifact compressed reception spool, content hashing or final content scan is used on this trusted path. Structural checks, cancellation ownership and shared permits remain enforced.

The selected ticket has **twenty shared permits**. A permit remains owned through the decoder's completion, including cancellation. Twenty is an active-worker limit; the runtime has 67 shards. The final run observed twenty simultaneous decoders across all artifacts, not twenty per artifact running independently. Inference weights retain their separate bounded tensor transport: 64 MiB RAM, 20 MiB pinned staging and 4 GiB NVMe overflow.

The adopted direct-stream source commit is `6864a43ab0ed275317594d8a3ad8a3ec374c8bef` (local cherry-pick `6b79139`). Its first integrated agent binary was:

```text
7ad88bb6aad0e3fe36838e11bb55bb6e48fc91ca9ec1c85203573192c67f583b
```

The rebuilt instrumented static agent used by the final serving rollout was:

```text
dd88bea58e99af3d59646965ef9006e3e8576a54b2272790f983c22ea32df200
```

Each receipt records the binary SHA-256 and shared concurrency. The static build is reproduced by `scripts/build-agent.sh` with its pinned Rust image and locked dependencies.

## Writeback diagnosis

Runtime-only hydration through the actual agent took **7.531 seconds**. Adding the build, CPU snapshot and compiled-cache artifacts increased runtime hydration to **26.556 seconds**, even with model downloads and serving disabled. This isolation reproduces the large delay without CUDA, tensor loading or inference startup.

The runtime writes 12,076,369,101 bytes of file content; the build artifact adds 7,284,857,593 bytes. During the concurrent case, charged dirty pages plateaued near **12.05 GB**, cgroup I/O pressure accumulated **14.83 seconds** of “some” stalls and **14.33 seconds** of “full” stalls, and active extraction threads repeatedly slept in **`balance_dirty_pages`**. CPU throttling counters remained zero. Linux regulates dirty pages using both global and cgroup writeback domains; Btrfs supports this accounting. See the [kernel writeback documentation](https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#writeback).

Per-shard timers locate most of the additional delay inside file writes. These are sums across workers, not elapsed phase durations:

| Runtime measurement | Runtime alone | With build, snapshot and cache |
| --- | ---: | ---: |
| Hydration elapsed | 7.531 s | 26.556 s |
| File-write worker-seconds | 22.699 s | 228.472 s |
| Source-read worker-seconds | 13.975 s | 8.500 s |
| Decoder-read time minus source-read time | 13.078 s | 11.959 s |
| Thread CPU worker-seconds | 21.607 s | 19.448 s |
| Shared-permit wait worker-seconds | <0.001 s | 73.680 s |
| Progress-callback worker-seconds | 0.053 s | 0.047 s |
| Filesystem finalization elapsed | 0.153 s | 0.167 s |

Decoder-read time includes blocking source reads. Subtracting source-read time estimates decoder work plus scheduling inside those reads; it is not an isolated decompression CPU measurement. Kernel wait-channel samples identify the blocked operation but do not measure each sleep's exact duration. The many idle `futex_do_wait` samples are not evidence that progress locking caused the delay.

The host NVMe recorded 5.84 GB of writes and 2.21 busy seconds during the 26.06-second sampled window. Those counters include other host I/O, and compressed filesystem writes differ from logical file bytes. They do not establish hardware saturation. The demonstrated cause is dirty-page writeback throttling during concurrent artifact materialization. The relative roles of Btrfs compression, writeback feedback and background filesystem work within that throttle remain unseparated; no filesystem or global VM setting was changed.

The earlier standalone direct-tool sample took 3.045 seconds; the actual-agent runtime-only sample took 7.531 seconds. The remaining 4.486-second gap has not been separated into HTTP-fixture differences, writeback history and agent orchestration. The isolation results establish the concurrent-artifact escalation, rather than attributing that entire standalone-to-agent gap to decompression.

## Selected repair and comparable results

The eager profile serves native binaries and final tensors already prepared during baking. Its successful serving rollout does not need the build artifact's compilers, headers or development assets. The preparation helper now omits that component and removes its compiler/cubin environment paths from the serving configuration. Baking still uses the build artifact. The matching local CRIU image contains tools used to build CRIU itself; a compiler-free base image was not tested. This dependency change is specific to the selected prepared eager profile; it does not remove build dependencies from other runtime configurations.

| Complete serving case | Serving process spawned | Endpoint available | First content | Restore-to-first-content |
| --- | ---: | ---: | ---: | ---: |
| Previous compressed-spool rollout | 34.617 s | 49.570 s | 49.954 s | 15.031 s |
| Direct streaming, build retained | 32.773 s | 47.178 s | 47.631 s | 14.572 s |
| Direct streaming, build omitted, original environment | 11.110 s | 25.398 s | 25.785 s | 14.358 s |
| Final serving configuration and profiler | 12.280 s | 26.567 s | 26.997 s | 14.447 s |

Times in this table start at local container launch. Separate final artifact durations were **8.177 s runtime**, **1.617 s CPU snapshot** and **0.232 s compiled cache**; they overlap and must not be added. The selected repair reduces runtime write-pressure rather than changing decode work. Final runtime file writes accumulated 30.663 worker-seconds and permit waits 1.811 worker-seconds. Peak charged container memory was **16,620,949,504 bytes**, versus **22,040,424,448 bytes** in the previous spool rollout. All reported maxima come from sampled counters and can occur at different times.

These are single local samples with warm HTTP source caches, fresh output directories, a pre-existing Ubuntu image, NVMe-backed Btrfs `zstd:3`, a 32 GiB container cap and swap disabled. The host has 24 logical CPUs. Containers ran serially through tmux session `0`; builds and tests ran outside the measured rollout windows. Source caches were not flushed. Network measurements exclude R2 and cloud allocation; container launch variation is included in launch-to-token totals.

## Engine timeline after CPU restore

The final CPU restore took **142.9 ms** and completed at **12.550 seconds**. First content followed **14.447 seconds** later. The following elapsed intervals partition that period; nested operation timings are listed in the explanation rather than added again.

| Interval | Duration | Evidence |
| --- | ---: | --- |
| Restore complete → async engine construction | 3.794 s | Parent import union 3.334 s; model-config construction 0.457 s, including tokenizer construction |
| Async engine construction → process-start call | 1.964 s | Within native engine-client setup; detailed attribution remains unresolved |
| Process-start call → child profiler start | 0.022 s | `BaseProcess.start` itself took 1.134 ms |
| Child profiler start → device initialization begins | 4.139 s | Worker import union within this interval 4.115 s |
| Device initialization | 0.289 s | Native `Worker.init_device`; includes device/distributed setup |
| Model loading | 2.050 s | Model construction 0.609 s; direct staging 0.709 s and CUDA DMA 0.067 s are nested/overlapping counters |
| Fixed-budget admission and KV setup | 0.008 s | Native KV allocation 2.274 ms; fixed 1 GiB plan accepted, profiling forward skipped |
| Native execution warmup | 0.946 s | Eager mode, no Torch compile or CUDA graph capture; native warmup/autotuning still executes |
| Engine reply and API startup → HTTP available | 0.805 s | App construction 0.219 s and app-state initialization 0.401 s, with nested imports |
| HTTP availability → client request observed | 0.082 s | Existing availability polling and host observation |
| First client request → content | 0.347 s | No preceding tokenization or inference request |

The API process's complete observed import union was **3.587 s**, and the spawned worker's was **4.413 s**. These unions merge nested import spans within each process. The CPU checkpoint preserves parent imports, but the worker still uses spawn and reimports Torch/vLLM. The native tokenizer registry also constructs its own tokenizer; the checkpoint's preserved low-level tokenizer object is not reused by that registry.

The **1.964-second pre-spawn interval remains unexplained at call level**. The process-start syscall path is fast, so it must not be called a two-second OS spawn. The `torch.cuda._lazy_init` probe recorded no span; this does not establish that all CUDA initialization was absent or attribute that interval to CUDA. Similarly, `Worker.init_device` is a combined native operation, not a pure CUDA-context measurement. Remaining small gaps include parser/setup work, scheduling, RPC completion and health observation. No inference readiness probes, automatic sizing or stall recovery were introduced.

## Instrumentation controls

With the same final agent binary, runtime-only hydration took **7.185 s** with detailed shard profiling disabled and **7.715 s** with it enabled: an observed **0.530 s / 7.4%** difference. Cgroup CPU usage differed by **0.669 CPU-seconds**. Detailed profiling also enables the privileged read-only wait sampler, which runs outside the consumer cgroup.

For complete serving, Python import/operation profiling disabled gave **26.460 s** launch-to-content and **13.481 s** restore-to-content. Enabling it gave **26.997 s** and **14.447 s**, respectively: **0.536 s total** and **0.966 s after restore**. These single-sample controls quantify observed changes, not a precise profiler-only cost; hydration and container-launch variation also differ. The writeback conclusion does not depend on those sub-second differences.

## Reproduction and telemetry

[Prepare the fixture](../crates/switchyard-weights/benchmarks/prepare_serving_artifacts.py) to obtain `prepared.json`, `runtime.json` and `engine-args.json`. The preparation receipt binds local paths to immutable model, cache and CPU snapshot artifacts. The checked-in selected runtime provides the serving dependency configuration; compatible snapshot preparation still requires the documented [CPU restore environment](cpu-process-snapshots.md).

Run the [prepared fixture](../crates/switchyard-weights/benchmarks/run_prepared_cold.py) in existing tmux session `0`:

```sh
python crates/switchyard-weights/benchmarks/run_prepared_cold.py \
  /path/to/prepared /path/to/runtime-set /path/to/legacy-ticket.json \
  --case runtime --output results/runtime-only
```

Use `--case artifacts` to include selected components/cache without weights or serving; use `--case serving` for the complete first-token rollout. The runtime-only diagnostic uses the same agent runtime preflight, hydration, progress and publication code, but schedules no model transfers and launches no inference process. Controls are `--without-hydration-profile` and `--without-native-profile`. Native profiling remains enabled for the default serving case. The optional wait sampler uses passwordless `sudo` to read kernel wait channels of the owned container; other telemetry does not require it.

Each output's `extents/` directory retains `hydration-profile.jsonl`, `container-memory.jsonl`, `thread-waits.jsonl`, `startup-profile.jsonl` when enabled, `timeline.jsonl`, `agent.log` and `receipt.json`. Console output shows phase summaries; per-import, per-chunk and CUDA-copy events stay in files. [The analyzer](../crates/switchyard-weights/benchmarks/analyze_agent_profile.py) reports artifact elapsed time, peak decoder concurrency, permit/write worker-seconds, pressure and disk deltas:

```sh
python crates/switchyard-weights/benchmarks/analyze_agent_profile.py \
  results/runtime-only/extents --output results/runtime-only/analysis.json
```

The measured final rollout is retained locally under `results/engine-startup/direct-cold-final-02/`. Eleven focused streaming tests passed, including twenty simultaneous piped decoders, structural rejection, shared limits and cancellation ownership. The rebuilt static agent completed the actual serving rollout. Every owned rollout container and sampler was stopped; tmux session `0` remains available.
