# Native CPU process snapshots

The vLLM adapter uses CRIU to preserve the native API process and its already-spawned engine worker immediately before `Worker.init_device`. The API process has constructed its `AsyncLLM`, configuration, renderer, input/output processors and native tokenizer. The worker has imported Torch, vLLM, the selected model class, tokenizer and CPU portions of its GPU runner. Restore resumes these existing processes and objects; it does not create another worker interpreter.

The [local timing receipt](receipts/native-cpu-tree-startup.json) records a complete agent rollout, including artifact hydration and the first streamed response. The [selected runtime configuration](../configs/serving-profiles/qwen3.5-0.8b-rtx5090.runtime.json) binds its runtime, CPU snapshot and compiled cache. This implementation belongs to `switchyard-weights`, the engine adapter. The generic simulator, scheduler and cloud-provider interface do not contain vLLM or CRIU behavior.

## Preparation boundary

Preparation runs the native vLLM serving CLI with the exact selected launch arguments, model metadata and serving policy. A general-plugin hook intercepts the worker's device initialization. The worker resolves its model class, prepares the native cached tokenizer and completes reusable CPU imports, then waits on an internal Unix control channel. The API process waits in the native engine startup handshake. CRIU checkpoints the entire process tree, including its native internal IPC.

The saved API renderer uses the native tokenizer object. The worker also retains its native tokenizer registry entry. Restore checks the API's `AsyncLLM`, configuration, renderer and tokenizer identities and independently checks the worker, configuration, tokenizer and model-class identities. These objects remain live in the restored process rather than being reconstructed from a separately preserved low-level tokenizer.

The model's GPU object graph is constructed after restore. GPU allocations, final tensor copies, fixed KV allocation and native GPU execution warmup remain launch work. The adapter retains the existing final tensor layout, bounded direct tensor transport, fixed GPU memory plan and baked kernel caches. CPU configuration and pre-spawn engine setup occur during preparation; they are not repeated at launch.

Preparation rejects Torch CUDA initialization and checks both captured processes for initialized CUDA or NVIDIA/DRI device descriptors. Torch 2.13's raw NVML inventory helpers initialize NVML without a matching shutdown. Preparation balances those inventory calls so they cannot retain GPU descriptors in the CPU image, then restores the original helpers before GPU continuation. Driver ELF mappings can exist without an initialized CUDA context; their mapped base-file identities remain part of snapshot compatibility.

The pinned libuv build creates `io_uring` mappings even with `UV_USE_IO_URING=0`. CRIU 4.2.1 rejected those mappings in the local bake. The native serving coroutine therefore runs on Python's selector event loop through `asyncio.run`; that loop survives restore. The impact of this event-loop choice on steady-state HTTP throughput has not been measured.

Unused Humming compiler helper processes and the Torch warm pool are disabled for this prepared eager profile. The checkpoint contains the selected worker rather than those background build processes. Compilation and cache preparation remain bake operations.

## Model metadata and configuration

CPU preparation must see the same metadata-only directory as the agent's bounded receiver. Exposing the original safetensors alongside metadata changes native vLLM configuration: its checkpoint inspection can disable tied embeddings when it finds an `lm_head`. A bake with those files and a launch without them constructed different tensor destinations. The helper copies exactly the selected manifest's metadata files, excluding safetensors, so native configuration matches the serving path. Metadata hashes are checked during baking; this adds no serving-time weight verification.

The selected proof uses language-only Qwen3.5 0.8B, BF16 final weights, eager execution, Triton attention, a 64-token context, one sequence and one TP/PP/DP worker. It retains 272 final tensors containing 1,504,786,720 bytes, a 1 GiB fixed KV plan and 1 GiB runtime headroom. Multi-worker snapshot preparation has not been implemented; the Rust validator requires exactly one prepared engine worker.

## Restore and process ownership

`serve.py` selects this adapter when `VLLM_CPU_SNAPSHOT_ROOT` identifies an installed snapshot component. It executes `switchyard-cpu-snapshot`, using `VLLM_CPU_SNAPSHOT_PROGRAM` when explicitly supplied. Writable CRIU work files go under `SWITCHYARD_CACHE_ROOT/cpu-restore`.

The Rust owner waits for the agent's ready file (`SWITCHYARD_READY_FILE`), restores the tree, and remaps its external Unix control socket. It sends fresh standard-stream descriptors, environment and native arguments in bounded frames. The API process checks the launch arguments, model content identity and serving policy against its prepared configuration, then forwards fresh streams and environment to the existing worker. Native internal engine channels remain those preserved by CRIU.

Preparation logs and transient standard streams are detached before checkpointing. Restored streams belong to the current agent; preparation configuration does not include agent credentials. Parent-death signals and owned process-group cleanup connect cancellation from the Rust restorer to the API process and its worker. Restore enters the current launch cgroup instead of the bake container's cgroup.

CRIU can preserve a time namespace whose clock excludes time spent as images. A bounded control exchange calibrates restored Python timestamps against the Rust supervisor's monotonic clock. Phase records contain the offset and uncertainty, while local operation durations remain unchanged.

## Compatibility and deployment limits

A snapshot requires matching runtime identity, absolute runtime and loader paths, adapter payload, CPU features, CPU vendor/model, kernel release, base OS, CRIU version and mapped base-file bytes. The native arguments, model identity and serving policy are also fixed by the CPU bake. The GPU serving profile separately enforces its selected device and memory plan. This CPU image is not a promise of portability across arbitrary CPU or GPU hosts.

Runtime content hashing occurs during preparation. Restore retains existing compatibility and mapped-base-file checks without rehashing the runtime or weights. No inference readiness request, automatic memory sizing or stall recovery is added. The benchmark observes HTTP availability and sends its first inference request only once the endpoint is available.

The demonstrated workflow uses privileged Docker with the host PID namespace and CRIU 4.2.1 in a matching Ubuntu image. The [RunPod deployment preflight](runpod-cpu-snapshots.md) tested a fresh Secure Cloud RTX 4090 pod on October 11, 2026. CRIU could not checkpoint even a `sleep` process because `CAP_CHECKPOINT_RESTORE` and `CAP_SYS_ADMIN` were absent from its effective and bounding capability sets. Native CPU baking and restore therefore remain untested on RunPod; matching GPU/CPU/CUDA does not grant those permissions. A reduced capability set has not been demonstrated. The local image also contains build tools; compiler-free base-image compatibility remains untested. CRIU's external Unix socket, inherited-FD, detached sibling restore and `--shell-job` mechanisms are described in the [CRIU command reference](https://github.com/checkpoint-restore/criu/blob/v4.2.1/Documentation/criu.txt).

## Local bake and rollout

Build the Rust wrapper for the serving image, and build the agent and loader identity tool from the same source. Supply a prepared serving fixture, runtime partition, native model/cache bake and loader configuration:

```bash
CPU_SNAPSHOT_PREPARED=/path/to/prepared-serving \
CPU_SNAPSHOT_PARTITION=/path/to/runtime-partition \
CPU_SNAPSHOT_BAKE=/path/to/native-model-cache-bake \
CPU_SNAPSHOT_LOADER_CONFIG=/path/to/loader.json \
CPU_SNAPSHOT_OUT=/path/to/fresh-cpu-tree \
  bash crates/switchyard-weights/benchmarks/run_cpu_snapshot.sh
```

The helper uses `switchyard-local-criu` by default; `prepare_cpu_tree.py --image` selects another matching image. Its output includes `serving.json`, a metadata-only model view, a writable cache clone, the packed CPU artifact, preparation receipt and CRIU images. The CLI's `prepare` subcommand requires `--serving-config`; the obsolete partial-import preparation path has been removed.

Bind that CPU artifact and the current `loader-identity` into a cache/runtime fixture with `prepare_serving_artifacts.py`, then run `run_prepared_cold.py`. All three helpers expose their required arguments with `--help`. Use the existing tmux observation pane; the helpers do not create sessions. Detailed logs remain under the selected output directory:

- `timeline.jsonl`: launch-relative milestones and native operation records.
- `startup-profile.jsonl`: imports and function spans, with per-process identities and clock calibration.
- `engine-startup.jsonl`: retained CPU state, device phases, bounded tensor copies and fixed-budget decisions.
- `hydration-profile.jsonl`: shared permits, decoder work, writes, finalization and publication.
- `thread-waits.jsonl` and `container-memory.jsonl`: host/cgroup samples.
- `receipt.json`: binary digest, final state, first-content timing and sampled memory peak.

## Startup without CRIU

`prepare_serving_artifacts.py --without-criu` prepares a serving fixture without a CPU snapshot. Omit `--cpu-artifact` and skip `prepare_cpu_tree.py` / `run_cpu_snapshot.sh` entirely. The native GPU bake still prepares final tensors, kernel caches and the fixed GPU memory plan. Assembly removes the CPU component and snapshot environment variables; `prepared.json` records `cpu_snapshot_enabled: false` and null CPU artifact fields. A CPU artifact and `--without-criu` cannot be supplied together.

An already-prepared snapshot fixture can also be tested without rebuilding its GPU cache:

```bash
python3 -u crates/switchyard-weights/benchmarks/run_prepared_cold.py \
  /path/to/prepared-serving /path/to/runtime-partition /path/to/legacy-ticket.json \
  --without-criu --output results/no-criu-rollout
```

The override writes a separate selected runtime JSON beside the output and leaves the input fixture unchanged. It excludes CPU-image transport, the restore binary mount, privileged mode and host PID sharing. A fixture assembled with `--without-criu` selects this path automatically, so the cold-start flag is unnecessary for that fixture. Ordinary startup defaults to the small pinned Ubuntu serving image; `--image` can override it. The [checked-in RTX 5090 configuration](../configs/serving-profiles/qwen3.5-0.8b-rtx5090.no-criu.runtime.json) pins the same runtime, model, prepared tensors and GPU cache as the snapshot configuration, with the CPU snapshot removed.

Python runs the normal prepared-import adapter and vLLM spawns its worker. Imports, CPU object construction and native worker startup execute at launch. Baked caches and the fixed serving budget remain active. No automatic sizing, inference readiness probes or recovery behavior is introduced. Receipts record `cpu_snapshot_enabled` and `cpu_restore_privileges` separately.

The [local no-CRIU receipt](receipts/no-criu-local-cold-start.json) records a fresh ordinary-permission container reaching first streamed content in 28.253 s. Runtime hydration took 7.933 s; engine spawn to first content took 16.131 s; the first request itself took 0.355 s. Native telemetry confirmed prepared tensor destinations, the fixed 1 GiB KV budget, skipped memory profiling and disabled automatic sizing. The no-CRIU assembly reproduced the same GPU cache artifact identity without a CPU artifact. This was one local HTTP sample with warm source caches, not a RunPod serving test or an isolated estimate of CRIU's net benefit.

## Local timing results

| Measurement | Parent-only snapshot | Native API/worker tree |
| --- | ---: | ---: |
| Launch → first streamed content | 26.997 s | 18.730 s |
| Engine process launch → first content | 14.716 s | 4.484 s |
| CPU restore itself | 0.143 s | 0.616 s |
| Restore complete → first content | 14.447 s | 3.712 s |
| Parent import interval union | 3.587 s | 0.032 s |
| Worker import interval union | 4.413 s | 0.015 s |
| New worker process starts after restore | 1 | 0 |
| Runtime artifact hydration | 8.177 s | 10.096 s |
| CPU artifact hydration | 1.617 s | 4.078 s |
| First request → streamed content | 0.347 s | 0.363 s |
| Sampled charged memory peak | 16.62 GB | 19.19 GB |

The API/worker tree has 431,245,534 compressed bytes and 2,181,431,021 bytes of CRIU images. Preparation finished at 13.037 s, dump took 0.958 s and publication finished at 30.334 s; these bake costs are excluded from launch timings. The retained tree restores more CPU state than the earlier image, increasing hydration and restore cost while reducing the post-restore interval by 10.735 s.

The other successful native-tree samples reached first content at 18.057, 18.570 and 18.659 s. The final 18.730 s sample follows already-imported native function aliases when profiling, so model construction and API initialization have measured spans rather than disappearing behind references captured before restore. These sequential samples do not isolate instrumentation overhead; their hydration and restore durations also varied.

### Complete launch timeline

| Interval after launch | Duration | Operation |
| --- | ---: | --- |
| 0.000 → 3.841 s | 3.841 s | Container launch |
| 3.841 → 3.979 s | 0.137 s | Agent admission and bootstrap setup |
| 3.979 → 14.246 s | 10.267 s | Runtime setup/hydration, overlapping CPU image, cache and weight reception |
| 14.246 → 14.402 s | 0.156 s | Launch engine adapter and prepare restore |
| 14.402 → 15.018 s | 0.616 s | CRIU restore and fresh control handshake |
| 15.018 → 15.050 s | 0.032 s | Worker stream/env binding and instrumentation |
| 15.050 → 15.358 s | 0.308 s | Device/runner initialization; CUDA initialization is a nested 0.051 s span |
| 15.358 → 16.588 s | 1.230 s | GPU model construction and final tensor loading |
| 16.588 → 16.597 s | 0.008 s | Fixed budget admission and KV setup; KV allocation itself takes 0.002 s |
| 16.597 → 17.586 s | 0.989 s | Native eager GPU execution warmup |
| 17.586 → 17.609 s | 0.023 s | Native engine handshake completion |
| 17.609 → 17.678 s | 0.069 s | HTTP application construction |
| 17.678 → 18.043 s | 0.365 s | API application-state initialization |
| 18.043 → 18.310 s | 0.267 s | Finish server startup and observe HTTP availability |
| 18.310 → 18.367 s | 0.057 s | Benchmark observes agent state and submits first request |
| 18.367 → 18.730 s | 0.363 s | First inference request to streamed `Paris` |

The artifact's internal runtime hydration interval is 10.096 s; the 10.267 s bootstrap interval also includes prerequisite setup. CPU-image and cache hydration last 4.078 s and 0.242 s respectively, overlapping runtime hydration. The final tensor transport moves 1,504,786,720 bytes in 364 chunks. Its summed staging counter is 0.482 s and summed GPU DMA counter is 0.058 s. Native model construction takes 0.173 s within the 1.230 s load operation; these nested and overlapping counters are not a partition of that operation.

These samples use a fresh installation, local HTTP delivery with warm source caches, a pre-existing Ubuntu image, NVMe-backed Btrfs with `zstd:3`, a 32 GiB memory cap, no swap and twenty shared hydration workers. They exclude R2, cloud allocation and image pulling. Native startup telemetry is enabled. Container launch and filesystem variation affect the totals; nested operation counters must not be added to elapsed phase durations.

Runtime hydration is now the largest elapsed phase. The latest sample confirms filesystem writeback backpressure: 115 agent-thread samples were waiting in `balance_dirty_pages`, and 28 were waiting in `btrfs_tree_read_lock_nested`. File writes accumulated 40.808 worker-seconds, file creation 4.419 and shared-permit waits 4.649. Runtime finalization took 0.159 s and atomic publication 0.062 ms. These worker sums overlap; they are not elapsed startup time. Sampled dirty pages reached 11.312 GB, with no CPU quota throttling or cgroup memory-limit/OOM events.

The local HTTP runtime transfer averaged 476.9 MB/s over its 9.568 s delivery window. This includes staggered requests and receiver pacing; it does not measure maximum network capacity or R2 performance. The network/decode/write stream can be paced by the confirmed filesystem waits. The concurrent CPU image is substantially larger, but these samples do not isolate its incremental contention or the cost of Btrfs compression. Detailed decoder, write and wait telemetry remains available for that diagnosis.

The former 1.964 s pre-spawn CPU engine interval is captured during preparation and no longer repeated at launch. Its historical call-level attribution remains unresolved; the result does not identify that old interval as a particular import or CUDA operation. The new continuation still has GPU-dependent runner/model construction, tensor placement and native warmup, plus API setup. Some allocator/runner work within model loading and finer HTTP startup/polling costs remain unseparated.

Residual imports total tens of milliseconds, mostly small HTTP integration and attention-backend modules. Cached imports still perform Python lookups. This result removes the multi-second import bottlenecks; it does not claim that every import statement or all CPU work has vanished.
