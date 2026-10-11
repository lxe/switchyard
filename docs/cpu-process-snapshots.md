# CPU process snapshots

The serving adapter uses CRIU to preserve a Python process after CPU preparation. The saved process contains imported Torch, Transformers, and tokenizers modules, a tokenizer object when supplied, and a vLLM import thread paused before `vllm.platforms` executes. CUDA initialization, worker construction, model weights, and serving sockets occur after restore.

## Preparation boundary

The pinned vLLM root import reaches platform selection through `env_override`, `torch_utils`, and a pin-memory availability check. The adapter places an import loader in front of platform and provider modules. The background vLLM import waits in that loader without executing platform code; its thread, Python import lock, and partial root module are preserved by CRIU. The root import is therefore incomplete at checkpoint time. CLI, GPU worker, provider, and kernel imports remain part of the resumed startup.

CUDA-enabled Torch loads cuBLAS Lt, which can load the driver ELF during import. A mapped driver library alone does not establish a CUDA context. The preparation checks require `torch.cuda.is_initialized()` to be false and reject NVIDIA or DRI device descriptors. They record mapped driver files and bind their exact bytes in the base-file compatibility inventory. The adapter blocks Torch's lazy CUDA initializer until the process resumes. It neither hides visible GPUs nor selects a CPU platform for a later GPU run.

The tokenizer object and its output are checked across restore to demonstrate retained CPU state. Native vLLM tokenizer construction remains native; this adapter does not replace its tokenizer factory with the retained low-level tokenizer.

## Restore and fresh configuration

`serve.py` selects the adapter when `VLLM_CPU_SNAPSHOT_ROOT` names an installed snapshot tree. It executes `bin/switchyard-cpu-snapshot` under `SWITCHYARD_RUNTIME_ROOT`; `VLLM_CPU_SNAPSHOT_PROGRAM` can name a different runtime-relative executable or an absolute path in an installed adapter component. Restore work files go under `SWITCHYARD_CACHE_ROOT/cpu-restore`. These variables belong to the engine adapter, without adding CRIU behavior to the generic agent or simulator.

The Rust wrapper consumes the agent's existing `switchyard-metadata-ready-v1` stdin gate before restore. CRIU remaps an external Unix control socket to a fresh socket. The wrapper then passes fresh stdin, stdout, and stderr descriptors and a bounded configuration frame containing the current native arguments and selected serving environment. The checkpoint contains scrubbed preparation configuration rather than agent credentials or a previous run's control streams.

The restored process checks retained object identities and its CPU boundary, applies fresh environment and Python module search paths, installs the current serving policy, and releases the vLLM import thread. It then runs the existing native startup adapter. A private stdin pipe supplies that adapter's normal metadata gate. CUDA initialization and weight loading proceed normally, including the adapter's existing fork or spawn decision. Telemetry reports CPU CRIU restore separately from worker process creation.

`SWITCHYARD_CPU_SNAPSHOT` records emitted by Rust use the supervising process's `CLOCK_MONOTONIC` time. CRIU can restore a time namespace whose clock excludes the interval spent as images, so the wrapper calibrates the resumed Python clock against the supervisor with a bounded control exchange. It passes an explicit offset and uncertainty to native telemetry; local operation durations remain unchanged. Checkpoint and restore records include their operation durations. The CPU probe exits after retained-state checks without resuming platform selection or using a GPU.

## Compatibility and privileges

A snapshot requires the exact runtime artifact ID, absolute runtime root, payload, CPU feature profile, kernel release, CRIU version, base OS identity, and mapped base-file bytes. It is not universally relocatable. Its compatibility metadata has no GPU architecture key, although the pinned Torch runtime and any mapped driver bytes still constrain compatibility. Full runtime hashing occurs during preparation; serving restore checks the runtime identity and mapped base files without rehashing the runtime tree.

Prepare inside the same base image and filesystem layout used for serving, normally at `/opt/switchyard/runtime`. An Arch-host snapshot must not be restored into the Ubuntu serving image. Build the Rust wrapper for the serving image's libc, and install the same CRIU executable and dependencies for preparation and restore. Immutable runtime and snapshot components must exist at the recorded paths; CRIU logs and PID files use a separate writable work directory.

The demonstrated host workflow runs the owner and CRIU as root. CRIU also supports `CAP_CHECKPOINT_RESTORE` under documented restrictions, with ptrace permissions and access to mapped files. Docker's default capability and seccomp settings are insufficient for this workflow. A local container proof should explicitly grant checkpoint/restore and ptrace capabilities and permit the necessary syscalls; namespace operations can additionally require `CAP_SYS_ADMIN`. The exact reduced capability set has not been demonstrated by the host proof. The adapter runs a normal CRIU feature check at bake time and uses strict CPU compatibility on dump and restore.

The implementation uses CRIU's documented external Unix socket and inherited-FD mechanism, `--shell-job` for a fresh supervising session, and detached sibling restore so the Rust owner can wait for its restored child. See the [CRIU 4.2.1 command reference](https://github.com/checkpoint-restore/criu/blob/v4.2.1/Documentation/criu.txt) and [CRIU checkpoint/restore overview](https://criu.org/Checkpoint/Restore).

## Local CPU proof

Build the wrapper with `cargo build -p switchyard-weights --bin switchyard-cpu-snapshot --release`. Run the proof script in the existing observation pane with a prepared runtime partition and a fresh output directory:

```bash
CPU_SNAPSHOT_PARTITION=/path/to/runtime-set \
CPU_SNAPSHOT_OUT=/path/to/fresh-cpu-proof \
CPU_SNAPSHOT_TOKENIZER=/path/to/tokenizer.json \
  bash crates/switchyard-weights/benchmarks/run_cpu_snapshot.sh
```

The output directory contains a regular packed `cpu_process` artifact, its source tree, a preparation receipt, and restore logs and receipt. Production preparation uses the same `prepare` subcommand. Production restore omits `--probe` and passes native vLLM arguments after `--`.

## Recorded host proof

The [CPU snapshot receipt](cpu-process-snapshot-acceptance.json) records an actual CRIU 4.2.1 round trip on the Arch host. CPU preparation took 1.40 seconds, dump took 214 ms, and restore plus the fresh control handshake took 151 ms. The packed artifact contains 97,498,142 compressed bytes from 447,458,221 bytes of CRIU images. All four module identities, the tokenizer identity, and the sentinel identity were preserved; tokenizer output matched. CUDA remained uninitialized with no GPU device descriptors. These measurements cover CPU restore only, not the native CUDA continuation or a container rollout.
