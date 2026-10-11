# switchyard-weights

`switchyard-weights` is a serving-layer component that parses completed safetensors headers and yields tensors as their byte ranges finish hydrating. The generic raw extent format and HTTP/filesystem writes live in `switchyard-artifacts::extents`. This crate contains no cloud adapter and does not alter simulator or scheduler behavior.

`WeightStream` discovers headers incrementally across checkpoint files. Its `next_ready` method returns a tensor descriptor and a completion lease only when every covering extent has reached exact EOF and all its writes have finished. A header cannot expose unfilled tensor data. `WeightPlan` also provides complete-header discovery for consumers that prefer it. The parser rejects duplicate tensor names and fields, overlapping or missing data, shape/byte mismatches, arithmetic overflow, unsupported dtypes, invalid metadata, and trailing data. It supports the listed byte-aligned scalar dtypes, including BF16; packed sub-byte tensors and tensor subclasses are unsupported.

The experimental Linux executable packs local trees or serves one local consumer:

```sh
cargo run -p switchyard-weights -- pack SOURCE OBJECTS --extent-bytes 134217728
cargo run -p switchyard-weights -- serve TICKET DESTINATION SOCKET --trusted --concurrency 20
```

A ticket is JSON containing `manifest` and `urls`, with one HTTP(S) URL per object name. URLs are not part of identity and are never printed by the hydrator. Packing performs preparation-time hashes but does not upload anything. See [the format and integration contract](../../docs/parallel-weight-loading.md) for raw HTTP range descriptors, readiness, identities, and agent integration.

The socket protocol has fixed five-byte `next\n` requests. Each response is a bounded JSON line containing a tensor descriptor, a completed backing-file descriptor path and its readiness time. `{"done":true}` means all checkpoint tensors have been yielded. The consumer sends the fixed seven-byte `finish\n` acknowledgment only after assignments and GPU reads have completed. The server retains backing file descriptors through that acknowledgment. Linux `/proc/PID/fd/N` paths name the same inode across the final directory rename; the adapter does not race a disappearing staging path. The socket is created exclusively with mode 0600 in a caller-owned directory. An invalid command, disconnect, deadline, transfer error or cancellation aborts the bridge. Publication belongs solely to the hydration future, and partial staging cleanup waits for all writers and leases.

`python/switchyard_extent_loader.py` registers `switchyard_extents` through vLLM's general-plugin and custom-loader extension points. Its small Python iterator maps only tensor intervals whose completion Rust has announced, then delegates parameter assignment, transposition, sharding and loaded-weight tracking to the pinned `DefaultModelLoader`. GPU/model allocation and inference readiness remain owned by vLLM. The default path yields CPU views; optional `cuda_stage` uses a bounded pinned buffer and a dedicated CUDA stream, waits for events before buffer reuse, and yields each device tensor only after its copy completes. The temporary device tensor can be as large as the largest checkpoint tensor. This is ordinary CPU-to-GPU DMA, without GPUDirect or HTTPS-to-VRAM.

The current bridge is a single-consumer experiment for one unprefixed dense checkpoint. It has been exercised through vLLM 0.31, Torch 2.13+cu129, Python 3.12.14 and safetensors 0.8.0 on an RTX 5090. It has not been validated for distributed workers, MoE filtering, quantized tensor subclasses or the pinned Qwen checkpoints. The live agent currently accepts legacy model tickets; the extent integration contract is documented separately.

For an end-to-end benchmark, first inspect GPU/process ownership and ensure the caller can reserve a small local GPU allocation. Then build the unchanged legacy benchmark helper and the new bridge:

```sh
cargo build -p switchyard-weights --bin switchyard-weights --example legacy_hydrate
PYTHON benchmarks/run_local.py /dev/shm/UNIQUE-OUTPUT --hidden-size 1024 --piece-bytes 4194304
```

`PYTHON` must be a caller-selected environment containing the pinned serving packages. The script creates known synthetic Llama weights, local plugin discovery metadata, throttled loopback HTTP objects, a legacy TAR ticket and an extent ticket. It loads both through actual vLLM, compares final file hashes, every GPU parameter's combined hash and deterministic output tokens, and asserts that a GPU copy finished before the final HTTP transfer. Receipts separate hydration, weight loading, engine readiness, inference and copy timing. `--wait-for-consumer` is used to begin both transfers at the loader boundary for this controlled comparison; normal bridge startup begins transferring immediately. Artifacts stay under the output directory and processes are owned and cleaned by the benchmark.
