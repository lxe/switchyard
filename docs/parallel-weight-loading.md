# Parallel weight hydration

Switchyard's raw extent path downloads independent byte intervals into private, pre-sized regular files. A serving consumer can load a safetensors tensor after every interval covering that tensor has finished writing. TAR is unnecessary for this path: it is inherited packaging in `shards-v1`, whose sequential TAR reader prevents later pieces of one large file from being consumed independently.

The implementation is split between `switchyard-artifacts::extents`, which owns generic format validation, HTTP and filesystem operations, and `switchyard-weights`, which owns safetensors parsing and the local vLLM completion bridge. Agent and live fleet model sources select either this extent representation or the existing legacy representation. Runtime and compiler-cache manifests retain artifact-v1.

The engine-independent `model_source.kind = "files"` path publishes complete raw files before ordinary engine startup. Its one-object-per-file HTTP range packaging and current serving measurements are documented in [Raw model file hydration](raw-model-hydration.md).

## Extent manifest

`switchyard-extents-v1` contains `format`, `id`, `content_id`, caller-defined `metadata`, filesystem `entries` and sorted `extents`. Entries reuse artifact-v1's canonical relative-path and permission rules. This initial layout accepts regular files and explicit directories; it rejects links and special files.

Each extent has these fields:

| Field | Meaning |
| --- | --- |
| `entry` | Index of the destination regular file in `entries` |
| `offset` | First destination byte to write |
| `length` | Exact number of bytes to write |
| `object` | Provider-neutral key resolved by the ticket's URL map |
| `object_offset` | First requested byte in the raw source object |
| `object_size` | Exact full size of that source object |
| `sha256` | Digest of this extent's bytes, checked in verified mode |

Extents are sorted by `(entry, offset)` and must cover each nonempty file contiguously, without overlap, holes or arithmetic overflow. Empty files have no extents. Repeated references to an object must declare the same object size. Object keys and destination paths obey canonical relative-path rules. Total file bytes and worker limits are checked before files are created.

`pack_extents` splits exact source-file bytes into raw content-addressed objects. A full-object request requires status 200 and the exact declared body length. A request for an interval within a larger raw object requires status 206, exact `Content-Range: bytes START-END/SIZE` and the exact interval length. Ignored ranges, content encoding, oversized responses and truncated bodies fail. Redirects are disabled, ordinary HTTPS certificate validation remains active, URLs never enter identities or errors, and each attempt has a complete request/body deadline. There are no retries or recovery mechanisms in this path.

The layout `id` covers format, content identity and extent descriptors. `content_id` hashes the separate `switchyard-files-v1` domain, metadata and exact entries, including full-file hashes. The same files and metadata have the same content identity across extent sizes. These hashes are computed during preparation and verified startup. Caller-trusted startup validates structure without recomputing them. Neither identity equals an existing `shards-v1` archive identity or an artifact-v1 identity.

## Completion and publication

`ExtentHydration::prepare` creates a private mode-0700 staging directory and pre-sizes mode-0600 regular files. `run` schedules extents round-robin across files and streams each HTTP body through a fixed-size copy buffer into Unix positional writes. There is no ordered archive reader, compressed cache or archive staging. A shared optional semaphore bounds transfers across concurrent calls and remains owned by a blocking writer until that writer exits.

An extent becomes complete only after exact EOF, any requested digest check and all writes. `wait_range` and `wait_any` register notifications before testing completion, so completion and cancellation cannot be missed between checking and waiting. A `CompletedRange` retains staging ownership and permits bounded reads of only that interval. Pre-sizing a file or creating a mapping does not make its unfilled bytes readable by the loader; the completion contract is mandatory.

Only the asynchronous hydration owner publishes the directory after all writers stop. File descriptors can be opened under the publication lock and continue naming the same inode after rename. Dropping the running future cancels readers and writers; cleanup waits for the final blocking worker or lease. No detached worker can publish. The generic API refuses an existing destination by default. An explicit trusted reuse option accepts a completed publication only when its private receipt matches the canonical content identity; verified mode does not use that receipt as verification. A completed publication may remain when a later consumer fails; partial files are never published, and consumer failure never authorizes inference readiness.

Verified mode checks manifest/content identities, hashes every extent and each final file, and fsyncs final file contents. Existing artifact-v1's verified API remains unchanged. Trusted mode performs no content hashes, archive/cache staging, fsync or final filesystem scan. It retains manifest topology, path/type/mode checks, exact pre-sized file coverage, HTTP boundaries and cancellation/publication ownership. The caller must own destination ancestors exclusively; these checks are not a defense against another process replacing paths concurrently.

Transport copy buffers are bounded by `buffer_bytes * min(concurrency, extent_count)`, plus one HTTP body part and TLS/HTTP state per active worker. The model's RAM-backed files consume the full model size and are not part of that staging-buffer claim. Tensor metadata and the configured header limit also consume bounded CPU memory. The Python adapter's optional pinned buffer is separately bounded; its temporary device tensor can reach the largest tensor size.

## Safetensors and vLLM loading

Safetensors offsets are relative to the data section following the eight-byte header length and JSON header. `WeightStream` first waits for that prefix, then the complete bounded header, validates tensor shapes/dtypes/offset coverage, and yields only completed tensor intervals. Header discovery is incremental: a tensor from one checkpoint file can load while another file's header is pending. Round-robin transport starts initial extents from different files before later extents of one large file. The multi-file test holds the second file's initial extent and proves the first file's tensor is returned before publication.

The minimal Python adapter registers a custom vLLM loader and overrides its iterator hook. It inherits the pinned default loader's model-specific assignment and loaded-weight checks, leaving GPU context, model allocation, transformations and inference initialization with vLLM. The Linux Rust bridge serves one local consumer through a mode-0600 Unix socket. Fixed-size commands and bounded response lines keep protocol buffering independent of checkpoint size. It retains backing descriptors and exposes `/proc/PID/fd/N` paths until the consumer acknowledges that all assignments and GPU reads have finished.

A CPU mmap view is ordinary host memory. The default adapter yields those completed views to vLLM's synchronous assignment path. Optional CUDA staging copies chunks into a bounded pinned host buffer, uses a dedicated CUDA stream, waits on events before reusing the buffer, and yields a temporary device tensor only after its copy is complete. The stream waits for the allocation stream, the temporary allocation records the copy stream, and the final acknowledgment follows model-specific assignment and CUDA synchronization. This permits network writes for other extents during CPU-to-GPU copies. It does not implement GPUDirect, cuFile or direct HTTPS-to-VRAM.

The pinned implementations were inspected in runtime export `8670fb233b5ad18dc48b504492943d36a45af4657e3527ba7a1f8c349aa2b5dc`: Python 3.12.14, Torch 2.13+cu129, vLLM 0.31+cu129 and safetensors 0.8.0. Its lazy loader opens whole local files with `safe_open`; fastsafetensors, runai and instanttensor are distinct loaders rather than a guarantee of incomplete-file safety. The completion bridge uses vLLM's actual custom-loader registration and default iterator extension points. The [official loader registry documentation](https://docs.vllm.ai/en/stable/api/vllm/model_executor/model_loader/) and [safetensors metadata documentation](https://huggingface.co/docs/safetensors/metadata_parsing) describe those interfaces.

The current adapter deliberately requires one engine worker, one unprefixed dense and unquantized checkpoint, supported byte-aligned scalar dtypes and ordinary tensors. Distributed workers, expert filtering, encoder-only filtering, packed dtypes and tensor subclasses fail or remain unsupported. The bridge is Linux-specific because its descriptor handoff uses `/proc`. The Qwen 0.8B comparison below validates native Qwen assignment for the tested checkpoint and runtime.

## Local validation and benchmark

The GPU experiment used an RTX 5090 after checking process ownership. No compute workload was present; desktop GPU processes continued running. The fixture was a deterministic four-layer BF16 Llama checkpoint with hidden size 1024, vocabulary size 256 and 126,901,364 total source bytes. Both layouts used the same bytes, twenty allowed transfers, RAM-backed destination directories and the same loopback HTTP server delaying each 64 KiB write by 4 ms. Two 8 MiB pairs measured an eighteen-extent layout. The final 4 MiB pair measured thirty-three extents spanning two transfer waves; the final adapter initializes its CUDA stream and pinned buffer before connecting. The current legacy production hydrator was invoked unchanged through `legacy_hydrate`; the new bridge and Rust helper used debug builds.

Actual vLLM engine workers registered the adapters, allocated the model, loaded parameters and generated four deterministic tokens. A private benchmark plugin records parameter hashes and timings without changing the immutable runtime. Transfers begin when the loader reaches its weight iterator in both cases (`--wait-for-consumer`), so import and allocation do not consume the entire synthetic transfer before overlap can be observed. Ordinary bridge startup does not defer transfer. Results were:

| Measurement | 8 MiB pair 1 legacy / extents | 8 MiB pair 2 legacy / extents | 4 MiB pair legacy / extents |
| --- | ---: | ---: | ---: |
| Hydration, seconds | 1.399 / 0.630 | 1.284 / 0.655 | 0.580 / 0.564 |
| Weight phase, seconds | 1.423 / 0.639 | 1.306 / 1.065 | 0.604 / 0.573 |
| Engine ready, seconds | 9.240 / 7.720 | 8.434 / 10.203 | 7.681 / 7.586 |
| GPU copies finished before last HTTP transfer | 0 / 34 of 39 | 0 / 4 of 39 | 0 / 32 of 39 |

Every final source-file hash matched in both paths. Both GPU parameter hashes were `2a16b0a32249f5d5cc6b7838d96ac88961480e398070d2c02a3e523caa6d01a4`, and both engines returned token IDs `[119, 233, 233, 233]`. The eighteen-extent layout had a 1,179,648-byte transport copy-buffer bound. The final thirty-three-extent layout had a 1,310,720-byte bound and a separate 1 MiB pinned GPU staging buffer. A different fixture with thirty-two extents proves twenty useful transfers by holding the first range while nineteen later ranges finish. The cancellation tests also prove shared permit release, no incomplete publication and cleanup after blocked writers, and an executable bridge test rejects an invalid fixed-size command while HTTP headers are held.

These runs establish byte equivalence, safe early GPU copies and actual model/inference equivalence. The weight phase was shorter in all three pairs, but total startup was noisy and the second extent engine took longer. An additional single-wave run passed file, parameter and inference equivalence without any early GPU copy: staging/context setup can consume that layout's small overlap window. Overlap is conditional on the consumer being ready before remaining transfer work finishes. They do not establish a stable whole-startup speedup, WAN/provider throughput, large Qwen performance or the cause of the existing 2B stall. Qwen and network-provider performance require separate end-to-end measurements.

The benchmark writes raw receipts and logs to its requested output directory. A compact [benchmark receipt](parallel-weight-loading-benchmark.json) accompanies this document, and the benchmark program and legacy helper reproduce the comparison. Unit/integration tests cover invalid paths, offsets, ranges, lengths, digests, duplicate headers, tensor shape/coverage, zero-sized tensors, multi-file discovery, completion/publication races, cancellation and protocol failure. Workspace tests, formatting, strict Clippy and documentation generation validate the integration. Build the benchmark helpers with `cargo build -p switchyard-weights --release` and `cargo build -p switchyard-agent --release --example legacy_hydrate`.

## Agent model sources

An endpoint ticket has one model source. A legacy source retains the existing top-level `manifest` and `urls` fields. An extent source instead has `model_source.kind = "extents"`, an `artifact` containing the raw extent manifest and temporary object URLs, and a `loader` selecting the runtime Python program and bounded pinned staging buffer. The live fleet configuration uses `model_artifact.kind = "extents"` with the published prefix, layout ID, canonical-content ID and loader configuration. Mixed legacy and extent references are rejected.

The agent prepares a private loader bundle and socket outside the immutable runtime. It downloads runtime archives and model extents under the same transfer semaphore. Complete model metadata is copied into that private directory, allowing the vLLM process to import and initialize after metadata is ready. Tensor delivery uses the completion bridge while remaining payload extents arrive. The engine process retains the private metadata and plugin directory for its lifetime, and the ordinary health, tokenization and inference checks determine endpoint readiness.

Runtime artifact-v1 reception uses physically reserved compressed backing and independent extraction permits. A combined preflight accounts for outputs sharing a filesystem and for model RAM, RAM-backed runtime or spool storage, decoder histories, transfer buffers and pinned staging under host and cgroup memory limits. Raw model extents write into model file offsets and do not use the compressed spool. Trusted startup performs no whole-file hashes, final verification scans or file fsyncs.

Extent serving caches identify canonical model content, loader code and settings, normalized engine arguments, runtime identity and launch configuration, and the selected GPU. Repacking identical files can change the layout ID while retaining the content ID. A legacy archive cache cannot be relabeled as an extent cache. Existing legacy cache requirements retain their archive identity.

## Header-first weight preparation

`switchyard-weights pack` parses and validates each safetensors header during preparation, then writes its length field and JSON header into one independent raw object. Bulk objects preserve the remaining canonical file bytes. Metadata files and checkpoint header extents are scheduled before bulk payload extents; the generic transport receives only prefix boundaries and priority indices.

The preparation command writes `manifest.json` and an identity-named copy. Publication validates local objects, uploads them to an immutable prefix, and publishes the captured manifest last:

```sh
cargo run -p switchyard-weights --release -- pack SOURCE OBJECTS --extent-bytes 16777216
cargo run --features artifacts -- artifacts publish-extents \
  --config STORAGE_CONFIG --env-file CREDENTIAL_FILE --directory OBJECTS \
  --prefix switchyard/weights/qwen3.5-0.8b/header-first-v1 --receipt PUBLICATION_RECEIPT
```

The actual Qwen 0.8B manifest and compact benchmark receipt are saved in `experiments/qwen-0.8b-header-first.manifest.json` and `experiments/qwen-0.8b-header-first.json`. The canonical weight file is 1,746,942,600 bytes; payload granularity is 16 MiB. Both the baseline and extent path loaded identical GPU parameters through vLLM's native Qwen assignment and generated token IDs `[220, 16, 15, 15]`.

In one controlled, ordered pair on an RTX 5090, the weight phase including hydration took 17.15 seconds through the legacy path and 6.38 seconds through extents. Of 488 GPU copies, 481 completed before the last HTTP response. The reported constructor times, 88.16 and 81.24 seconds, exclude the initial vLLM import and include benchmark-only GPU-to-CPU parameter hashing. They are correctness measurements with isolated compiler caches, not production request-to-first-token timings.

## Transport observations

Optional generic transport observers report object-relative boundaries, received and written bytes, HTTP body waits, storage-write waits, and typed reqwest, Hyper and I/O causes. They retain no URLs, vendor response bodies or formatted error chains. The agent stores bounded observations and captures host state before publishing the first actual transport failure. Intentional cancellation does not create a failure snapshot or replace an earlier failure.

GPU completion reports distinguish host staging time from CUDA-event DMA time and locate the first and last completed copies relative to bridge readiness. Assignment completion is acknowledged only after native model assignment and CUDA synchronization. These spans can overlap reception and should not be added to elapsed startup time.

## Cold endpoint startup

The [endpoint startup receipt](../experiments/qwen-0.8b-agent-startup.json) records actual agent launches with the pinned artifact-v1 runtime and canonical Qwen 0.8B weights. The request clock begins before creation of a private Docker container and includes ticket fetch, preflights, runtime reception and extraction, tokenizer initialization, native vLLM startup, readiness checks and the first non-whitespace streamed response content. Model parameter hashing belongs to the separate correctness comparison and is absent from this timing.

One ordered cold pair returned “Paris” after 95.47 seconds through legacy hydration and 94.98 seconds through extents. This small whole-startup difference does not establish a stable speedup. Model hydration completed before GPU copies began on the fast loopback transport. The endpoint nevertheless exercised native Qwen assignment, the new GPU completion report and all existing admission checks.

A fresh extent container seeded with the prior local compiler directory returned “Paris” after 39.76 seconds. Native model loading fell from 5.93 to 1.59 seconds, and post-load initialization fell from 54.51 to 7.93 seconds. The startup-plan fingerprint remained `bb30bfe2df83e0c1`, but its memory estimate was rejected because available GPU memory was 26.61 GiB rather than the recorded 26.66 GiB. Native full memory profiling ran. The experiment used `cp --archive`, preserving compiler file timestamps; it does not prove an artifact-v1 cache export, extraction or R2 round trip.

These receipts use the recorded agent checksum and loader generation. Later lifecycle leases and hot-publication capacity checks changed the implementation and loader identity, so the receipts do not establish cache acceptance for a newly published agent. The remaining startup cost requires native phase profiling and fresh-process validation of each optimization.

## Loader directory ownership

The agent derives the loader directory path from the complete model-content and loader identities. Stable paths let compatible engine compilation and startup-plan fingerprints survive a clean restart. A separate, permanent lock file coordinates ownership without changing the model or runtime artifacts.

The agent takes an exclusive kernel lease to create or reclaim the directory, then shares the lease with the native engine. Each Python worker opens an independent shared lease before importing Torch. A bounded ownership record identifies the creator process, boot, PID namespace, and engine process group. The record is published atomically before directory creation; the engine group is registered before the metadata gate opens.

A subsequent launch reclaims an abandoned directory only after obtaining an exclusive lease and establishing that the recorded owner cannot still use it. Unknown records, foreign PID namespaces, reused creator PIDs during construction, and live process groups prevent reclamation. Cleanup preserves a stable lock inode, because unlinking a locked file would allow two different inodes to represent the same apparent lease. An asynchronous engine termination may leave private loader files for the next safe reclamation.

## Native startup profiling

The endpoint benchmark's `--profile-startup` option records native Python imports and vLLM initialization operations in each API and worker process. It loads an owned `sitecustomize.py` through the runtime's `PYTHONPATH`; it leaves the immutable runtime and every original operation in place. The JSONL records contain process/thread IDs, monotonic timestamps, operation names, nesting depth, and exception types. Arguments and exception messages are omitted.

The harness records its request epoch before container creation and captures `/proc/uptime` immediately before executing the unchanged agent binary. That rounded timestamp and the agent's first ticket request bound the Rust startup epoch. Kernel process creation is recorded separately because GPU container hooks execute before the agent. Python spans and HTTP observations use the host's Linux monotonic clock, allowing imports, worker launch, CUDA initialization, model loading, memory profiling, cache allocation, warmup, and API construction to be placed on the request timeline. Nested spans overlap and must not be summed.

The `--env-override-source` option stages a caller-selected `vllm.env_override` source before the request epoch for a private patch experiment. Both pristine and patched controls use the same staged pathname and import mechanism. Their receipts record the original manifest hash, staged source hash and diagnostic scope; the overlay does not create a published runtime identity. `--verify-concurrency 24 --concurrency 8` runs a bounded correctness burst after the first useful token and records completion counts and request latencies separately from startup.

A [matched lazy-import experiment](../deploy/hydration/patches/vllm-lazy-inductor-fallback.md) reduced standalone `import vllm` time but moved most of that cost into configuration loading. Its profiled and unprofiled endpoint pairs did not establish a stable first-token gain. The candidate remains experimental, with its native semantic checks and numeric results preserved for review.

The `--cli-main-source` option supports a matched private CLI adapter experiment through the same staging and source-hash mechanism. A [serve-only runner comparison](vllm-serve-only-startup.md) preserved native argument parsing and serving dispatch while skipping other command registration. The dependencies attributed to the benchmark import moved into the serve import, and the reversed unprofiled pair showed no first-token benefit. The canonical runner and its cache identity remain unchanged.

Use the same agent, model, runtime, engine arguments, and copied compiler-cache directory for comparisons. `--runtime-env KEY=VALUE` adds an explicit native setting to the benchmark ticket and the receipt. Profiled runs identify costs; fresh runs without instrumentation establish request-to-token performance. A copied local cache directory does not prove a published artifact-v1 cache roundtrip, and the native fingerprint and free-memory guards remain active.
