# Parallel weight hydration

Switchyard's raw extent path downloads independent byte intervals into private, pre-sized regular files. A serving consumer can load a safetensors tensor after every interval covering that tensor has finished writing. TAR is unnecessary for this path: it is inherited packaging in `shards-v1`, whose sequential TAR reader prevents later pieces of one large file from being consumed independently.

The implementation is split between `switchyard-artifacts::extents`, which owns generic format validation, HTTP and filesystem operations, and `switchyard-weights`, which owns safetensors parsing and the local vLLM completion bridge. Simulator, scheduler, provider adapters, fleet pins and existing artifact-v1 manifests are unchanged.

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

Only the asynchronous hydration owner publishes the directory after all writers stop. File descriptors can be opened under the publication lock and continue naming the same inode after rename. Dropping the running future cancels readers and writers; cleanup waits for the final blocking worker or lease. No detached worker can publish. An existing destination is refused, preserving exclusive publication rather than introducing an implicit reuse contract. A completed publication may remain when a later consumer fails; partial files are never published, and consumer failure never authorizes inference readiness.

Verified mode checks manifest/content identities, hashes every extent and each final file, and fsyncs final file contents. Existing artifact-v1's verified API remains unchanged. Trusted mode performs no content hashes, archive/cache staging, fsync or final filesystem scan. It retains manifest topology, path/type/mode checks, exact pre-sized file coverage, HTTP boundaries and cancellation/publication ownership. The caller must own destination ancestors exclusively; these checks are not a defense against another process replacing paths concurrently.

Transport copy buffers are bounded by `buffer_bytes * min(concurrency, extent_count)`, plus one HTTP body part and TLS/HTTP state per active worker. The model's RAM-backed files consume the full model size and are not part of that staging-buffer claim. Tensor metadata and the configured header limit also consume bounded CPU memory. The Python adapter's optional pinned buffer is separately bounded; its temporary device tensor can reach the largest tensor size.

## Safetensors and vLLM loading

Safetensors offsets are relative to the data section following the eight-byte header length and JSON header. `WeightStream` first waits for that prefix, then the complete bounded header, validates tensor shapes/dtypes/offset coverage, and yields only completed tensor intervals. Header discovery is incremental: a tensor from one checkpoint file can load while another file's header is pending. Round-robin transport starts initial extents from different files before later extents of one large file. The multi-file test holds the second file's initial extent and proves the first file's tensor is returned before publication.

The minimal Python adapter registers a custom vLLM loader and overrides its iterator hook. It inherits the pinned default loader's model-specific assignment and loaded-weight checks, leaving GPU context, model allocation, transformations and inference initialization with vLLM. The Linux Rust bridge serves one local consumer through a mode-0600 Unix socket. Fixed-size commands and bounded response lines keep protocol buffering independent of checkpoint size. It retains backing descriptors and exposes `/proc/PID/fd/N` paths until the consumer acknowledges that all assignments and GPU reads have finished.

A CPU mmap view is ordinary host memory. The default adapter yields those completed views to vLLM's synchronous assignment path. Optional CUDA staging copies chunks into a bounded pinned host buffer, uses a dedicated CUDA stream, waits on events before reusing the buffer, and yields a temporary device tensor only after its copy is complete. The stream waits for the allocation stream, the temporary allocation records the copy stream, and the final acknowledgment follows model-specific assignment and CUDA synchronization. This permits network writes for other extents during CPU-to-GPU copies. It does not implement GPUDirect, cuFile or direct HTTPS-to-VRAM.

The pinned implementations were inspected in runtime export `8670fb233b5ad18dc48b504492943d36a45af4657e3527ba7a1f8c349aa2b5dc`: Python 3.12.14, Torch 2.13+cu129, vLLM 0.31+cu129 and safetensors 0.8.0. Its lazy loader opens whole local files with `safe_open`; fastsafetensors, runai and instanttensor are distinct loaders rather than a guarantee of incomplete-file safety. The completion bridge uses vLLM's actual custom-loader registration and default iterator extension points. The [official loader registry documentation](https://docs.vllm.ai/en/stable/api/vllm/model_executor/model_loader/) and [safetensors metadata documentation](https://huggingface.co/docs/safetensors/metadata_parsing) describe those interfaces.

The current adapter deliberately requires one engine worker, one unprefixed dense and unquantized checkpoint, supported byte-aligned scalar dtypes and ordinary tensors. Distributed workers, expert filtering, encoder-only filtering, packed dtypes and tensor subclasses fail or remain unsupported. The bridge is Linux-specific because its descriptor handoff uses `/proc`. Qwen-specific loading has not been validated.

## Agent integration contract

The current agent's `Ticket` still requires legacy `Manifest` and `urls`, and its engine gate waits for complete legacy model hydration. An agent integrating extents needs a tagged model-artifact choice containing an extent ticket, with a distinct representation from `shards-v1` TAR pieces. Runtime and cache artifacts can continue using artifact-v1, and all hydrators should share the existing transfer semaphore.

Engine startup requires complete local config, index, tokenizer and other engine metadata in a stable metadata directory, followed by a private completion socket and its Rust owner. The agent can release a metadata-ready gate and start vLLM once that metadata and the runtime are ready, while weight hydration continues. It passes the socket in `model_loader_extra_config` and installs the adapter as a general plugin in every engine process. The loader must use the bridge for weights instead of opening pre-sized incomplete safetensors through the default loader. The bridge and hydration owner must remain alive until the engine completes assignments; failure cancels both and keeps admission closed.

Inference readiness still requires successful model loading, publication, post-load initialization and the agent's existing health/tokenization/minimal-inference proof. A `done` message is tensor delivery completion, not inference readiness. The existing early-import gate alone does not provide this behavior; its full-model gate and cooperative launcher need separate metadata-ready and tensor-completion gates.

Cache fingerprints should include stable model content identity, adapter implementation/loader ABI, runtime identity, engine arguments and complete GPU compatibility. The content identity's cross-layout stability is useful only after cache producers and consumers use and validate that contract. Existing cache `2b3188be…` depends on the legacy archive identity; its fingerprint cannot be relabeled. The model and plugin runtime must be rebaked, or a separate explicit migration must prove byte content and all compatibility fields before reuse.

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

The benchmark writes raw receipts and logs to its requested output directory. A compact [benchmark receipt](parallel-weight-loading-benchmark.json) accompanies this document, and the benchmark program and legacy helper reproduce the comparison. Unit/integration tests cover invalid paths, offsets, ranges, lengths, digests, duplicate headers, tensor shape/coverage, zero-sized tensors, multi-file discovery, completion/publication races, cancellation and protocol failure. Formatting and Clippy are required before integration.
