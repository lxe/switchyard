# switchyard-agent

`switchyard-agent` is the pod-side process that hydrates a serving runtime, model, and compatible compiler caches, starts the inference engine, and exposes authenticated lifecycle observations and inference requests. The provisioning client supplies an expiring hydration ticket; the agent never receives permanent storage credentials. Archive packing, transport, and filesystem verification belong to [`switchyard-artifacts`](../switchyard-artifacts/README.md); this agent owns serving-host compatibility checks, runtime launch, and readiness.

See [RunPod live execution](../../docs/RUNPOD.md) for configuration, workloads, receipts, and recovery.

## Hydration ticket

Set `SWITCHYARD_TICKET_URL` to the signed URL of a JSON [`Ticket`](src/lib.rs), and `SWITCHYARD_AGENT_TOKEN` to a random token containing at least 32 characters. The ticket identifies the model revision, its `shards-v1` manifest, one signed download URL per chunk, vLLM arguments, transfer concurrency, transfer stall timeout, and agent lifetime. Ticket values contain temporary access credentials and must remain outside logs and Git.

### Serving runtime and compiler caches

An optional `runtime` field supplies an [`ArtifactTicket`](../switchyard-artifacts/src/manifest.rs), required compatibility labels, a relative executable `program`, prefix `args`, an `environment` map, and named `caches`. A ticket without this field launches the image's `vllm` executable for compatibility with older deployments. When a runtime is supplied, the agent installs it at `/opt/switchyard/runtime` and invokes its executable at that fixed path. Internal executable symlinks are validated for containment and preserved during invocation so Python can discover its virtual environment.

Each cache artifact must identify the selected `runtime_id`; a declared `model_revision` must also match the model ticket. All required labels are checked before transfer. An artifact's declared `architecture` is compared to the running host. If any runtime or cache declares `gpu_compute_capability`, a bounded `nvidia-smi` probe checks every exposed GPU against that value. Other engine and GPU labels are checked against the ticket's requirements; the deployment integration supplies those requirements.

Immutable cache seeds live at `/opt/switchyard/cache/<runtime-id>/.seeds/<name>`. The agent copies each seed into a separately published writable directory at `/opt/switchyard/cache/<runtime-id>/<name>`, retaining a seed identity receipt. Subsequent starts preserve compiler-generated files only when that receipt identifies the same seed. A different seed or an unowned destination is rejected. Cache names `runtime` and `.seeds` are reserved. In the default verified mode, runtime and immutable seed directories are verified again before reuse. Trusted streaming uses their pinned identity receipts; modifications to writable compiler caches do not invalidate their immutable seeds.

Environment values can use `{runtime_root}` and `{cache_root}` placeholders; the latter expands to the directory for the selected runtime identity. The child receives a small allowlist of inherited host execution variables and offline Hugging Face settings. Cloud and storage credentials are removed, credential-like names in the ticket's environment are rejected, and Python bytecode writes are disabled to preserve the verified runtime tree. The deployment config supplies engine-specific writable cache paths.

Runtime and model hydration run concurrently. A shared semaphore limits all runtime, cache, and model HTTP streams to the ticket's concurrency; configuring twenty means twenty in total. Trusted runtime, component, and cache shards pass HTTP response bodies directly to independent zstd/tar decoders, with up to twenty active streams. There is no compressed reception spool on this path. The decoders write disjoint files into a private staging tree and publish the completed artifact atomically. Each worker retains its shared permit until decoding finishes, including after cancellation. Verified extraction remains capped at eight workers.

Direct streaming uses bounded body buffering, so a slow decoder can backpressure its own HTTP stream. The objective is minimum download-and-extraction elapsed time; the agent does not reserve disk space for an entire compressed artifact merely to keep the network receiving. Runtime transport observations retain per-object received bytes, body wait time, completion, and failure. Compressed spool write counters remain zero for direct streaming. Each HTTP attempt has a stall timeout and a total timeout capped at 600 seconds and at the agent lifetime. Waiting for a decoder to request its next body frame does not count as a network stall. The overall agent deadline bounds hydration and engine startup together.

### Startup policy

The optional `startup` field selects storage and hydration behavior. Omitted fields preserve verified hydration, `/work/switchyard/model`, and launch after both dependencies are ready.

```json
{
  "mode": "trusted_streaming",
  "model_root": "/dev/shm/switchyard",
  "require_model_ram": true,
  "early_launch": true
}
```

`model_root` is the absolute parent directory of the published `model` directory. It must contain real directories and cannot overlap runtime, cache, or chunk storage. Before any transfer, the agent measures the model filesystem's available bytes, its RAM backing, host `MemAvailable`, and any discoverable cgroup memory headroom. A RAM requirement fails when the filesystem is not RAM-backed or has insufficient capacity; there is no disk fallback. Memory checks reserve the model file bytes plus 136 MiB per configured concurrent stream and 64 MiB of fixed headroom. This covers the 128 MiB maximum zstd history per active decoder and bounded channel/network buffers. Engine memory requirements remain a deployment sizing responsibility. The model is extracted directly into a private directory on its selected filesystem and atomically published there.

`trusted_streaming` is for pinned artifacts whose manifest and contents the controller already trusts. It skips content hashes, identity rehashing, complete-tree rereads, compressed chunk cache writes, and per-file fsync. It still checks archive structure, containment, entry types, declared compressed/raw/file lengths, and decompressor validity. Receipt reuse checks the real destination directory and a regular mode-0600 identity receipt without rescanning its contents; the agent must exclusively own these directories. `verified` retains content and cache verification.

`early_launch` requires an explicitly supplied cooperative runtime program. After runtime and cache hydration finish, the agent launches that program with piped stdin while model hydration continues. The program may perform imports or other preparation, then must wait for exactly `switchyard-model-ready-v1\n` followed by EOF before opening model files. The agent sends that line only after the current model directory is published. A failed or cancelled model transfer kills the owned engine process group without sending the gate. After successful gate delivery, the agent observes the engine HTTP health endpoint and immediately accepts client traffic once it is available. It sends no tokenization or inference request during startup. Engine-specific preparation belongs in the configured program; the agent contains no vLLM import logic.

### Model archive

The agent validates the manifest before performing I/O. Every chunk has a SHA-256 digest, compressed and raw byte counts, a shard number, and a contiguous sequence number. The manifest's archive identity is the SHA-256 digest of the concatenated chunk digest strings in manifest order. A shard is a tar stream split into consecutive chunks. With `compress: "zstd"`, each chunk is a separate compressed frame.

In verified mode, downloads run concurrently with at most 64 transfers. A failed attempt is retried twice, with bounded delays, and its retained byte count is removed from reported progress. Extraction begins after all chunks arrive, uses at most eight workers, and never exceeds the configured transfer concurrency. Each extracted regular file must appear exactly once in the manifest and have the declared size; optional file SHA-256 digests are checked before the directory is published. Absolute paths, parent traversal, symlinks, hard links, duplicate files, and unlisted files are rejected. In verified mode, a cached directory is reused only after the archive marker and all declared file checks pass. In trusted streaming mode, bounded 64 KiB channels feed each shard's ordered TAR pieces directly to extraction, with up to twenty active shard workers. Download slots remain occupied until each piece is decoded and its declared raw length is checked, so completed small pieces cannot accumulate the whole model in memory. A partial legacy TAR transfer fails the hydration rather than replaying bytes that extraction has already consumed; runtime artifacts have independent shard retries. Private staging remains owned by workers until they stop, and only the awaiting caller can publish it.

## Observation and inference API

The agent binds port 8080 before hydration begins. Every request requires `Authorization: Bearer <SWITCHYARD_AGENT_TOKEN>`.

| Endpoint | Behavior |
| --- | --- |
| `GET /state` | Returns model revision, phase, retained download bytes, hydration duration, effective download rate, cache observations, and sanitized failure information. |
| `GET /probe` | Returns the current hydration observation, including transfer measurements already collected by the agent. It does not initiate a separate bandwidth test. |
| `/v1/*`, `/tokenize`, `/health`, `/metrics` | Proxies the same method, query, and body to vLLM on `127.0.0.1:8000` after readiness. Streaming responses remain streaming. |

Lifecycle phases are `booting`, `hydrating`, `initializing`, `ready`, `exporting`, and `failed`. `ready` means the engine HTTP health endpoint is available. Availability is polled at 100 ms intervals; the agent performs no tokenization or decode verification. The first client request pays any remaining lazy initialization costs. The historical `inference_verified` field remains false on new launches.

Model `volume_bytes_per_s` is effective retained download progress divided by elapsed transfer time, so its duration includes failed attempts, storage service delays, and local writing overhead. `hydration_seconds` includes model extraction; trusted-mode extraction overlaps downloads. An image cache observation remains `null` because a process inside a newly started container cannot determine whether the cloud host downloaded the container image.

Runtime and cache counters are separate from the model counters. `artifact_bytes_downloaded` counts verified compressed chunks, including verified cache hits, and never exceeds `artifact_bytes_total`. `artifact_network_bytes_downloaded` counts received network bytes, including retries. `artifact_volume_bytes_per_s` divides newly downloaded, verified chunk bytes by elapsed runtime/cache hydration time; it excludes cached bytes and duplicated retry bytes from the numerator. Fully cached artifacts produce no network rate observation. The `artifacts` list records each runtime or cache identity, phase, verification progress, cache hits, and sanitized failures without signed URLs. `artifact_network_bytes_per_s` reports actual received wire bytes divided by elapsed runtime/cache hydration time, including retries, and is available in trusted mode. Trusted verification counters remain zero. These rates describe observed transfers and do not guarantee future network capacity.

`model_storage` and `runtime_storage` report the measured filesystem, mount source, available bytes, and memory headroom. An overlay filesystem is reported as overlay; its backing device is not assumed to be NVMe. `startup_milestones` records monotonic seconds from process start for model/runtime hydration, engine spawn, model gate release, and decoded-token readiness. These timestamps do not measure cloud allocation or container startup before the agent ran.

The proxy accepts bodies up to 2 MiB, returns 503 while the engine is unavailable, and never forwards the agent token to vLLM. The Linux agent owns a separate process group for the engine and kills that group when its owner is cancelled. The lifetime deadline aborts engine and export work and permits at most five seconds for HTTP shutdown. **The provisioning client must terminate the cloud pod:** stopping the agent process does not stop cloud rental billing.

## Compiler cache export

A bake ticket can opt into a one-shot export with `runtime.cache_export`. Ordinary serving tickets omit it and all export routes return 404. The capability selects a single writable cache child, such as `compiled`, and fixes the output artifact's compatibility labels in the ticket. It accepts no client-selected filesystem paths, shell commands, or HTTP metadata. The selected name follows cache naming rules, including the reserved names `runtime` and `.seeds`.

```json
{
  "name": "compiled",
  "compatibility": {
    "runtime_id": "<runtime artifact identity>",
    "model_revision": "<model revision>",
    "architecture": "x86_64",
    "gpu_compute_capability": "8.9",
    "gpu": "L40S",
    "engine_configuration": "<engine configuration identity>",
    "runtime_launch_configuration": "<runtime launch identity>"
  },
  "drain_timeout_s": 60,
  "shutdown_timeout_s": 30
}
```

`runtime_id` and `model_revision` must match the ticket. `architecture` must identify the actual host; a declared `gpu_compute_capability` participates in the same bounded GPU probe as hydrated artifacts. The controller supplies other compatibility labels and must include every runtime, model, engine, launch, and GPU characteristic needed for safe cache reuse. The two timeouts default to the values shown; valid ranges are 1–3600 seconds for drainage and 1–300 seconds for shutdown. The ticket's overall lifetime still bounds the operation.

| Endpoint | Behavior |
| --- | --- |
| `POST /cache/export` | Requires an empty body and a ready engine. Closes inference admission and starts one export, returning 202 while running, 200 after completion, or 409 before readiness or after failure. Repeated requests return the existing result. |
| `GET /cache/export` | Returns `{ "phase", "manifest", "error" }`. Phases are `idle`, `draining`, `stopping`, `packing`, `ready`, and `failed`; only `ready` includes a manifest. |
| `GET /cache/export/chunks/{name}` | Streams an exact manifest-listed chunk with `Content-Length` and `x-artifact-sha256`. Unknown names return 404; unfinished exports return 409. |

Every export request uses the same bearer authentication as inference. Once POST is accepted, `/state` changes to `exporting` and new inference receives 503. Admission ownership remains attached to each upstream request and its complete response body. The exporter waits for EOF, an upstream error, or client disconnection before requesting shutdown from the engine owner. That owner sends SIGTERM to the whole process group and checks Linux `/proc` until both the child and every non-zombie group member have exited. A shutdown timeout forces group cleanup and fails the export instead of publishing caches that may not have flushed. Drainage, shutdown, or packing failure leaves admission closed and reports `failed`.

After confirmed shutdown, the agent snapshots `/opt/switchyard/cache/<runtime-id>/<name>` into private export scratch and calls the generic artifact packer. The snapshot preserves file content, modes, directories, and safe internal symlinks. It omits the agent-owned `.switchyard-cache-seed` receipt and never traverses or changes immutable seed trees. Special files, escaping links, empty file content across the entire cache, and sources exceeding 16 GiB are rejected. Packing uses the generic defaults of 128 MiB shard targets and zstd level 6. The agent verifies the snapshot and every output chunk before exposing a manifest whose kind is `cache`, whose compatibility labels match the ticket, and whose attributes are empty.

A completed export leaves `/state` in `exporting`, with `inference_verified: false`, because the engine is stopped. Export chunks remain available until agent shutdown. A cancelled packing waiter cannot expose late worker output, and private scratch is removed when its ownership ends. Compression already running in a blocking worker may finish before its private files are removed; it cannot publish a result after cancellation. The controller should download and verify all chunks with its authenticated client, terminate the GPU pod, then publish the generic artifact with its own storage credentials. File caches do not preserve in-memory CUDA graphs or engine process state.

## Model hydration diagnostics

Authenticated `/state` and `/probe` observations include `model_diagnostics` while a legacy model is hydrated in `trusted_streaming` mode. Each transfer reports its TAR shard and sequence, the operation currently awaited, and separate counters for response-body receipt, channel delivery, compressed bytes read by the decoder, and decoded TAR bytes. Each shard reports its active sequence, extraction boundary, decoded bytes, and bytes actually written to model files. TAR headers and padding count as decoded bytes but do not count as file writes. These counters describe data movement; trusted mode does not verify content hashes.

All diagnostic timestamps are seconds from one `std::time::Instant` captured immediately before model hydration starts. They share an origin across model transfers and shard readers. They are separate from the agent-start origin used by `startup_milestones`. `elapsed_s` is the time of serialization; `boundary_since_s` and `consumer_since_s` identify the last boundary transition. Their corresponding ages describe an in-progress wait even when the aggregate byte count is unchanged.

The downloader's `boundary` can be `pending`, `permit_wait`, `headers_wait`, `body_wait`, `progress_state`, `channel_send`, `body_complete`, `decoder_completion`, `complete`, or `failed`. The independent `consumer_boundary` reports `pending`, `decoding`, `channel_receive`, `archive`, `complete`, or `read_error`. The shard boundary distinguishes `piece_read`, `archive`, and `file_write`, with terminal extraction/error states. An ahead piece with `channel_send`, zero sampled available slots, and an untouched consumer is waiting for ordered extraction. A `body_wait` with spare channel slots means the caller is awaiting a response frame. That observation alone cannot locate the cause within the remote service, network, client transport, or local scheduling.

`request_s`, `headers_s`, `first_body_s`, `last_body_s`, and `body_eof_s` describe request progress. The HTTP version, status, declared response length, and numeric remote socket address are retained. `permit_wait_s` and `permit_held_s` include current unfinished intervals. `channel_wait_s` and `body_wait_s` sum completed waits; add the current boundary age when measuring an unfinished wait. `longest_body_wait_s` reports the longest completed frame wait, including a timed-out wait. Channel capacity is a sample taken around sends and receives. Counter updates on the producer and blocking consumer are independent, so a snapshot can briefly show consumption ahead of delivery; completed transfers conserve all three wire-byte counters.

Diagnostics retain at most 256 transfer records and 64 shard records. Omitted counts are explicit. Buffer sizes, ordered extraction, semaphore ownership, HTTP settings, and retry behavior are unchanged. Updates use small per-record mutexes and fixed boundary enums. Serialization samples the live records; there is no background telemetry task or per-frame event log in the agent.

The first transfer failure freezes the current transfer and shard observations, then synchronously samples Linux process/task CPU ticks, memory, process I/O, and owned TCP socket tables. The agent publishes the failure only after capture completes and retains the channel sender until then, so decoder EOF or async cancellation cannot leave a placeholder host snapshot. Response-frame failures and body-wait deadlines capture while the response stream is still owned. `host_capture_complete` records completion of this sampling attempt, including explicitly unavailable proc data; `host_capture_seconds` records its duration. Older observations deserialize with `host_capture_complete: false`.

The fixed host allowlists exclude command lines, environment variables, URLs, headers, and response bodies. Task and socket counts are capped at 64 and 128; missing proc data is reported. Task CPU times are kernel clock ticks; memory fields from `meminfo` are KiB, and I/O and socket queue fields are bytes. TCP addresses and states use Linux proc hexadecimal notation. Socket tables provide queues, retransmission counters, and timer state; they do not provide packet traces or establish which endpoint caused a wait.

Reqwest errors add `transport_error` to the transfer observation. It records the `timeout`, `connect`, `dns`, `request`, `body`, `decode`, `builder`, `status`, `redirect`, and `upgrade` predicates, plus at most 16 typed causes in source-chain order. Known causes retain reqwest predicates, Hyper predicates, or the underlying I/O `ErrorKind` and numeric `raw_os_error`; other types use the `opaque` category. `causes_truncated` marks a longer chain. Error display text and arbitrary source type names are never recorded. `reqwest_timeout` means reqwest's timeout predicate was true, which can include an underlying timed-out I/O error; it does not identify the configured request timer. `body_wait_deadline` and `headers_wait_deadline` identify the agent's explicit wait timers. Historical `request_timeout` records remain readable.

Runtime and cache retry options are supplied through `bootstrap.runtime.hydration` in the checked fleet configuration and copied into `RuntimeTicket.hydration`. Set `{ "retries": 0 }` for diagnostic reproductions. `retries` counts additional attempts; values from 0 through 10 are accepted, and an omitted `hydration` preserves the existing two retries. Trusted model streaming already performs one attempt. Verified legacy model hydration retains its existing retry policy, so diagnostic reproductions use trusted mode. There is no environment override for these options.

### Explicit local replay

The ignored test `hydrate::diagnostics::local::production_pipeline_without_retries` accepts a JSON input on stdin. Its fields are `ticket`, optional generic `runtime` artifact ticket, absolute `model_root`, absolute `runtime_root`, and `receipt` output path. Supply a trusted-streaming model ticket, fresh exclusively owned destinations, and short-lived download capabilities directly through the pipe. Ticket JSON must never be written into receipts or logs.

```sh
<ticket-generator> | cargo test -p switchyard-agent --lib \
  hydrate::diagnostics::local::production_pipeline_without_retries \
  -- --exact --ignored --nocapture
```

The replay uses the production model preflight, shared semaphore, model hydrator, and generic runtime hydrator. Runtime retries are explicitly zero, and trusted model streaming performs one attempt per piece. It writes state samples every 250 ms and a terminal receipt, with a 600-second run deadline. Start with model hydration alone; then supply the same pinned runtime as the single changed variable. This replay does not launch the engine, verify GPU compatibility, or reproduce the cloud filesystem and network. Those differences must accompany any causal claim. The caller owns cleanup of its output roots after inspecting the receipt.

## Verification

```sh
cargo test -p switchyard-agent
cargo clippy -p switchyard-agent --all-targets -- -D warnings
```

Tests serve synthetic archives over local HTTP and check twenty simultaneous streams, extraction before the final transfer, runtime imports during model hydration, gate ordering, cancellation ownership and permit retention, RAM capacity, trusted receipts, and chunk boundaries, zstd framing, retries, verified and wire-byte accounting, stall deadlines, compatibility rejection, writable-cache reuse, executable symlinks, corrupt data, unsafe archive entries, authentication, readiness inference, request bounds, and streaming proxy behavior. Export tests cover disabled capabilities, idempotent initiation, active response drainage through EOF and disconnect, invalid download names, output checksums, seed preservation, cancellation, descendant shutdown, and packing failures. They do not require a GPU or cloud credentials.
