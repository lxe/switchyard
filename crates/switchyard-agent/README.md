# switchyard-agent

`switchyard-agent` is the pod-side process that hydrates a serving runtime, model, and compatible compiler caches, starts the inference engine, and exposes authenticated lifecycle observations and inference requests. The provisioning client supplies an expiring hydration ticket; the agent never receives permanent storage credentials. Archive packing, transport, and filesystem verification belong to [`switchyard-artifacts`](../switchyard-artifacts/README.md); this agent owns serving-host compatibility checks, runtime launch, and readiness.

See [RunPod live execution](../../docs/RUNPOD.md) for configuration, workloads, receipts, and recovery.

## Hydration ticket

Set `SWITCHYARD_TICKET_URL` to the signed URL of a JSON [`Ticket`](src/lib.rs), and `SWITCHYARD_AGENT_TOKEN` to a random token containing at least 32 characters. The ticket identifies the model revision, its `shards-v1` manifest, one signed download URL per chunk, vLLM arguments, transfer concurrency, transfer stall timeout, and agent lifetime. Ticket values contain temporary access credentials and must remain outside logs and Git.

### Serving runtime and compiler caches

An optional `runtime` field supplies an [`ArtifactTicket`](../switchyard-artifacts/src/manifest.rs), required compatibility labels, a relative executable `program`, prefix `args`, an `environment` map, and named `caches`. A ticket without this field launches the image's `vllm` executable for compatibility with older deployments. When a runtime is supplied, the agent installs it at `/opt/switchyard/runtime` and invokes its executable at that fixed path. Internal executable symlinks are validated for containment and preserved during invocation so Python can discover its virtual environment.

Each cache artifact must identify the selected `runtime_id`; a declared `model_revision` must also match the model ticket. All required labels are checked before transfer. An artifact's declared `architecture` is compared to the running host. If any runtime or cache declares `gpu_compute_capability`, a bounded `nvidia-smi` probe checks every exposed GPU against that value. Other engine and GPU labels are checked against the ticket's requirements; the deployment integration supplies those requirements.

Verified cache seeds live at `/opt/switchyard/cache/<runtime-id>/.seeds/<name>`. The agent copies each seed into a separately published writable directory at `/opt/switchyard/cache/<runtime-id>/<name>`, retaining a seed identity receipt. Subsequent starts preserve compiler-generated files only when that receipt identifies the same seed. A different seed or an unowned destination is rejected. Cache names `runtime` and `.seeds` are reserved. Runtime and immutable seed directories are verified again before reuse; modifications to writable compiler caches do not invalidate their immutable seeds.

Environment values can use `{runtime_root}` and `{cache_root}` placeholders; the latter expands to the directory for the selected runtime identity. The child receives a small allowlist of inherited host execution variables and offline Hugging Face settings. Cloud and storage credentials are removed, credential-like names in the ticket's environment are rejected, and Python bytecode writes are disabled to preserve the verified runtime tree. The deployment config supplies engine-specific writable cache paths.

Runtime and cache downloads use the ticket's transfer concurrency and stall timeout. Extraction concurrency is capped at eight; each HTTP attempt has a total timeout capped at 600 seconds and at the agent lifetime. The overall agent deadline bounds runtime hydration, model hydration, and engine startup together. Required artifacts are verified before the engine starts; this version does not expose partially extracted files to it.

### Model archive

The agent validates the manifest before performing I/O. Every chunk has a SHA-256 digest, compressed and raw byte counts, a shard number, and a contiguous sequence number. The manifest's archive identity is the SHA-256 digest of the concatenated chunk digest strings in manifest order. A shard is a tar stream split into consecutive chunks. With `compress: "zstd"`, each chunk is a separate compressed frame.

Downloads run concurrently with at most 64 transfers. A failed attempt is retried twice, with bounded delays, and its retained byte count is removed from reported progress. Extraction uses at most eight workers and never exceeds the configured transfer concurrency. Each extracted regular file must appear exactly once in the manifest and have the declared size; optional file SHA-256 digests are checked before the directory is published. Absolute paths, parent traversal, symlinks, hard links, duplicate files, and unlisted files are rejected. A cached directory is reused only after the archive marker and all declared file checks pass.

## Observation and inference API

The agent binds port 8080 before hydration begins. Every request requires `Authorization: Bearer <SWITCHYARD_AGENT_TOKEN>`.

| Endpoint | Behavior |
| --- | --- |
| `GET /state` | Returns model revision, phase, retained download bytes, hydration duration, effective download rate, cache observations, and sanitized failure information. |
| `GET /probe` | Returns the current hydration observation, including transfer measurements already collected by the agent. It does not initiate a separate bandwidth test. |
| `/v1/*`, `/tokenize`, `/health`, `/metrics` | Proxies the same method, query, and body to vLLM on `127.0.0.1:8000` after readiness. Streaming responses remain streaming. |

Lifecycle phases are `booting`, `hydrating`, `initializing`, `ready`, and `failed`. `ready` requires engine `/health`, successful tokenization, and a nonstream completion confirming one decoded token. Each readiness request has a 60-second deadline and a 1 MiB response bound; failure diagnostics omit the engine response body.

Model `volume_bytes_per_s` is effective retained download progress divided by elapsed transfer time, so its duration includes failed attempts, storage service delays, and local writing overhead. `hydration_seconds` includes model extraction after downloads complete. An image cache observation remains `null` because a process inside a newly started container cannot determine whether the cloud host downloaded the container image.

Runtime and cache counters are separate from the model counters. `artifact_bytes_downloaded` counts verified compressed chunks, including verified cache hits, and never exceeds `artifact_bytes_total`. `artifact_network_bytes_downloaded` counts received network bytes, including retries. `artifact_volume_bytes_per_s` divides newly downloaded, verified chunk bytes by elapsed runtime/cache hydration time; it excludes cached bytes and duplicated retry bytes from the numerator. Fully cached artifacts produce no network rate observation. The `artifacts` list records each runtime or cache identity, phase, verification progress, cache hits, and sanitized failures without signed URLs. These rates describe observed transfers and do not guarantee future network capacity.

The proxy accepts bodies up to 2 MiB, returns 503 while the engine starts, and never forwards the agent token to vLLM. The lifetime deadline aborts the engine task and permits at most five seconds for HTTP shutdown. **The provisioning client must terminate the cloud pod:** stopping the agent process does not stop cloud rental billing.

## Verification

```sh
cargo test -p switchyard-agent
cargo clippy -p switchyard-agent --all-targets -- -D warnings
```

Tests serve synthetic archives over local HTTP and check chunk boundaries, zstd framing, retries, verified and wire-byte accounting, stall deadlines, compatibility rejection, writable-cache reuse, executable symlinks, corrupt data, unsafe archive entries, authentication, readiness inference, request bounds, and streaming proxy behavior. They do not require a GPU or cloud credentials.
