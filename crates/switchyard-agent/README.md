# switchyard-agent

`switchyard-agent` is the pod-side process that downloads a model archive, verifies its files, starts vLLM, and exposes authenticated lifecycle observations and inference requests. The provisioning client supplies an expiring hydration ticket; the agent never receives permanent storage credentials.

See [RunPod live execution](../../docs/RUNPOD.md) for configuration, workloads, receipts, and recovery.

## Hydration ticket

Set `SWITCHYARD_TICKET_URL` to the signed URL of a JSON [`Ticket`](src/lib.rs), and `SWITCHYARD_AGENT_TOKEN` to a random token containing at least 32 characters. The ticket identifies the model revision, its `shards-v1` manifest, one signed download URL per chunk, vLLM arguments, transfer concurrency, transfer stall timeout, and agent lifetime. Ticket values contain temporary access credentials and must remain outside logs and Git.

The agent validates the manifest before performing I/O. Every chunk has a SHA-256 digest, compressed and raw byte counts, a shard number, and a contiguous sequence number. The manifest's archive identity is the SHA-256 digest of the concatenated chunk digest strings in manifest order. A shard is a tar stream split into consecutive chunks. With `compress: "zstd"`, each chunk is a separate compressed frame.

Downloads run concurrently with at most 64 transfers. A failed attempt is retried twice, with bounded delays, and its retained byte count is removed from reported progress. Extraction uses at most eight workers and never exceeds the configured transfer concurrency. Each extracted regular file must appear exactly once in the manifest and have the declared size; optional file SHA-256 digests are checked before the directory is published. Absolute paths, parent traversal, symlinks, hard links, duplicate files, and unlisted files are rejected. A cached directory is reused only after the archive marker and all declared file checks pass.

## Observation and inference API

The agent binds port 8080 before hydration begins. Every request requires `Authorization: Bearer <SWITCHYARD_AGENT_TOKEN>`.

| Endpoint | Behavior |
| --- | --- |
| `GET /state` | Returns model revision, phase, retained download bytes, hydration duration, effective download rate, cache observations, and sanitized failure information. |
| `GET /probe` | Returns the current hydration observation, including transfer measurements already collected by the agent. It does not initiate a separate bandwidth test. |
| `/v1/*`, `/tokenize`, `/health`, `/metrics` | Proxies the same method, query, and body to vLLM on `127.0.0.1:8000` after readiness. Streaming responses remain streaming. |

Lifecycle phases are `booting`, `hydrating`, `initializing`, `ready`, and `failed`. `volume_bytes_per_s` is effective retained download progress divided by elapsed transfer time, so it includes failed attempts, storage service delays, and local writing overhead. It is an observation of this transfer, rather than a network capacity guarantee. `hydration_seconds` includes extraction after downloads complete. An image cache observation remains `null` because a process inside a newly started container cannot determine whether the cloud host downloaded the container image.

The proxy accepts bodies up to 2 MiB, returns 503 while the engine starts, and never forwards the agent token to vLLM. The lifetime deadline aborts the engine task and permits at most five seconds for HTTP shutdown. **The provisioning client must terminate the cloud pod:** stopping the agent process does not stop cloud rental billing.

## Verification

```sh
cargo test -p switchyard-agent
cargo clippy -p switchyard-agent --all-targets -- -D warnings
```

Tests serve synthetic archives over local HTTP and check chunk boundaries, zstd framing, retries, cache validation, corrupt data, unsafe archive entries, authentication, request bounds, and streaming proxy behavior. They do not require a GPU or cloud credentials.
