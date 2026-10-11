# Runtime hydration profile

Trusted runtime hydration passes each independently compressed filesystem shard from its HTTP response into a zstd/tar decoder. The agent runs up to twenty shard workers, bounded by the ticket concurrency and its shared transfer semaphore. Workers extract disjoint files into a private staging directory before atomic publication. Runtime, component, and cache artifacts use this path without compressed spool files. Verified hydration continues to retain checked compressed chunks.

The [measurement receipt](receipts/runtime-hydration-profile.json) profiles the same 67-shard runtime artifact used by the local serving rollout: 4.56 GB compressed, 12.08 GB of file content and 95,427 filesystem entries. Each output directory was fresh. Source objects were served over local HTTP with warm page cache; no cache eviction was performed. These measurements exclude container creation and inference startup.

## Elapsed time

| Variant | Hydration process time |
| --- | ---: |
| Static Rust buffered hydration on host, four extraction workers | 5.358 s |
| Static Rust direct hydration on host, twenty workers | 2.930 s |
| Ubuntu buffered hydration, four extraction workers | 9.160 s |
| Ubuntu buffered hydration, twenty extraction workers | 6.659 s |
| Ubuntu direct hydration, twenty workers | 3.045 s |

Decompression to discarded output took 2.375 seconds with four workers and 1.247 seconds with twenty. Twenty-worker decompression averaged 11.80 busy logical cores on the 24-logical-CPU host. Ubuntu direct hydration averaged 7.31 busy logical cores, with zero cgroup CPU throttling and about 12.61 GB peak charged memory. Container CPU measurements use cgroup counters rather than the docker client process. HTTP server CPU is outside the consumer accounting.

The Ubuntu container had a 32 GiB memory cap with swap disabled. Output resides on NVMe-backed Btrfs with transparent zstd:3 compression. Each variant has one sample. The traced host variant includes additional tracing overhead; its summed syscall wait times are not wall-clock phase durations. Disabling copy-on-write only on the compressed spool did not improve the four-worker host sample.

## Transport and capacity

Direct hydration retains bounded HTTP-body buffering and allows extraction to backpressure each stream. Twenty is the active-worker limit; artifacts can contain more than twenty shards so workers pick up subsequent shards as they finish. The goal is minimum complete hydration elapsed time, rather than an independent guarantee of continuous network saturation.

Trusted runtime capacity accounting reserves output files and decoder memory, without reserving complete compressed backing. The worker retains its shared transfer permit through extraction and cancellation. Per-object transport telemetry reports compressed bytes received, body wait time, completion, and failure; compressed spool write counters are zero. No new serving-time hashes, tree scans, or per-file fsyncs are introduced.

## Integrated startup gap

The latest local CPU-snapshot serving rollout took 49.954 seconds to first streamed content. Runtime reception completed in 3.866 seconds and extraction completed in 30.191 seconds, measured from that artifact's own start. Standalone direct hydration demonstrates that the runtime does not inherently require thirty seconds to materialize. It does not attribute the integrated delay to a specific cause. Concurrent artifact work, agent progress/state handling, filesystem writeback and instrumentation remain separate variables to profile using the selected direct path.

The inference weights retain their independent bounded tensor transport; this runtime change does not select a different weight representation or loader.
