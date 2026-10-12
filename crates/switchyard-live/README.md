# switchyard-live

A live execution backend for the simulator scheduling contract. It plans inputs from pinned tokenizer assets, admits and routes requests, records real streamed usage and timing, and cleans up owned provider resources.

See [RunPod live execution](../../docs/RUNPOD.md) for configuration, workloads, receipts, and recovery.

## Workload failure receipts

Successful workloads write `workloads/<name>.json` with streamed completions and aggregate timing. If a provider, host observation, routing operation, or run limit returns an error, `workloads/<name>-partial.json` preserves the completions already observed by the controller, request states, attempt counts, elapsed time, and estimated rental cost. The original error still stops the workload, and owned-resource cleanup follows through the existing runtime.

A partial receipt does not claim that requests still in flight completed. External cancellation can interrupt the workload before this error handler runs; ownership journals and incremental observations remain the evidence for such interruptions. Read the outcome and cleanup receipts alongside request measurements.

## Compiler-cache preparation

`switchyard live bake` provisions one configured model on a selected GPU and runs
the request shapes in a cache preparation plan. The agent closes admission,
drains streamed responses, confirms that the engine's process group has stopped,
and packs the selected writable cache subtree. The controller checks the exported
manifest, downloads and hashes every chunk, and confirms GPU termination before
publishing the artifact to object storage.

```sh
cargo run --release --features live -- live bake \
  --config configs/runpod/qwen-small.json \
  --plan configs/runpod/cache-bake-l40s-0.8b.json \
  --env-file /path/to/compute.env --env-file /path/to/storage.env \
  --out results/cache-bake-0.8b
```

The output includes ownership and rental receipts, workload measurements,
`cache-publication.json`, and `cache-configuration.json`. Add the latter object to
the matching model's `caches` array to hydrate it on subsequent allocations. The
cache name must match the writable subtree used by the runtime environment; the
example uses `{cache_root}/compiled` and exports `compiled`.

Required compatibility labels pin the runtime artifact, model repository,
revision and archive, engine settings, launch program/arguments/environment,
GPU type, architecture, and the plan's compute capability. Additional labels may
constrain reuse further. Changing any pinned setting requires new preparation.
Existing cache seeds can be inputs to another preparation run; the agent keeps
those seeds immutable and exports the writable copy.

The example fleet uses eager execution. Its preparation captures on-demand
Triton, FlashInfer, and CUDA file caches for the exercised request shapes. It
does not enable `torch.compile` or persist CUDA graphs. Removing `--enforce-eager`
changes engine identity and requires a separate cache preparation run. File-cache
reuse may reduce compilation work; fresh processes still initialize the engine,
load model weights, allocate device memory, and verify serving readiness.

## Scheduler integration

[Inference scheduling](../../docs/SCHEDULING.md) describes deployment-bound measurements, background allocation reservations, feedback histories, and ownership-safe cleanup. The checked-in measured L4 and RTX 4090 configurations reuse cloud observations while retaining their known simulation accuracy limitations.
