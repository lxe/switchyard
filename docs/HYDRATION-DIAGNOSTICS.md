# Model hydration stall diagnosis

The unexplained failure is a trusted legacy TAR hydration that stopped at 2,824,342,916 of 4,571,330,560 transferred bytes and reported `chunk-00020 stalled`. Existing aggregate progress cannot establish whether the awaited body frame was delayed by transport, prior consumer backpressure, local scheduling, or another cause. A body-wait deadline is an observation of the caller's wait, not an attribution to object storage or the compute provider.

## Observable boundaries

The agent's `model_diagnostics` field records each bounded transfer's request, headers, first and last body receipt, EOF, permit lifecycle, and current transfer and consumer boundaries. Receipt, channel delivery, decoder input, decoded TAR output, and filesystem writes have distinct counters. The independent shard reader identifies the active TAR sequence and distinguishes reads from file writes. The [agent documentation](../crates/switchyard-agent/README.md#model-hydration-diagnostics) defines the timing origin, states, units, record caps, sampling semantics, and safe failure classifications.

At the first transfer failure, the agent preserves transfer and shard observations before cancellation. At a body-wait deadline it captures bounded Linux task, memory, process I/O, and owned TCP socket information while the response stream exists. This provides socket queues and timer/retransmission information without recording signed URLs, authorization headers, process arguments, environment variables, or raw response bodies. Packet loss, remote service behavior, and a particular provider remain unconfirmed unless additional evidence establishes them.

## Local replay evidence

Two sequential replays on 2026-10-10 used the pinned uncompressed 2B archive: 34 ordered 128 MiB-or-smaller pieces for one safetensors TAR and seven metadata TARs. Both used the production preflight and streaming hydrator, four 64 KiB channel slots per piece, twenty shared permits, a 20-second actively awaited network deadline, a 600-second total HTTP deadline, and zero retries. The second replay added the pinned runtime artifact under the same permit pool. Runtime/cache retries were explicitly zero.

| Replay | Terminal receipt time | Model body bytes | Model file bytes written | Largest sampled channel-send wait age | Longest completed body-frame wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model only | 71.843 s | 4,571,330,560 | 4,571,274,023 | 42.698 s | 0.249 s |
| Concurrent runtime/model | 92.482 s | 4,571,330,560 | 4,571,274,023 | 59.562 s | 3.521 s |

In the model-only replay, piece 20 received all 134,217,728 bytes after accumulating 42.248 seconds of channel-send waits. Its longest completed body-frame wait was 0.100 seconds. Both replays conserved received, delivered, and consumed bytes for every piece and completed without a transfer failure. Post-run model checks matched all manifest-declared sizes and available SHA256 hashes. All 174,597 regular runtime files matched their declared SHA256 hashes, covering 19,361,161,949 file bytes.

The retained local receipts are `results/model-baseline.ndjson`, `results/runtime-overlap.ndjson`, and `results/local-replay-summary.json` in the diagnostic checkout. These are measurements from a debug build on a shared local workstation using tmpfs for models, a persistent local filesystem for runtime extraction, and an IPv6 HTTP/1.1 path to the object endpoint. Desktop work and validation compilation were concurrent host activity. No GPU, engine import, inference, or paid pod was used. The runs test transfer conservation and diagnostic boundaries; they are not a controlled performance comparison or a reproduction of the original cloud path.

## Cloud reproduction proposal

The next useful reproduction preserves the original bootstrap image, model and runtime identities, overlay runtime destination, tmpfs model destination, twenty shared permits, four 64 KiB channel slots, 20-second stall timer, 600-second HTTP deadline, and cooperative early-import program. Replace the published agent with the checksum-pinned telemetry binary from reviewed source. Set `bootstrap.runtime.hydration.retries` to `0` in the checked fleet configuration; trusted model streaming already has no retries. Do not introduce range requests, replay failed TAR pieces, change manifests, alter concurrency, or provision another pod after failure.

The coordinating controller should own exactly one L40S allocation, reserve $0.50 in the shared ledger, accept only a fresh all-in quote at or below $2.50/hour, and persist its exact pod ID/name before monitoring. Poll authenticated state every 250 ms. Stop after model publication or the first failure; enforce an independent 600-second deadline measured from allocation, including image startup. Keep a parent-owned termination guard, allow 60 seconds for cleanup, and confirm permanent termination or authoritative HTTP404 absence independently. The 660-second ceiling corresponds to $0.4584 at the maximum accepted rate. A historical $1.10096/hour quote would cost approximately $0.202 for that interval; it is not a current offer.

The placement is US-TX-3, matching the preserved failed-run rental receipt and ownership journal. The checked `configs/runpod/hydration-diagnostic-2b.json` specifies the bounded fleet and zero runtime/cache retries. Its diagnostic agent URL is reserved for parent publication and must resolve to the pinned telemetry checksum before allocation; it is not a published release. The prepared `results/cloud-diagnostic-proposal.json` contains the complete preserved runtime early-import arguments, model settings, budget, stop conditions, ownership and cleanup requirements. It has not allocated a pod.

The local results leave the 2B cloud failure unconfirmed. No stall recovery patch is justified by these receipts. A failed cloud replay should be assessed using the frozen wait boundary, piece offset, queue and decoder state, permit availability, socket queues/timers, and host task/CPU/I/O/memory evidence together. Any proposed repair then needs a causal reproduction and a repeat that removes the failure while keeping retries disabled.

### Controller API plan

The cloud controller needs a short diagnostic wrapper inside `switchyard-live::runtime`; that wrapper is a parent integration step and is not implemented by the local replay test. Construct `LiveRuntime::new(provider, storage, configuration.execution(), Policy::Adaptive, fresh_output_root)` using the checked single-model, single-offering fleet. Call the existing `provision(0)` exactly once. This writes `pending-launches.json` before creation and `owned-pods.json` before endpoint setup. It also creates the original serving ticket and cooperative runtime command through the existing lifecycle module.

Poll the existing `observe()` method under a bounded two-second control-request timeout and a 250 ms interval. Inspect the latest sanitized observation for the `ModelHydrationFinished` milestone, which follows model publication. Stop on that milestone or the first agent failure. A byte counter reaching its declared total is insufficient because queued data may still await extraction and publication. Enclose provisioning and polling in an overall 600-second timeout and Ctrl-C branch, then call `runtime.cleanup()` unconditionally before saving the final outcome. Do not call `bake_cache`, `run_workload`, cache export, or another provision action. Cleanup preserves failed observations, writes the rental receipt, reconciles exact pending launch names, and confirms termination.

The parent can verify the fresh offering without provisioning using the existing command:

```sh
cargo run --features live -- live discover \
  --config configs/runpod/hydration-diagnostic-2b.json \
  --env-file "$RUNPOD_ENV_FILE" --out results/hydration-diagnostic-quote
```

Use the existing recovery command for the independent parent-owned termination guard, with both locally supplied credential files and the exact diagnostic output root:

```sh
cargo run --features live -- live recover \
  --config configs/runpod/hydration-diagnostic-2b.json \
  --env-file "$RUNPOD_ENV_FILE" --env-file "$R2_ENV_FILE" \
  --out results/hydration-diagnostic-cloud
```

Start the guard before the paid creation can outlive its controller. It must act only on that run's journals, retain ownership after an ambiguous cleanup response, and independently confirm HTTP404 absence before releasing the ledger reservation. No cloud allocation or publication has been performed by the diagnostic branch.
