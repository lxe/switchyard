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

## Cloud experiment configuration

The checked [diagnostic fleet](../configs/runpod/hydration-diagnostic-2b.json) pins the instrumented agent, Ubuntu bootstrap image, runtime artifact, and Qwen3.5-2B revision. It selects one L40S in US-TX-3, uses tmpfs for models and overlay storage for runtime extraction, and retains twenty shared transfers, four 64 KiB channel slots, the 20-second actively awaited network deadline, and cooperative early engine import. `bootstrap.runtime.hydration.retries: 0` disables runtime/cache retries; trusted model streaming has no retries.

The experiment polls authenticated `/state` observations and stops on the `model_hydration_finished` milestone or the first failure. That milestone follows model publication. Reaching the declared received-byte total alone is insufficient because extraction may still be pending. The controller owns a fresh allocation journal, applies a 600-second deadline and $0.50 rental cap, and terminates its owned pod before recording the result. Cleanup is followed by an independent provider read confirming permanent termination or HTTP404 absence.

Query current capacity and prices before running an experiment:

```sh
cargo run --features live -- live discover \
  --config configs/runpod/hydration-diagnostic-2b.json \
  --env-file "$RUNPOD_ENV_FILE" --out results/hydration-diagnostic-quote
```

The existing `live recover` command operates on an exact run's ownership journals after controller interruption. Supply the diagnostic fleet, both credential files, and the directory containing that run's `pending-launches.json` and `owned-pods.json`. A failed agent process does not establish termination of its cloud allocation.

## Cloud replay result

The instrumented L40S/TX3 replay completed all 4,571,330,560 model bytes with zero retries. Runtime hydration took 37.324 seconds; model hydration and publication took 71.801 seconds. Engine import began when runtime hydration finished, while model transfer continued. The controller stopped at model publication and made no inference calls. Estimated rental cost was $0.02462, and an independent provider GET returned HTTP404 after cleanup.

Piece 20 completed all 134,217,728 bytes. It accumulated 28.471 seconds of channel-send waits, while its longest completed body-frame wait was 0.0725 seconds. Across the model, the longest body-frame wait was 12.531 seconds. All recorded transfers conserved received, delivered, and consumed bytes. The [public measurement receipt](../experiments/hydration-diagnostic-l40s-2b.json) records source and artifact identities, milestones, wait measurements, quotes, and cleanup evidence. These pipeline durations include extraction and local backpressure and do not measure network capacity.

This successful replay does not establish the cause of the earlier failure. A future failure can be assessed using the frozen wait boundary, piece offset, queue and decoder state, permit availability, socket queues/timers, and host task, CPU, I/O, and memory evidence together. A causal repair requires evidence connecting the failure to the proposed change and a repeat with retries disabled.
