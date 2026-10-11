# Live scheduler gauntlet

The October 11, 2026 RunPod experiment completed all 4,218 configured inference calls: 1,406 calls for each of Qwen3.5 0.8B, 2B and 4B. Every model completed ten workloads without failed calls, unfinished work, invalid scheduler actions or completion-budget overruns. The controller exited successfully, and an independent watcher confirmed termination of all six owned pods.

## Configuration and reproduction

The [checked fleet](../configs/runpod/qwen-extents-l40s-strict.json) pins L40S hardware, model revisions, raw weight extents, the serving runtime, distinct model-specific compiler caches, engine settings and probe policy. The [workload definition](../configs/runpod/gauntlet.json) includes cold and hot single calls, a 1,000-call burst, three 100-call policy cases, staggered arrivals, long inputs, underestimated output lengths and serial dependencies. Models run sequentially, with at most two concurrent pods per model. All actual allocations in this run were in US-TX-4.

```sh
cargo build --locked --release --features live
target/release/switchyard live check \
  --config configs/runpod/qwen-extents-l40s-strict.json \
  --env-file /path/to/credentials.env --out results/preflight
target/release/switchyard live gauntlet \
  --config configs/runpod/qwen-extents-l40s-strict.json \
  --env-file /path/to/credentials.env \
  --workloads configs/runpod/gauntlet.json --out results/live-gauntlet
```

The [numeric receipt](../experiments/qualified-probe-live-gauntlet.json) records configuration and binary hashes, source identities, timestamped quotes, every workload summary, all observed cold starts and independent cleanup confirmations. The run used configuration commit `0ff8dc8`; its release controller was compiled from Rust source `828caa2`. The intervening commit changed configuration and documentation. Local credentials and detailed journals remain outside Git. Changing hardware, runtime, engine arguments or caches creates a different experiment.

## Measured inference results

TTFT is time from the scheduled client arrival to first streamed content. Burst latency includes time waiting for admission. Output rate divides completed output tokens by the complete workload interval, including queuing and startup within that interval.

| Model | Cold TTFT | Hot TTFT | Burst output tokens/s | Burst p95 TTFT | Estimated model rent |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen3.5-0.8b | 73.94 s | 210 ms | 1,794 | 25.47 s | $0.0465 |
| qwen3.5-2b | 74.23 s | 225 ms | 1,538 | 23.47 s | $0.0511 |
| qwen3.5-4b | 103.49 s | 213 ms | 1,029 | 41.84 s | $0.0788 |

The full experiment took 443.88 seconds and estimated $0.1764 of compute and disk rental. GPU quotes were $1.09/hour; the configured 80 GB container disk added $0.01096/hour. These are ownership-interval estimates using the provider quotes and published disk tariff, rather than reconciled invoice charges.

The 2,000-output-token/s burst objective was missed by all three models. Cold calls, hot single calls and serial dependencies also missed their 200-token/s objectives. These are measured scheduling outcomes even though every call completed within its time budget. Three policy cases share a warmed endpoint and fixed execution order; one run does not establish a policy ranking, model quality comparison or isolated GPU capacity curve.

## Cold-start phases

Durations below use the agent's monotonic epoch. Runtime hydration and model hydration overlap and must not be added. Engine startup follows the runtime gate; tensor assignment follows the first tensor request. The final interval includes remaining engine setup and native acceptance. Allocation, bootstrap and client dispatch use separate controller milestones and are included in cold-request TTFT above.

| Model / allocated pod order | Runtime hydration, s | Model hydration, s | Engine spawn to first tensor, s | Tensor assignment, s | After tensors to verified, s | Last observed phase |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3.5-0.8b / 1 | 26.25 | 5.31 | 26.34 | 2.48 | 8.07 | ready |
| qwen3.5-0.8b / 2 | 26.24 | 4.54 | unobserved | unobserved | unobserved | hydrating |
| qwen3.5-2b / 1 | 26.89 | 11.38 | 26.92 | 4.37 | 8.08 | ready |
| qwen3.5-2b / 2 | 26.75 | 12.64 | unobserved | unobserved | unobserved | hydrating |
| qwen3.5-4b / 1 | 50.10 | 40.50 | 26.12 | 6.76 | 9.07 | ready |
| qwen3.5-4b / 2 | 30.12 | 19.60 | 26.82 | 8.68 | unobserved | initializing |

Second pods received no proposed routing actions and had no observed verified readiness before cleanup. The workload drained on the existing endpoint while these pods started. Their cost remains included in the results. This exposes a scheduling limitation: elastic acquisition should account for whether a new endpoint can become useful before the remaining queue drains. A low advertised hourly cost alone does not establish that acquisition improves a finite workload.

The unpatched production runtime still takes approximately 26–27 seconds from engine spawn to first tensor on these hosts. The [local process-startup experiment](vllm-process-startup.md) validates a guarded import fix under a different GPU and serving fixture; that patch is not enabled in this fleet. These timings therefore do not claim the local improvement has reached production.

## Observation overhead

Every state fetch records HTTP start and finish times. Summed state-fetch durations were 12.53, 16.16 and 24.30 seconds for the 0.8B, 2B and 4B epochs; median fetch durations were approximately 76, 81 and 78 ms. The live loop awaits these observations before its next scheduling decision, while existing inference tasks continue. The sums span complete epochs and are not equivalent to added latency for any individual request. Explicit probe fetches are recorded separately. The journals support measuring dispatch gaps before changing observation concurrency.
