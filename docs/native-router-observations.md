# Native router state observations

The live router reads each agent's `/state` endpoint to discover hydration progress and serving availability. Passive reads run as independently owned tasks, with one outstanding read per live node. The request loop consumes completed observations between dispatch iterations and never waits for their HTTP responses. The observations retain request start and finish timestamps, transport counters, startup milestones and visible failures.

A task ID maps each pending read to its stable node index. Completed or failed tasks remove that entry before their result is applied. Results for terminated nodes are discarded. Cleanup cancels and joins every pending read before releasing owned GPU resources. An HTTP observation failure during startup still enforces the startup deadline; a transient failure after a node became ready preserves its last observed state. Actual agent failure responses and inference errors remain visible.

Explicit scheduler `ProbeHost` operations retain their existing bounded request path. This change does not alter the shared scheduling contract, hydration permits, GPU request bounds, serving artifacts, or model arithmetic.

## Baseline live gauntlet

The [raw-range L4 gauntlet receipt](receipts/native-raw-range-l4-full-gauntlet.json) records the unchanged eleven-case corpus with 1,470 requested calls per model. The 0.8B and 2B models completed every call and stayed within each completion budget. The 4B model completed its cold and hot singles, then completed 341 of 1,000 burst calls before its 300-second budget expired. Remaining 4B cases were not executed. No inference request failed in that burst; 659 were unfinished. All five pods returned HTTP 404 after cleanup. Estimated total rental was $0.3267.

The old router awaited passive reads serially before dispatching more work. Summing the recorded request/response intervals during each burst gives the time the routing loop spent in those awaits:

| Model | Recorded burst decision span | State reads | Time awaiting state reads | Completed burst calls |
| --- | ---: | ---: | ---: | ---: |
| 0.8B | 190.609 s | 141 | 43.824 s | 1,000 |
| 2B | 278.282 s | 237 | 143.971 s | 1,000 |
| 4B | 298.836 s | 179 | 162.168 s | 341 |

These are elapsed HTTP waits on the old serial routing path, not decoder CPU time or worker-seconds. Inference tasks continued receiving streams during those waits, but the controller could not consume their completions and fill released slots. The numbers establish a dispatch bottleneck; they do not predict a speedup or isolate every cause of the 4B failure. Provider allocations and explicit host probes can still block the controller.

The 4B expansion host also spent 163.202 seconds hydrating model files. Summed file writes took 7.472 worker-seconds, versus 2,544.281 worker-seconds awaiting HTTP bodies. Those sums overlap across ranges and must not be added to the elapsed timeline. The underlying R2 or host-network cause remains unresolved. No transport retry or stall recovery was introduced.

## HTTP contract test

`slow_passive_observation_does_not_block_hot_dispatch_or_cleanup` warms an endpoint through a useful completion, then makes its state response take three seconds. Thirty-two subsequent inference calls must complete within a 1.5-second budget while exactly one read remains outstanding. Cleanup must cancel that read without waiting for the server's delay and terminate only the owned mock pod. The complete live HTTP contract suite passes nineteen tests, including uncertain allocation, rejected scaling, failed hosts, export failures and ownership-safe cleanup.

## Cloud rerun

The [RTX 4090 receipt](receipts/native-raw-4b-4090-background-observation.json) records all 1,470 calls completed across eleven workloads after the passive-observation change. The 1,000-call burst completed in 199.868 seconds at 160.106 output tokens/s, below its requested 1,000 tokens/s target. Every completion budget passed, and all three owned pods independently returned HTTP 404 after cleanup. Two CUDA cache artifacts were published after release. Estimated total rental was $0.1594.

The rerun retained the serving binary, model bytes, twenty hydration permits, request bounds and workload budgets. It used the separately checked RTX 4090 configuration and a new controller executable SHA. The unchanged L4 rerun obtained no pod before it was ended; a fresh capacity query returned no eligible offerings. This is therefore a functional cloud gate for the new observation implementation, not a controlled L4 speedup measurement. Expansion-host model hydration took 14.574 seconds here versus 163.202 seconds in the L4 baseline, which also materially changes the burst's available capacity.

The [fresh 4B cache-hit smoke](receipts/native-raw-4b-4090-cache-hit.json) separately confirms exact-driver cache reuse after the gauntlet, ten useful completions, 29.172 seconds to first content, and provider-confirmed cleanup. It is a startup/cache test rather than another throughput comparison.
