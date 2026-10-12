# Scheduler feedback validation

This comparison evaluates contribution-based scaling, bounded background provisioning, measured deployment profiles, adaptive output and capacity estimates, and availability-aware placement. The current executable was built from commit `8ce989588afe4614e7b63891b749401cfbc80e58` with a clean source tree. The baseline executable matches the SHA-256 recorded in the earlier cloud-simulation receipt. Both executables replayed the same observation dataset and existing calibrated fields. The updated serialization adds default heuristic settings, so the calibrated-file hashes differ. Full identities, hashes, and measurements are in the [validation receipt](receipts/scheduler-feedback-validation.json).

## Recorded cloud workloads

The replay contains 33 recorded workload cases across three seeds. These are offline simulations of existing observations; this validation did not allocate new cloud resources.

| Metric across all three seeds | Baseline | Updated scheduler |
| --- | ---: | ---: |
| Completed calls | 13,230 | 13,230 |
| Pod launches | 18 | 15 |
| Capacity rejections | 165 | 114 |
| Invalid decisions | 0 | 0 |
| Held-out replays exceeding 20% throughput or p95 TTFT error | 30 / 72 | 30 / 72 |

The scheduler buys less unused capacity and encounters fewer allocation rejections in these scenarios. The service-model accuracy gate still fails 10 of 24 unique held-out cases. Mixed-length performance remains inaccurately represented, so these results do not establish reliable predictions for every production workload. Repeating seeds checks reproducibility rather than providing independent cloud samples.

## Late-capacity fixture

The analytical fixture starts with one hot node and a 32-call burst. Additional nodes would take 45 seconds to start, after the hot node has completed the work. The desired throughput is deliberately unattainable; it remains a soft planning target.

| Metric | Baseline | Updated scheduler |
| --- | ---: | ---: |
| Completed calls | 32 | 32 |
| Additional launches | 7 | 0 |
| Burst completion time | 10.364 s | 10.364 s |
| p95 first-token latency | 7.884 s | 7.884 s |
| Modeled fleet cost | $0.12597 | $0.02778 |

The cost includes the full modeled fleet horizon, including idle time; it is not a cloud bill. This fixture demonstrates that avoiding late capacity can reduce spending without delaying completion.

## Correctness and controller overhead

All 140 executed tests in the core, scheduler, live adapter, and gauntlet suites pass. Clippy passes for all selected targets with warnings denied. The fuzz run covers 1,000 generated cases across five policies, totaling 5,000 simulations with zero failures. Live HTTP tests exercise delayed provider allocation while existing endpoints continue serving and verify ownership during cleanup. Two environment-dependent tests remain ignored.

The software benchmark runs each policy three times with 100 and 1,000 requests. For the 1,000-request adaptive scenario, median whole-simulator execution increased from 23.043 ms to 34.738 ms, approximately 51%. This is measurable bookkeeping overhead, not an inference throughput measurement. Other policies increased by approximately 13–16% at that size. These short runs do not isolate scheduler CPU time or establish a production latency distribution.

Raw logs and replay outputs remain under the ignored `results/scheduler-feedback-validation/` directory. The checked-in receipt contains the portable measurements without private credentials or provider tickets.
