# Native cloud simulation comparison

The configured simulator completed all 33 workloads covering Qwen3.5 0.8B and 2B on L4, and 4B on RTX 4090. Three replay seeds completed 13,230 requests with zero invalid scheduler decisions. The measured cloud request sizes, startup samples, capacity traces, and controller pauses are stored in the [editable dataset](../configs/calibration/native-cloud-observations.json).

The 20% held-out throughput and TTFT p95 gate failed on 10 of 24 cases per seed. The remaining 14 cases passed that gate, including all three 1,000-call bursts. The full [receipt](receipts/native-cloud-simulation.json) preserves p50/p95/p99 latency, elapsed time, throughput, rental cost, allocations, rejections, configuration identity, and failures.

Fitting used hot single-call, reactive, and planned-hybrid workloads only. The comparison below uses those saved parameters without refitting. Environmental traces remain in-sample. Sequential startup samples produced identical results for seeds 7, 8, and 9; these repetitions do not establish confidence intervals for future cloud behavior. See [configuration and methodology](EMPIRICAL_SIMULATION.md).

## Burst throughput and latency

| Model / GPU | Cloud tok/s | Simulated tok/s | Throughput error | Cloud p95 TTFT, s | Simulated p95 TTFT, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B / L4 | 134.8 | 139.1 | +3.2% | 182.52 | 176.97 |
| 2B / L4 | 112.1 | 114.1 | +1.8% | 265.81 | 265.97 |
| 4B / RTX 4090 | 160.1 | 185.4 | +15.8% | 190.22 | 165.64 |

All bursts completed 1,000 requests. TTFT includes waiting from workload arrival, rather than starting its clock when a request reaches the selected endpoint. Throughput divides completed output tokens by the entire burst duration.

## Workload residuals

Each row shows seed 7. Training rows are identified explicitly; only held-out rows participate in the accuracy gate.

| Model | Workload | Training | Cloud tok/s | Sim tok/s | TPS error | Cloud p95, s | Sim p95, s | p95 error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B | cold-single | no | 0.3 | 0.3 | +1.0% | 101.62 | 100.55 | -1.1% |
| 0.8B | hot-single | yes | 20.5 | 21.1 | +3.1% | 0.77 | 0.78 | +0.3% |
| 0.8B | burst-1000 | no | 134.8 | 139.1 | +3.2% | 182.52 | 176.97 | -3.0% |
| 0.8B | hot-cheapest_elastic | no | 121.5 | 129.5 | +6.6% | 20.23 | 18.06 | -10.7% |
| 0.8B | hot-reactive | yes | 133.6 | 134.3 | +0.5% | 18.28 | 17.36 | -5.0% |
| 0.8B | hot-planned_hybrid | yes | 137.0 | 125.3 | -8.5% | 17.12 | 18.76 | +9.5% |
| 0.8B | staggered | no | 125.4 | 135.8 | +8.3% | 3.76 | 3.16 | -16.0% |
| 0.8B | mixed-length-burst | no | 71.0 | 90.4 | +27.4% | 15.70 | 11.82 | -24.7% |
| 0.8B | shared-system-prefix | no | 109.0 | 92.9 | -14.8% | 6.97 | 9.14 | +31.1% |
| 0.8B | output-underestimate | no | 131.4 | 152.3 | +15.9% | 6.90 | 5.85 | -15.3% |
| 0.8B | dependencies | no | 24.4 | 19.8 | -18.7% | 9.95 | 12.18 | +22.4% |
| 2B | cold-single | no | 0.7 | 0.8 | +3.4% | 37.00 | 36.60 | -1.1% |
| 2B | hot-single | yes | 13.2 | 12.4 | -5.5% | 1.30 | 1.27 | -2.0% |
| 2B | burst-1000 | no | 112.1 | 114.1 | +1.8% | 265.81 | 265.97 | +0.1% |
| 2B | hot-cheapest_elastic | no | 106.3 | 114.6 | +7.8% | 25.50 | 23.13 | -9.3% |
| 2B | hot-reactive | yes | 102.9 | 103.8 | +0.9% | 26.55 | 26.23 | -1.2% |
| 2B | hot-planned_hybrid | yes | 108.7 | 112.6 | +3.5% | 24.07 | 22.93 | -4.8% |
| 2B | staggered | no | 84.9 | 112.0 | +31.9% | 5.77 | 5.52 | -4.4% |
| 2B | mixed-length-burst | no | 76.1 | 77.3 | +1.5% | 13.68 | 13.03 | -4.8% |
| 2B | shared-system-prefix | no | 69.5 | 60.1 | -13.5% | 8.37 | 11.17 | +33.4% |
| 2B | output-underestimate | no | 95.6 | 152.6 | +59.6% | 7.73 | 4.99 | -35.5% |
| 2B | dependencies | no | 12.6 | 11.0 | -12.5% | 18.38 | 21.57 | +17.4% |
| 4B | cold-single | no | 0.6 | 0.6 | -1.1% | 54.69 | 55.62 | +1.7% |
| 4B | hot-single | yes | 26.4 | 25.7 | -2.7% | 0.52 | 0.53 | +0.7% |
| 4B | burst-1000 | no | 160.1 | 185.4 | +15.8% | 190.22 | 165.64 | -12.9% |
| 4B | hot-cheapest_elastic | no | 189.7 | 246.9 | +30.2% | 14.87 | 10.22 | -31.3% |
| 4B | hot-reactive | yes | 229.0 | 246.9 | +7.8% | 11.21 | 10.22 | -8.8% |
| 4B | hot-planned_hybrid | yes | 227.3 | 227.6 | +0.1% | 11.17 | 11.32 | +1.3% |
| 4B | staggered | no | 184.1 | 199.8 | +8.5% | 1.61 | 1.02 | -37.0% |
| 4B | mixed-length-burst | no | 73.4 | 172.7 | +135.2% | 14.46 | 5.48 | -62.1% |
| 4B | shared-system-prefix | no | 148.7 | 148.6 | -0.1% | 4.75 | 4.75 | +0.0% |
| 4B | output-underestimate | no | 202.4 | 267.7 | +32.3% | 3.47 | 2.61 | -24.8% |
| 4B | dependencies | no | 25.8 | 24.7 | -4.2% | 9.04 | 9.63 | +6.5% |

## Remaining physical limits

Mixed-length 4B is the largest observed error: simulated throughput is 172.7 tok/s versus 73.4 tok/s in the cloud, and simulated p95 TTFT is 5.48 seconds versus 14.46 seconds. The fluid capacity model does not represent native prefill and decode cohort behavior or evolving prefix-cache occupancy. These are concrete modeling gaps; this comparison does not isolate which one caused each cloud discrepancy.

Underestimated outputs, repeated prefixes, and staggered requests also expose errors. Matching one burst therefore does not establish accuracy across request distributions. Capacity timelines and first-use penalties were reconstructed with the uncertainty described in the methodology, so future inventory and cold-start tails remain unvalidated.

## Verification

The 15 targeted empirical/observability tests passed, including training isolation, partial-run throughput accounting, hidden startup samples, zero-scale startup, capacity rejection accounting, and controller pauses. Clippy passed for the CLI, simulator, and gauntlet with warnings denied. The existing generative gauntlet passed 5,000 simulations across five policies. An HTTP check confirmed that the observations file produces a 429 allocation rejection without a new instance, permits allocation after configured capacity returns, and reaches readiness at the configured time without publishing private startup samples.

The replay used source commit `4b6169233d94515edb58b40723bfeee97e8bea21` with a clean working tree. Executable SHA-256: `80d28f68853c33efd6572098f864a8b7b2925b628f7a8d4c9b42c73d8cd9a7b6`. Detailed run files remain under `results/native-cloud-calibration-02`; the portable receipt is checked in.
