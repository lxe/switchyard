# Scheduler validation and workload results

The simulator and its controllers run without GPUs or provider credentials. The checks below establish the behavior of the event engine, observation boundary, HTTP serving interface, and reference policies. Hardware throughput remains a property of the configured profiles; passing the suite does not validate unmeasured GPU capacity.

## Test gauntlet

Run these commands from the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace --all-targets
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps
cargo run --locked --release -- init --out data/current
cargo run --locked --release -- fuzz --cases 1000 --seed 42 --out results/final-fuzz
cargo run --locked --release -- suite --seed 7 --seeds 3 --out results/final-suite
cargo run --locked --release -- bench --sizes 1,100,1000,10000 --repetitions 3
```

The October 10, 2026 local run passed 51 tests, strict Clippy, formatting, and documentation compilation. The generated-workload test exercises 128 seeds across all five policies. A separate release-mode run exercised 1,000 seeds, beginning at 42, across five policies: 5,000 simulations with no invariant failures. This is seeded generative fuzzing; it does not use coverage-guided mutation.

| Check | Behavior established |
| --- | --- |
| Analytical service fixtures | Shared GPU work, exact completion within a tick, adapter waits, and mixed-adapter penalties |
| Admission and accounting | Hard output-bound KV reservation, slot limits, minimum billing, retries, and request/token conservation |
| Observation fixtures | Unknown output and startup remain private; unannounced work stays absent; probes and rejected traffic do not perturb future draws |
| Scheduling regressions | Pending capacity prevents duplicate provisioning, memory-fitting requests can overtake a blocked head, and observed hydration changes future profile choices |
| HTTP contract | Atomic admission, bounded queues and bodies, error envelopes, lifecycle/probe timing, JSON and SSE, and manual-clock shutdown |
| Generated workloads | One to four models, text and vision inputs, adapters, dependency chains, inaccurate output estimates, failures, and uncertain hydration |
| Software benchmark | Every measured run drains its workload and passes accounting and admission invariants |

HTTP was also exercised over a real loopback TCP listener in manual-clock mode. The Rust example completed 100 requests through the stable routing endpoint and 100 through the provider API, including provisioning, readiness polling, probing, admission, usage collection, and release. A separate streaming request emitted content, final usage, and `[DONE]`, with receipt usage matching the 20 delivered synthetic tokens. These are transport and serving-contract checks against the simulator.

The Switchyard workspace refactor was revalidated on October 10: all 51 tests passed, along with strict workspace Clippy and rustdoc. The 1,000-seed corpus again passed all 5,000 simulations. One repetition per policy and size passed a two-second software execution budget, with the slowest 10,000-request run taking 1.107 seconds on the local host. This is a software timing receipt, not GPU performance evidence.

GitHub Actions runs the local checks and a smaller generated/performance corpus. The workflow has been added but has not been executed on GitHub.

## Policy comparisons

The comparison uses the sixteen default scenarios, five policies, and seeds 7, 8, and 9. Values below are medians across those three seeds. Costs include each scenario's entire observation window, including baseline rental, startup, minimum billing, and idle retention. Client TTFT includes queueing and delivery; it is distinct from a stream's decode rate.

All 240 runs were checked for request conservation, completed-output conservation, slot and memory bounds, nonnegative cost, per-instance billing sums, valid client timing, and zero invalid controller actions. An objective miss remains a reported outcome rather than an invariant failure.

The mixed workload contains 1,200 burst requests, 180 short calls, and 300 dependent calls. The burst requests target 3,000 aggregate output tokens per second within a 180-second completion budget. Every policy finishes all 1,680 requests in these runs.

| Policy | Burst output tokens/s | Client TTFT p95, seconds | Total rental USD | Burst target met |
| --- | ---: | ---: | ---: | --- |
| Warm only | 662 | 567.84 | 0.2950 | No |
| Cheapest elastic | 2,378 | 156.76 | 0.5266 | No |
| Planned hybrid | 4,040 | 87.68 | 0.7348 | Yes |
| Adaptive | 4,040 | 87.68 | 0.7348 | Yes |
| Reactive | 7,053 | 50.40 | 1.7474 | Yes |

Planned hybrid spends about 58% less than reactive scaling while meeting this workload's burst objective. Reactive buys more capacity and delivers lower TTFT. Cheapest elastic spends less but misses the desired burst throughput. These results make the objective tradeoff visible; none establishes a universal winning policy.

For the 72-request workload targeting 200 output tokens per second, the existing warm capacity suffices. All policies meet the target without elastic launches, at about 743–752 burst output tokens per second. With five simultaneous 1,000-request model workloads, planned hybrid completes all 5,000 requests, while warm-only leaves 928 unfinished at the cutoff. Planned hybrid still meets only four of the five per-model throughput objectives; the fleet's aggregate rate cannot establish that each model meets its target.

The 1,000-call burst targets 2,000 output tokens per second. Planned hybrid achieves about 3,117 at USD 0.4091, while cheapest elastic achieves about 1,896 at USD 0.4400 and misses the target. Startup, memory-fitting batches, and idle billing can make a cheaper advertised capacity less cost-effective for a finite job.

The unknown-hydration scenario samples additional uncached transfers over hidden 5 or 250 MB/s volume links. The controller still begins with the original profile priors. Adaptive achieves a median 3,756 burst output tokens per second at USD 0.9694, compared with planned hybrid's 2,394 at USD 1.8785. That is about 57% more throughput and 48% less rental cost in this stress fixture. Adaptive meets the burst target in two of three seeds; planned hybrid meets it in none. The diagnostics record probe observations, learned beliefs, and cancelled starts. Three seeds demonstrate the mechanism but do not establish tail reliability.

## Software performance

The `bench` command measures full simulator and scheduler execution with an analytical warm-host workload. Its capacities are synthetic, and its wall-clock rate measures software overhead. It separately records the virtual output rate produced by those capacities.

The local release build ran on an Intel Core i7-13700 with three repetitions per size and policy. Every 10,000-request run completed within the optional two-second budget:

| Policy | Wall time p50, ms | Wall time p95, ms |
| --- | ---: | ---: |
| Warm only | 1,029 | 1,035 |
| Reactive | 692 | 702 |
| Cheapest elastic | 1,092 | 1,100 |
| Planned hybrid | 1,114 | 1,122 |
| Adaptive | 1,111 | 1,118 |

Equal-time event coalescing, cached workload totals and active progress, and available-pool checks keep simultaneous bursts from repeating unnecessary queue copies and admission scans. The values include controller work and result construction, while JSON file writing and invariant checks happen outside the timed interval.

Use `--max-seconds` to enforce a regression budget calibrated to the machine running the benchmark. Timing assertions in ordinary CI deliberately avoid a universal CPU-speed threshold. Profile assumptions, clock resolution, request count, and controller complexity can all affect execution cost.

## Interpreting and extending the results

Use `summaries.json` for comparisons and individual run JSON for request timings, per-instance cost, profile selection, fleet timelines, and controller diagnostics. Check unfinished counts and each job's `target_met` before relying on aggregate output rates. Fuzz failures save profiles, scenario, and run data for deterministic replay.

The reference controller incorporates workload planning, memory-limited batch capacity, just-in-time provisioning, pending-capacity accounting, service-class priority and aging, and optional host learning. Further experiments can replace it through `Scheduler`, or use a separate HTTP client against the provider API. Change scheduling decisions while preserving the same physical profiles and workload seeds.

The default catalog includes boundary-adjusted and documented historical text measurements alongside assumed A6000, comparative high-end GPU, and vision capacities. Its sample startup distributions are small and do not establish rare-event tails. [Calibration](CALIBRATION.md) describes how to replace those assumptions with measurements of the intended serving configuration.
