# Configurable cloud observations

The simulator accepts private environment observations alongside public hardware profiles and request traces. These inputs describe capacity availability, paired startup phases, and intervals during which the controller cannot dispatch. They work in offline replay and the HTTP provider. The scheduler learns about the environment through allocation outcomes and host observations; it cannot read future capacity changes or selected startup samples.

## Configuration fields

Pass an observations file with `run --observations FILE` or `serve --observations FILE`. The [example](../configs/calibration/environment.example.json) uses the analytical quickstart profiles. Its values are assumptions, not cloud measurements.

```sh
cargo run --locked --release -- run \
  --profiles configs/quickstart/profiles.json \
  --scenario configs/quickstart/burst.json \
  --observations configs/calibration/environment.example.json \
  --policy adaptive --out results/observed-quickstart.json
```

| Field | Meaning |
| --- | --- |
| `profiles` | Map from a public profile ID to its private environment configuration. Missing profiles retain ordinary simulated physics. |
| `capacity` | Ordered `{at_s, instances}` entries specifying the maximum live instances of that profile. The last entry at or before the current time applies; before the first entry, capacity is unrestricted. Include an entry at zero to bound initial availability. |
| `startups` | Paired startup samples, selected for each successful cold allocation. An empty list retains the existing profile startup and host-cache model. |
| `sampling` | `sequential` cycles through paired samples in order; `bootstrap` samples with replacement using the run's seeded generator. Default: sequential. |
| `rejection_delay_s` | Controller blocking time for an unsuccessful allocation. Offline routing then observes the scenario's `scheduling.retry_cooldown_s`. Default: zero. |
| `controller_pauses` | Ordered, disjoint `{start_s, end_s}` intervals during which the bundled controller cannot dispatch or provision. Already admitted GPU work and token delivery continue. An external HTTP controller controls its own dispatch timing. |

Capacity limits count both starting and ready instances. A reduction prevents additional allocations; it does not evict existing instances. A rejected allocation creates no instance and incurs no rental charge. It increments `summary.capacity_rejections` rather than `invalid_decisions`. Provider quotas and per-model limits still apply independently.

Each startup sample contains the following fields, all measured from successful allocation unless stated otherwise:

| Field | Meaning |
| --- | --- |
| `boot_s` | Duration before the simulated hydration phase begins. |
| `hydration_s` | Hydration duration. |
| `hydration_bytes` | Bytes represented by the hydration progress counter. The simulated transfer rate is bytes divided by hydration duration. |
| `ready_s` | Total allocation-to-endpoint-ready duration, including boot and hydration. Must be at least `boot_s + hydration_s`. |
| `first_use_s` | Additional delay triggered once by the first admitted inference request. Requests admitted during that delay share the same gate. |

Samples replace the cold host's normal startup and additional hydration calculation, avoiding duplicate transfer costs. `startup_scale` scales boot, hydration, and readiness durations; first-use delay remains independently configured. Nodes listed in `warm` are already initialized and bypass these cold-start samples and first-use gates. Boot, hydration, readiness, and first-use values stay together when sampled so their correlations are preserved.

For serving performance, edit each public profile's `curve` (aggregate work units per second by active request count), `input_work_per_token`, `network_s`, `network_bytes_per_s`, `max_in_flight`, and `kv_tokens`. Existing host cache and bandwidth distributions remain available when paired samples are absent. Request `cached_fraction`, input lengths, planned outputs, hard output limits, and private actual output lengths belong in the scenario. See [configuration](CONFIGURATION.md) for those fields. No provider-specific concepts are required by the physical engine.

Unknown fields, unknown profile IDs, negative or nonfinite times, unordered capacity changes, overlapping pauses, and impossible startup samples are rejected before execution.

## Replay and fitting

An empirical dataset contains `provenance`, `profiles`, and `cases`. Each case holds its `model`, `policy`, `training` flag, `scenario`, private `observations`, and reference `measured` metrics. The [native cloud dataset](../configs/calibration/native-cloud-observations.json) contains 33 workloads and 4,410 requests covering Qwen3.5 0.8B and 2B on L4, and 4B on RTX 4090. It includes observed token lengths and cold-start samples, quoted costs, allocation availability traces, and recorded controller pauses.

```sh
cargo run --locked --release -- calibrate \
  --input configs/calibration/native-cloud-observations.json \
  --out results/cloud-replay --seeds 3
```

Replay preserves edited parameters. Add `--fit` to estimate four effective service parameters: the single-request and maximum-batch capacity points, input-work coefficient, and network latency. Fitting pools profiles for the same model and uses only cases marked `training`. The checked-in dataset marks hot single-call, reactive, and planned-hybrid cases as training; the remaining eight cases per model are held-out serving checks. These fitted values describe end-to-end effective service, not isolated GPU kernel throughput or regional performance differences.

The command prints one concise summary per case and saves:

- `calibrated-input.json`: the complete runnable configuration, including edits or fitted parameters.
- `comparison.json`: measured and simulated completions, elapsed time, output throughput, TTFT p50/p95/p99, rental cost, relative errors, allocations, rejections, and invalid controller decisions.
- `identity.json`: input and executable SHA-256, source revision and dirty status, seeds, fitting selection, tolerance, and saved configuration SHA-256.

Add `--max-relative-error 0.2` to require held-out throughput and TTFT p95 errors within 20%, matching completion counts, and zero invalid decisions. A failing gate returns an error after saving the reports. Reference timing values do not drive the simulation toward known completion timestamps. Actual output lengths are private engine inputs; the scheduler sees estimated output lengths and hard bounds.

Throughput counts output tokens from completed requests divided by complete workload elapsed time. A partial run uses the full simulation horizon as its denominator. TTFT is arrival-to-first-token latency with nearest-rank percentiles. Rental cost covers each case through its completion boundary, including configured minimum billing; it excludes capture and cleanup work between cloud cases.

## Cloud evidence and limitations

The offline [receipt importer](../examples/import_native_observations.rs) translates native live-run files into the generic format. It uses no cloud credentials and lives outside the simulator and scheduler crates:

```sh
cargo run --locked --example import_native_observations -- \
  results/imported-cloud-input.json \
  results/native-raw-full-gauntlet-02 \
  results/native-raw-4b-4090-background-observation-01
```

The importer selects completed model suites. It obtains request sizes and outputs from completed usage, initial warm fleets from host observations, startup phases from agent telemetry, and pauses from synchronous state-observation calls. First-use delay is estimated from the first batch's prefill duration above later warm prefill duration. Hosts without a complete ready sample use pooled samples; these are estimates rather than observations of that particular host. Allocation rejection delay defaults to an assumed one second and remains editable.

Capacity availability is interval-censored: a rejection establishes unavailability at one instant, while a later successful allocation establishes availability later. The imported capacity trace uses the lower bound immediately after the last preceding rejection. It does not establish when inventory actually returned or a probability distribution for future availability.

Serving validation cases are held out from fitting, but their environmental traces remain in-sample. This replay tests whether the generic physics reproduces known workloads under recorded conditions; it does not independently validate future cloud startup distributions. The default sequential samples produce identical results across seeds when no other randomness affects execution. Such repetitions demonstrate determinism, not confidence intervals. Use bootstrap sampling and additional independently collected observations for sensitivity studies.

The physical engine shares aggregate GPU work among active requests. It does not reproduce native decode cohorts, prefill scheduling, or evolving shared-prefix cache occupancy. Residuals on mixed-length and underestimated-output workloads must therefore remain visible. Do not replace capacity curves with per-case measured output rates to conceal those gaps.

The [native cloud comparison](native-cloud-simulation.md) records all workload residuals and the failing 20% accuracy gate. It includes a portable receipt with executable/configuration hashes and complete latency distributions.
