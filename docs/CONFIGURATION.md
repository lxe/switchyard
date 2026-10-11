# Simulator configuration

Configuration is JSON loaded into Rust types in [`switchyard-core/src/config.rs`](../crates/switchyard-core/src/config.rs). `init` writes a hardware catalog, an HTTP environment, and sixteen workload traces. The catalog describes public hardware priors; a trace or HTTP environment can override physical conditions to test a scheduler with stale priors. Unknown configuration fields are rejected so spelling mistakes do not silently change an experiment.

## Serving artifacts

Live deployments can set `bootstrap.runtime` to hydrate an executable environment
independently of their container image. Its `artifact` contains the immutable
manifest `id` and object `prefix`; `compatibility` lists required manifest labels.
The agent executes the relative `program`, followed by `args` and the managed
model-serving arguments. `environment` values can reference `{runtime_root}` and
`{cache_root}`. The live preparation step checks identities and compatibility
before creating a cloud resource.

Each model can declare `caches` with a `name`, provider catalog `gpu` name,
artifact reference, and additional compatibility requirements. Every cache must
identify the runtime artifact, model revision, complete engine configuration
fingerprint, runtime launch arguments and environment fingerprint, and GPU. When caches are configured, profiles are created only for
GPU offerings with matching cache entries. Changes to context, concurrency,
dtype, or engine arguments require matching cache preparation.

Cache seeds remain verified immutable trees. The serving integration creates
writable copies for engine updates, which preserves the original seed identity.
Artifact format and filesystem rules are documented in
[Serving hydration](HYDRATION.md).

## Hardware profiles

Each profile identifies one model, workload kind (`text` or `vision`), accelerator, and serving configuration. Different quantization, parallelism, adapter settings, or engine settings should have separate IDs and measured capacity curves.

| Fields | Units and meaning |
| --- | --- |
| `id`, `model`, `gpu`, `workload` | Offering identity, exact model identity, accelerator label, and supported workload |
| `usd_per_hour`, `minimum_bill_s` | Rental price and minimum billable allocation duration |
| `max_in_flight`, `kv_tokens` | Serving slots and conservative total input-plus-maximum-output reservation budget |
| `curve` | Ordered concurrency points containing `in_flight` and aggregate `work_units_per_s` |
| `input_work_per_token`, `image_work_per_megapixel` | Input processing cost in output-equivalent work units |
| `network_s`, `network_bytes_per_s` | One-way latency and per-request upload/egress bandwidth |
| `adapter_load_s`, `adapter_cache_slots`, `multi_adapter_factor` | Adapter miss delay, residency slots (`0` supports the base model only), and mixed-adapter capacity multiplier |
| `evidence`, `source` | Calibration status and provenance; these fields are carried into exported profiles |

One output token consumes one work unit. Input work is `input_tokens × (1 − cached_fraction) × input_work_per_token + image_megapixels × image_work_per_megapixel`. The curve is linearly interpolated between configured points and bounded above its last point. GPU-ready requests share aggregate work capacity according to their service weights. Uploaded or adapter-loading requests reserve memory and a slot, but do not contribute to the ready batch's capacity.

`adapter_cache_slots: 0` declares that the serving configuration supports requests without an adapter. A request selecting an adapter requires a profile with positive adapter slots. The scheduler and physical admission use the same compatibility rule, so unsupported adapter requests cannot occupy a slot or add GPU work. In an offline trace, an unsupported request is rejected on arrival; the provider API refuses it during admission.

`network_bytes_per_s` is an independent per-request link approximation. Output uses four synthetic bytes per token; delivery follows the slower of generation and wire transfer, plus one-way latency. This does not model shared NIC contention, packet loss, or network topology. Input token counts do not imply payload bytes: provide `upload_bytes` when that cost matters.

## Startup and host reality

`startup` contains `allocation_s`, `container_s`, `weights_s`, and `engine_s`, plus symmetric fractional `jitter`. Optional `restore_samples_s` and `build_samples_s` are end-to-end startup samples and override the component sum for their respective path. `build_probability` belongs to the experiment environment; selected priming runs do not establish a production build probability. Without build samples, the reference fixture uses eight times the component sum as an explicit fallback assumption.

A profile's `host` supplies a discrete `volume_bandwidth_bytes_per_s` distribution, image/model byte sizes, cache-hit probabilities, `probe_duration_s`, and `measurement_noise`. Every new instance samples a private bandwidth and cache state. Warm instances have loaded caches. Missing image and model bytes add hydration time to the startup baseline.

Host byte sizes represent additional hydration beyond the configured baseline. Keep them zero when an end-to-end startup sample already describes the environment being replayed. To construct a component model of downloads, exclude those downloads from the fixed component times. The configuration cannot detect duplicated calibration costs.

The provider reports booting, hydrating, initializing, ready, or ended phases. Download progress becomes visible after the fixed boot/setup phase, which this approximation places at one quarter of the startup baseline. A probe takes configured virtual time and reports a noisy bandwidth measurement plus discovered cache state. The shared observation permits an unknown image-cache result (`null`); a controller retains its prior when a backend cannot observe that field. Simulator probes report the sampled cache state. The actual future readiness timestamp is not published before readiness. Cancelled hosts do not later publish pending probe results.

## Workload traces

A `Scenario` defines the observation horizon, scheduling tick, initial warm instances, jobs, requests, optional failures, and physical stress factors. Ticks must be between `0.001` and `1` seconds. GPU completions are calculated within ticks; admission and periodic planning use event opportunities. Events at the same instant are applied before one controller decision, so a simultaneous burst is observed as a set of requests.

Jobs have an `announced_s`, an `arrival_s`, a `completion_budget_s`, and a desired aggregate `desired_output_tps`. An announcement permits planning before arrival. A throughput goal of zero disables the rate constraint. Request IDs and job IDs are dense array indices in trace files; dependencies must refer to earlier request IDs. A dependency becomes satisfied after its response reaches the client.

Each request contains model and workload identity, a user-defined service `class`, arrival time, input tokens and bytes, image size, cache estimate, optional adapter, and output planning fields. `planned_output` is an estimate; `max_output` is the hard generation and memory bound. `actual_output` is fixture truth read only by the physical engine. It must be positive and no greater than the hard bound, and does not appear in scheduler views or accepted HTTP inputs. `latency_target_s` measures full response latency; TTFT is reported separately.

`capacity_scale`, `startup_scale`, and optional `host_reality` override the experiment's physical environment. The scheduler continues to receive catalog priors and observations. `faults` identify allocated instance IDs and failure times; failed unpinned streams restart their full work when rerouted. The reference model does not transfer active KV state between hosts.

## HTTP environment and routing controls

`CloudConfig`, exported as `environment.json`, configures initial warm instances, faults, physical stress factors, build and launch-failure probabilities, `tick_s`, `max_instances_per_model`, idle retention, per-instance queue capacity, global router queue capacity, and the default router policy. `request_completion_budget_s` supplies the convenience front door's default completion objective. `--clock-rate` controls virtual time independently of that file.

The generic provider interface does not autonomously provision or release externally managed instances. Sending to the stable client endpoint attaches the bundled routing runtime; its controller can then operate the shared fleet. Use separate server processes when comparing independent controllers whose decisions should not interact.

`Scheduling` controls planning cadence, reactive drain targets, minimum planning windows, reserved warm slots, priority aging, output-estimate floors, slow-start cancellation thresholds, retry cooldown, learning rate, and whether queued requests may overtake a memory-blocked request. Reactive delay and idle retention belong to the enclosing scenario or HTTP environment. The `classes` map defines priority (smaller values first), relative warm-node service weight, permission to use reserved warm slots, and eligibility for planning announced future work. Classes are scheduling controls rather than application roles.

The adaptive controller records each completed cold-host probe once and updates profile bandwidth, hydration bytes, and cache-hit estimates with an exponentially weighted moving average. Its configurable `learning_rate` is the weight of a new observation. These estimates influence later startup forecasts; a slow host can also trigger cancellation and profile cooldown. This is a baseline learning heuristic, and cannot promise optimal placement when a profile contains several very different host populations.

## Live scheduling configuration

Live `DeploymentConfig` and its backend-independent `ExecutionConfig` contain the same `Scheduling` object used by simulator traces and HTTP environments. Deployment projection preserves every scheduling field, and both backends use `validate_scheduling` from `switchyard-core`. Timing controls and service weights must be finite and positive; learning rate must be in `(0, 1]`, and the warm batch capacity fraction must be in `[0, 1]`.

An explicit `scheduling` object uses the shared defaults for omitted fields. Older live files that omit the entire object retain their previous behavior: `maximum_startup_s` equals `limits.startup_timeout_s`, the hydration floor is `5_000_000` bytes/s, and other fields use shared defaults. This compatibility default is resolved during decoding and saved as a complete object in experiment receipts. Adding an empty object opts into shared defaults, including a 180-second controller startup deadline and a 25 MB/s hydration floor.

The controller's `maximum_startup_s` is a cancellation threshold; `limits.startup_timeout_s` is the hard live reachability deadline. Keeping the controller threshold below the hard deadline permits cancellation and another placement attempt. `retry_cooldown_s` controls when the same profile can be tried again; the workload and experiment budgets still bound all attempts. The checked Qwen hydration experiment sets the controller threshold to 600 seconds, a 5 MB/s hydration floor, and a 60-second cooldown, while preserving the 900-second hard startup limit. A retry succeeds only if another placement and its startup fit within the remaining budget.

## Reproducibility and output

Replay requires the same simulator version, catalog, trace, policy, and seed. Startup/host draws, probe noise, and HTTP output sampling use separate random streams. Rejected immediate-admission requests do not consume output samples. Automatic HTTP clock mode follows wall time and concurrent arrival order, so exact replay uses the manual clock and a fixed request order.

JSON results include completion and rejection counts, unfinished work, client TTFT and class response latency, job throughput and target satisfaction, costs, failure/retry waste, useful output still unfinished at cutoff, and controller diagnostics. Warm rent spans the entire configured observation horizon. Fuzz failures save the exact catalog, trace, and run for replay.

Adaptive diagnostics key `host_beliefs` by zero-based catalog index and `probed_hosts` by allocated instance ID. The suite's `profiles-used.json` preserves catalog order, allowing belief entries to be matched to profile IDs and model configurations.
