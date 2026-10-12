# Profile calibration and evidence

A calibrated profile connects a simulated serving configuration to observations from that configuration. It should preserve which values were measured, which were derived, and which remain assumptions. A GPU name or a single throughput headline is insufficient to establish a model's capacity curve, startup distribution, and memory limit.

The example catalog combines several evidence types. It is useful for demonstrating sensitivity and controller behavior, but it does not establish a production capacity guarantee. Read each profile's `evidence` and `source` fields before comparing hardware. Historical observations are summarized below and encoded in [`switchyard-gauntlet/src/calibration.rs`](../crates/switchyard-gauntlet/src/calibration.rs); private raw measurement notes are excluded from the repository.

## Configuration identity

Record model and checkpoint revision, precision and quantization, engine and dependency versions, processor settings, accelerator count and type, active-sequence limit, context limit, adapter mix, input shape, output limits, and cache state. For images, include dimensions or megapixels and the processor's resize/patch limits. A text profile does not become a vision profile by adding the observed image-token count to its prompt.

Client concurrency, active engine sequences, containers, and GPUs are separate counts. For example, 64 clients can saturate an engine that admits 32 sequences, while a managed endpoint can spread those clients over an unknown number of containers. Preserve every known count and label every missing one.

## Evidence categories

| Category | Meaning | Suitable use |
| --- | --- | --- |
| Measured | Raw observations of an identified configuration | Calibrate only the fields the experiment observed |
| Documented | A reported aggregate result whose raw samples are unavailable | A labeled calibration point with restricted scope |
| Boundary-adjusted | An arithmetic estimate correcting a known counting limitation | Sensitivity bounds; preserve the original observation |
| Assumed | An unmeasured coefficient, price, curve, or behavior | Explicit what-if experiments |

These are provenance descriptions, not confidence scores. A profile can have measured startup and assumed throughput at the same time. Keep field-specific distinctions in the text rather than relabeling the entire profile “measured.” A quoted rental price is a separate observation from GPU service performance and does not establish inventory availability.

## Current historical examples

The existing receipt-photo bursts recorded end-to-end fleet output rates and call latency. Their container count was not captured, so rates such as 2,722.6 or 3,591.4 output tokens per second cannot be assigned to one H100. Similar fleet throughput with different reported TTFT also shows that aggregate output alone does not determine client responsiveness. The example vision curves therefore remain assumed.

Older synthetic text observations pinned one serving container or node, used approximately 500 input tokens and exactly 200 output tokens, and saturated 32 engine sequences with up to 64 clients. Their original measurement-window code counted some responses that completed after cutoff. The examples retain conservative boundary-adjusted estimates for matching model/GPU text combinations; they do not claim a corrected rerun.

For a window of `T` seconds, `C` clients, and fixed `O` output tokens per response, a maximum boundary adjustment subtracts `C × O / T` from the reported output rate. It assumes one unfinished response per client was overcounted. Exact completion timestamps are required to repair the raw result. Without them, the adjustment is an estimate rather than proof that every client caused that error.

The documented RTX4090 points are single-GPU aggregate text rates at 32 clients: 3,373 output tokens/second for the 0.8B example and 2,449 for the 2B example. Raw sample counts and duration are unavailable. Calibration adds only that aggregate point. The lower-concurrency capacity curve remains estimated interpolation; per-answer decode medians are not multiplied by client count to invent aggregate measurements.

Default A6000, comparative H100/H200, and vision capacities are assumptions unless a matching observation is attached. Model/GPU startup samples do not change that classification. A measured result from a fleet of three GPUs also does not establish one GPU's token rate when actual output totals were not recorded.

## GPU service calibration

Measure output tokens by completed usage records or by counted delivered streaming tokens, using one declared timing boundary. Record request failures and unfinished work. For a finite burst, divide successful output tokens by the complete burst wall time. For a steady-state window, count only events inside that window and document warm-up, cutoff, and drain behavior. Avoid mixing the two definitions.

Sample multiple active-batch sizes using the same request distribution. Report client TTFT, inter-token latency when measured, complete-response latency, aggregate throughput, successful request counts, and utilization. Fit a curve within those observations. Do not extrapolate from one saturated point into a claimed low-concurrency measurement.

The engine consumes prefill plus output-equivalent work. A measured output rate therefore needs a transparent conversion for the matching workload:

```text
macro work per output token = 1
                           + assumed input_work_per_token × input/output ratio
                           + assumed image work / output tokens
```

For the 500-input/200-output text workload with `input_work_per_token = 0.025`, this factor is 1.0625. The conversion keeps the observed workload's output rate consistent with the engine also paying for input work. It does not independently measure prefill. If output lengths or image shapes change, recalibrate the conversion or label the resulting estimate as an assumption.

Do not use a benchmark's aggregate `in_tok_s` as a separate prompt-processing speed when its denominator includes generation. Do not reuse training-token rates for inference. Likewise, a stream's median decode rate is not an aggregate fleet rate, and subtracting separately calculated queue and TTFT medians does not recover an exact prefill time.

## Startup distributions and hydration

`Startup.restore_samples_s` and `build_samples_s` hold end-to-end duration samples. The engine selects the restore/build path using scenario `build_probability`, then samples uniformly from that path's entries. Entries need not be sorted; the controller calculates nearest-rank quantiles from their sorted values. Equal entries retain their multiplicity, so their frequency affects the discrete empirical distribution.

When a path lacks samples, component totals and jitter provide a fallback prior. Restore jitter does not perturb empirical samples. The fallback build multiplier is eight; it is a simulator assumption, not an inferred build law. The built-in planned controllers use a high startup quantile and other controllers use a median, then mix restore/build estimates according to the configured build probability. This mixture is a heuristic rather than the quantile of a fully combined distribution.

Selected priming experiments observed both restores and snapshot builds. They deliberately discovered or initialized worker types, so their observed build fraction does not estimate the chance that ordinary traffic needs a build. Use separate normal-restore and forced-build scenarios and vary production build probability independently. Small samples also cannot establish rare-event startup tails, correlated allocation delays, or failure probability.

The readiness equation adds `Profile.host` hydration to the sampled baseline. Treat the configured `container_image_bytes` and `model_bytes` as additional transfer sizes beyond what that baseline already covers. Leave them at zero when replaying an unchanged end-to-end startup measurement. Adding a full image and full checkpoint download to a baseline that already included both would double-count the same work.

To study a different host, keep the measured baseline and declare a deliberate additional hydration stress, or replace the baseline with measured non-transfer stages and model its transfer separately. Do not subtract guessed weight-download time from an end-to-end sample and label the result measured. `startup_scale` multiplies baseline startup; it does not multiply the additional byte-transfer term.

`Profile.host` is the controller's published prior. `Scenario.host_reality` can specify a different distribution sampled only by the engine. Cache probabilities and bandwidth samples should come from repeated hosts or be labeled assumptions. Probe noise is configured independently. The current model samples nodes independently and gives each one its own bandwidth; shared-link contention or correlated provider queues are not inferred automatically.

## Memory and adapter calibration

Choose `kv_tokens` from the exact configuration's usable memory after model weights, activation headroom, graphs, adapters, and safety margin. If measured bytes per KV token are available, record the conversion and its assumptions. The simulator reserves full input plus maximum output at admission, regardless of prefix-cache reuse or the actual output length.

Measure adapter cold-load time separately from warmed mixed-adapter throughput. The current historical factor compares a four-adapter workload with a matching base workload after applying the same conservative boundary correction to both. Its use for two, three, or other adapter counts is an unmeasured simplification. Vision-tower adapters, quantization, and different engine versions require their own compatibility and quality checks outside this simulator.

## Calibration workflow

The [cloud observation replay guide](EMPIRICAL_SIMULATION.md) describes the checked-in native cloud dataset, configurable private environment inputs, training selection, optional fitting, and residual reports. Use `calibrate --input configs/calibration/native-cloud-observations.json` to replay it without changing its editable profile parameters.

1. Preserve a sanitized measurement artifact with the exact configuration, timing boundary, request counts, token totals, failures, and replica/GPU count.
2. Add a profile or update only fields supported by that artifact. Retain assumptions for unobserved behavior and identify the observation date.
3. Replay the matching workload. Check completed usage, client latency, aggregate throughput, startup, and rental duration separately.
4. Vary input length, output estimates, concurrency, cache state, startup tails, capacity, and failure rates. Identify conclusions that change when an assumption changes.
5. Compare schedulers across multiple seeds with the same declared priors. Identical configuration and seed give deterministic replay; different action sequences need not receive identical host samples.
6. Validate predictions on a held-out workload before using them for a deployment budget or service promise.

Store the profile catalog, scenario, seed, simulator revision, and results together. Report incomplete jobs alongside achieved rates. A scheduler that completes only the easy part of a burst has not demonstrated the throughput target for the whole job.

The largest current gaps are per-GPU vision curves, independent image/prefill costs, realistic output distributions, long-context admission limits, cold adapter behavior, production startup/failure distributions, and correlated allocation behavior. Sensitivity experiments can expose how those gaps affect a decision, but simulated results cannot close them.
