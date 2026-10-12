# Inference scheduling

The scheduler receives tokenized request plans, observed endpoint state, and allocation outcomes. It returns provisioning, routing, release, and host-observation actions. The live controller and simulator execute those actions independently of the policy. Model preparation and baking remain separate operations.

## Measured serving profiles

`CapacityPrior.measured` holds version 1 `ServingCalibration` records. Each record contains a generic `Profile` and a deployment SHA-256 binding. The live adapter computes the binding from the model repository, pinned revision, weight artifact, GPU name, engine configuration (including precision), hydrated runtime, static caches, learned-cache compatibility, and hydration policy. Provider prices and scheduler settings are excluded so a new quote or policy experiment can reuse compatible measurements.

Configuration validation rejects a stale binding before allocation. Profile preparation applies a record only to its exact model/GPU/region profile ID, checks its memory and concurrency bounds against deployment limits, and preserves the fresh provider price. Other offerings retain explicitly labeled priors. `profiles-used.json` and `profile-bindings.json` record the effective selection. Replacing a runtime or model artifact requires a new measurement binding; simply editing its hash does not establish measured performance.

The checked-in `native-fleet-f32-l4-measured.json` and `native-fleet-f32-4090-measured.json` configurations reuse the public cloud observation dataset. Serving curves are fitted to declared training cases, while startup samples are deduplicated observed allocation-to-ready times. Regional serving differences remain unmeasured. These profiles retain the known mixed-length simulation residuals; a deployment binding establishes compatibility, not predictive accuracy. Their startup totals already include hydration, so the profile's additional modeled hydration bytes are zero. Agent observations still report real transfer progress.

The generic profile embedded in a calibration can also be used directly by the simulator. The offline exporter constructs deployment-bound live configurations from the same dataset:

```sh
cargo run --features live --example export_serving_calibration -- \
  configs/calibration/native-cloud-observations.json \
  configs/runpod/native-fleet-f32-l4-raw-ranges-candidate.json \
  results/measured-fleet.json
```

## Background provisioning and ownership

The live routing loop records a launch intent before starting cache lookup, ticket publication, and provider creation in a background task. Pending intents reserve an instance and quoted hourly cost. The controller also includes their elapsed estimated cost in the run budget. `maximum_pending_allocations` bounds cloud I/O concurrency. Each pending intent contributes forecast capacity but has no routable endpoint ID; requests continue through observed ready hosts.

Provider completion is collected on subsequent controller iterations. An accepted pod is recorded in the ownership journal before endpoint setup, and its actual quote is checked against the fleet budget including other reservations. A definite rejection releases the reservation, records its outcome, and starts the existing cooldown. An ambiguous creation remains in the ownership journal for reconciliation. Cleanup drains outstanding provider operations rather than aborting a possibly paid creation, then reconciles and terminates owned resources. The provider adapter must bound its HTTP operations.

The simulator likewise exposes delayed rejection intents as pending capacity until their completion event. Explicit controller pauses in imported historical observations still represent the measured old controller behavior. Passive state reads and explicit bandwidth probes share bounded background tasks; probe refresh cannot block hot dispatch.

## Contribution-based scaling

The planner estimates when existing compatible hosts will drain known work, accounting for startup and usable capacity. A candidate must become ready before that drain time and must provide at least `minimum_scale_gain_s` of estimated completion improvement. A desired throughput target shortens the preferred planning horizon. If that target is unattainable, a host that arrives afterward can still be useful when it improves the forecast drain time. Learning output lengths preserves the original job target. When no capacity exists, the planner permits one best-effort launch even when its startup exceeds the target window. Later decisions consider that pending capacity before buying another host.

Routing admission still reserves input tokens plus the hard output limit against KV capacity. Output estimates affect planning and placement costs, never memory safety or generation limits. Streamed generated-token counts reduce remaining-work estimates while a call is running; a floor prevents a call that outlives its estimate from disappearing from demand.

## Adaptive feedback

`Scheduler::observe(SchedulerFeedback)` receives delivered usage, failed requests, and provider outcomes. The default implementation does nothing, preserving custom scheduler compatibility. Wrappers around `BuiltinScheduler` should forward this method as well as `decide` if they want adaptive learning. All five policies use contribution-based scaling; only `adaptive` learns estimates and placement penalties. Replacing a policy creates a new learning state, which keeps benchmark policy comparisons independent.

Output estimates use the 90th percentile of completed calls after `minimum_samples` observations. Histories are separated by model, modality, service class, adapter, logarithmic input/output-limit buckets, and whether input caching was requested. Estimates are capped by each request's hard output limit. Each history retains at most `history_limit` values, and output/capacity maps retain at most 1,024 workload buckets.

Serving estimates use completed homogeneous windows that have drained a node. Input work and delivered output are divided by elapsed client time. Summed request durations determine approximate concurrency; these are worker/request-seconds, while the window duration is elapsed time. Queue and network latency make the resulting effective capacity conservative. Failed or mixed-bucket windows are discarded. The scheduler applies the lower quartile capacity ratio with a safety fraction and a bounded multiplier only to comparable work. These observations change scheduling beliefs without changing the simulator's physical service model.

Observed endpoint readiness updates allocation-to-ready startup estimates using the 90th percentile. Allocation successes and rejections update a bounded history with a Laplace-smoothed acceptance probability; slow observed provider responses add a placement penalty. Existing cooldowns remain hard temporary exclusions. These signals are generic; HTTP status interpretation, GPU catalogs, artifact lookup, and credentials belong to adapters.

## Configuration and diagnostics

Existing configurations receive these defaults through the shared `scheduling` block:

```json
{
  "heuristics": {
    "history_limit": 128,
    "minimum_samples": 4,
    "capacity_safety_fraction": 0.9,
    "maximum_capacity_multiplier": 2.0,
    "minimum_scale_gain_s": 1.0,
    "maximum_pending_allocations": 2
  }
}
```

This is a fragment of `scheduling`, not a complete deployment. Validation rejects invalid or unbounded settings. `scheduler-decisions.ndjson` records actions, pending reservations, observed nodes, and controller diagnostics. `allocation-outcomes.ndjson` records provider duration and outcome. Adaptive diagnostics include bounded output/capacity samples, startup samples, and allocation histories. They contain counts and timing rather than prompt or response text.

## Validation

Scheduler regressions cover late capacity, an empty fleet, pending-slot reservations, failure-aware placement, output-history bounds and separation, and measured-profile invalidation. Live HTTP tests delay provider creation while serving hot requests and verify that cleanup accounts for the eventual paid resource. Existing ambiguity, ownership, cache preparation, and token-admission tests remain applicable.

The `scheduler_feedback` example exports a deterministic analytical late-capacity fixture for replaying two executable versions against identical inputs. Cloud-observation replay measures controller changes against recorded workloads, while fuzzing and software benchmarks check invariants and controller overhead. These simulations do not establish new real-cloud latency or repair the known mixed-length service-model accuracy gap.
