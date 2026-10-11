# Host bandwidth probes

A host probe is a bandwidth measurement over an observed time window. `HostProbe.started_s` and `at_s` identify the window endpoints, `received_bytes` is the new payload received between them, and `volume_bytes_per_s` equals `received_bytes / (at_s - started_s)`. Times use the controller's elapsed-seconds epoch. Image and model cache observations and total hydration bytes accompany the sample for placement estimates.

## Live counter windows

Every successful agent `/state` observation records a counter snapshot. An explicit probe fetches another snapshot and attempts to qualify a window. The sampler chooses the most recent previous snapshot that meets both the minimum elapsed time and minimum new received bytes, within `maximum_window_s`. After accepting a sample, it retains only the endpoint as the next baseline. Successive accepted windows therefore measure new payload without overlap. Counter history has a 256-snapshot bound; unusually frequent polling can reduce the available window and delay qualification.

Model counters use `bytes_downloaded` and `bytes_total`, with model name, revision, and hydration mode identifying the counter source. This received-byte counter can decrease when a model attempt is rolled back. Runtime and cache artifacts use each artifact's `network_bytes_downloaded`, name, ID, and declared transfer total. These artifact wire counters include retries. Verified or extracted artifact bytes do not contribute to measured bandwidth. If an older agent supplies only aggregate artifact counters, the sampler uses the aggregate wire counter; it never adds aggregate and individual counters together.

Two snapshots match only when their ordered counter sources and declared totals agree and every individual received counter is nondecreasing. A source change, total change, rollback in any channel, or backwards timestamp resets the window and invalidates the current live probe. Summing channels first would hide a rollback when another channel advances, so matching occurs before summing deltas. Nonfinite times and empty counter sets also reset the window. Cumulative agent rate fields remain diagnostic telemetry and do not qualify a sample.

Transfer progress used for remaining-byte decisions is separate from bandwidth. In trusted mode, received artifact bytes and completed or cached artifact bytes both indicate progress, capped at the artifact total. Verified mode uses verified artifact progress. A cached model contributes its full declared size even when it receives no new network bytes. Each probe updates this progress and the startup phase before the next scheduler decision, including probes that cannot qualify a bandwidth window.

## Refresh, learning, and cancellation

The adaptive controller requests another probe after `reprobe_interval_s` when a host is hydrating or initializing, has remaining transfer bytes, and has no pending probe. The runtime records attempted time even if the HTTP fetch fails, which limits repeated failed requests. The scheduler does not request a probe in the same decision that releases the host.

A sample is fresh when its endpoint is no later than the controller time, its age is at most `maximum_age_s`, its window and payload satisfy the policy, and its finite positive rate agrees with the payload divided by the window. Adaptive learning accepts a newer fresh endpoint once per host. Repeated ticks and older samples do not add EWMA updates. Previously learned profile estimates remain placement priors after a sample ages out.

Rate-based release requires a fresh sample, positive remaining transfer bytes, a measured rate below `minimum_hydration_bytes_per_s`, and `remaining_bytes / measured_rate > maximum_startup_s`. A completed transfer cannot trigger this rule while its engine initializes. The independent elapsed-startup rule still releases a nonready host once `controller_time - allocated_s > maximum_startup_s`, including hosts with missing or stale bandwidth evidence. The enclosing live reachability and experiment deadlines continue to apply.

The [configuration reference](CONFIGURATION.md) lists policy defaults and validity rules. These controls are generic across model names, providers, and execution backends.

## Controller evidence

`agent-observations.ndjson` records each successful state and its HTTP `start_s`, `finish_s`, and `at_s`, where `at_s` equals completion time. `host-probes.ndjson` records explicit fetch timing, state, counter sources, reset status, and the optional qualified window. Both journals record failed fetch timing with `status: "fetch_failed"`. Their times share the controller epoch. HTTP completion is the counter snapshot time used for sampling; fetch duration remains available to assess polling uncertainty.

`scheduler-decisions.ndjson` preserves the scheduler time, relevant host observations, learned beliefs, and selected actions before those actions execute. Together, these files allow a release to be evaluated against the exact available progress and sample age. Allocation ownership and cleanup remain governed by the existing live journals.

## Simulator measurements

The simulator retains its active synthetic probe model. A probe creates a zero-byte baseline and later records simulated received bytes using the sampled host bandwidth and bounded measurement noise. Its duration is the maximum of `host.probe_duration_s`, `minimum_window_s`, and the time needed for `minimum_received_bytes`, with the payload-duration extension capped at `maximum_window_s`. The same counter-window qualifier decides whether the completed measurement is accepted. A slow host or excessive configured native duration can produce no qualified sample.

Probe completion, pending ownership, and refresh timing remain observable events. The scheduler receives no private bandwidth value or exact future readiness time. Until it has a fresh sample, readiness estimates use published profile priors. Probe noise has a separate random stream, and ending a host prevents a pending result from being published. Active simulator probes do not model competition for transfer capacity; live measurements are passive observations of actual hydration.

## Validation scope

Deterministic tests cover metadata-only payload, mixed model/artifact counters, counter identity and rollback, refreshed samples learned once, stale slow samples, completed and cached transfers, genuinely slow qualified transfer, refresh intervals, and the startup deadline. The manual-clock HTTP tests verify delayed probe completion and refresh. These regressions establish the corrected contract. Earlier 2B releases lack the exact probe and decision evidence needed to identify their cause; this change does not establish a cause for those historical events.
