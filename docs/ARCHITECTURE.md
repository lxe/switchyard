# Switchyard architecture

Switchyard is a Rust toolkit for inference requests served by rented GPU instances. It models provisioning, startup, request admission, shared GPU service, response delivery, failures, and rental cost. A scheduler chooses which instances to provision and which requests to route. The simulator evaluates those choices without acquiring GPUs or running a language model.

The same event engine supports offline trace replay and an incremental provider API. Trace replay compares scheduler policies over a fixed observation window. The provider API lets an independent client manage a virtual fleet and advance its clock. Both use configurable hardware profiles rather than a hard-coded cloud provider.

## Components

| Package | Responsibility |
| --- | --- |
| [`switchyard-core`](../crates/switchyard-core) | Serializable configuration, known inputs, observations, actions, and the `Scheduler` trait |
| [`switchyard-scheduler`](../crates/switchyard-scheduler) | Reference policies, adaptive host beliefs, and routing runtime |
| [`switchyard-sim`](../crates/switchyard-sim) | Physical event engine, instance lifecycle, synthetic serving, delivery, and accounting |
| [`switchyard-gauntlet`](../crates/switchyard-gauntlet) | Workloads, calibration, generated tests, comparisons, and performance measurements |
| [`switchyard-provider`](../crates/switchyard-provider) | Cloud lifecycle contracts, RunPod REST v2, and signed S3 storage |
| [`switchyard-agent`](../crates/switchyard-agent) | Bounded model/runtime hydration, startup observations, and authenticated serving proxy |
| [`switchyard-live`](../crates/switchyard-live) | Token planning, live admission, streamed usage receipts, and owned-resource recovery |
| [`switchyard-engine`](../crates/switchyard-engine) | Native Rust model execution through Candle/CUDA, bounded batching, and streamed text generation |
| [`switchyard-engine-kernels`](../crates/switchyard-engine-kernels) | Validated CUDA launch boundary and embedded paged decode kernel |
| Root `switchyard` package | CLI, library facade, and runnable examples |

The scheduler depends only on the shared contracts. The simulator accepts any implementation of `Scheduler`; its convenience replay and HTTP front door also depend on the reference scheduler package. The gauntlet composes these packages, and its integration tests exercise their boundaries. Workspace dependencies, compiler requirements, and lints are declared in the root manifest.

Profiles describe one serving configuration: a model, workload type, accelerator label, price, startup distribution, network behavior, batch capacity, memory budget, and adapter behavior. A profile does not automatically infer capacity from GPU specifications. Two profiles can describe different engines or settings on the same GPU by using distinct identifiers and provenance.

## Scheduler observations and private state

The simulator separates information available to a controller from information used to generate outcomes. A scheduler receives a `FleetView` and returns actions. It does not receive the complete scenario or the engine's private node state through this interface.

| Information | Controller visibility |
| --- | --- |
| Model, workload, input tokens, image size, upload bytes, adapter, service class | Known request inputs |
| `planned_output`, `max_output`, latency target, dependencies | Declared planning inputs |
| Announced jobs and their future arrivals | Visible after `announced_s` |
| Queue state, admitted work, generated tokens, delivered completion | Runtime observations |
| Profile prices, startup samples, host distributions, capacity curves | Published priors |
| Sampled future output length | Private until generation and delivery reveal it |
| Sampled host bandwidth and cache state | Private until progress or a probe reveals information |
| Exact future readiness time | Private while starting; controller sees an estimate |

In offline traces, `actual_output` is supplied to the physical engine as a fixture. `PlannedRequest` deliberately omits it. The incremental API instead samples output from a bounded multiplier of the estimate and clamps it to the generation limit. Neither method models answer content or accuracy. A caller that closes over a complete trace can bypass the intended information boundary; comparisons must give custom schedulers only the declared interface.

`cached_fraction` describes an assumed reusable prompt fraction for prefill. It is distinct from the host's model-file cache. The former reduces request input work; the latter changes startup hydration. Neither represents an automatically discovered prefix-cache policy.

## Event timing

The event queue contains request arrivals, node readiness, node failures, probe completion, and controller ticks. Events have a timestamp and serial number, making equal-time ordering repeatable. GPU work advances up to each event time. All events at that timestamp are applied before one controller decision, so a simultaneous burst produces one complete observation rather than repeatedly copying its growing queue. A configurable tick supplies recurring planning and admission opportunities; service completions inside an interval are calculated from remaining work rather than rounded up to a full tick.

A request can run after its arrival and after every `depends_on` request has delivered its complete response. Dependencies reference earlier request indices, which makes traces acyclic. Generation completion and response delivery are distinct: per-request egress bandwidth and network delay determine `completed_s`. A dependent request therefore waits for delivered output. Admission of newly runnable work may still wait until the next event or controller tick. The controller sees a `Delivering` state after GPU service has finished, preventing unnecessary replacement capacity for output already in transit.

Warm nodes begin ready at simulated time zero and remain billable over the observation window. Elastic nodes are billable from allocation, including startup and idle time. Failed active requests return to the queue and restart their input and output work. Previously emitted tokens remain in node accounting but do not count as a second successful completion.

## Startup, caches, and probes

Startup first chooses restore or build according to `build_probability`. If that path has empirical samples, it uniformly resamples one observed end-to-end duration. This is a discrete empirical distribution. Otherwise, restore uses the sum of allocation, container, weight, and engine component estimates with symmetric fractional jitter; the build fallback uses eight times that sum. These fallback components are configurable priors, not measured stage boundaries.

For an elastic instance, readiness is calculated as:

```text
ready time = allocation time
           + sampled baseline startup × startup_scale
           + additional uncached host bytes / sampled host bandwidth
```

The host-byte term is an additional hydration perturbation beyond the baseline. If an empirical startup sample already includes downloading the same image and weights, adding their full sizes again counts that hydration twice. Use zero additional bytes for an unchanged measured environment. Supply nonzero bytes only for an explicitly additional transfer or a controlled host/cache stress experiment. Replacing a complete baseline with measured stages is a separate calibration exercise; the current engine does not subtract an unknown hydration component from an end-to-end sample.

`Profile.host` publishes a prior over bandwidth, cache probabilities, additional byte sizes, probe duration, and measurement noise. A scenario can override the sampled reality with `host_reality` while leaving the controller's prior unchanged. This supports experiments with unexpectedly slow hosts or stale assumptions. Independent draws currently approximate each node; provider-wide queue correlation and shared network contention require separate model extensions.

The visible phase sequence is booting, hydrating, initializing, ready, or ended. Hydration progress starts after one quarter of the sampled baseline. These phases are a coarse observation model, not a reconstruction of the configured component timings. A simulator probe becomes available after that setup interval and measures synthetic received bytes with bounded noise. Its duration is extended from `probe_duration_s` to satisfy the shared minimum window and payload controls where possible. Only a qualified completed sample becomes scheduler-visible. Probe duration consumes elapsed rental time; probes currently do not compete for GPU or transfer capacity. Live probes instead measure deltas between observed transfer counters. The [host probe contract](HOST-PROBES.md) defines both paths.

The adaptive built-in controller refreshes probes while starting hosts still have transfer bytes remaining. It updates per-profile bandwidth, uncached-byte, and cache-hit beliefs once per fresh sample using an exponentially weighted moving average with configurable `learning_rate`. Rate-based cancellation requires fresh qualified evidence and observable remaining progress; elapsed startup can independently exceed the configured deadline. A cancellation temporarily avoids the affected profile. Subsequent startup forecasts use learned observations, but cancellation does not establish that another host will be faster.

## Bounded fluid GPU service

The service model expresses input processing and generation in output-equivalent work units:

```text
prefill work = input tokens × (1 − cached_fraction) × input_work_per_token
             + image megapixels × image_work_per_megapixel

request work = prefill work + actual output tokens
```

An active request consumes its prefill work before generating output. These coefficients approximate tokenizer, image-processing, encoder, and prompt-processing costs in one scalar. They are not independently measured input-token rates unless a calibration experiment establishes that interpretation.

`curve` gives aggregate work units per second at selected ready-request counts. Capacity increases linearly from zero to the first point, interpolates between points, and remains at the final point beyond its sampled count. The curve need not be increasing: a measured configuration may become slower under contention. Admission is bounded independently by `max_in_flight` and memory.

Only requests whose upload and adapter waits have finished contribute to the batch's service rate. A batch with more than one distinct adapter applies `multi_adapter_factor`. The available aggregate capacity is divided by positive service weights; each ready request receives its weight divided by the total weight. A request finishing within an interval releases its share for the rest of that interval. Waiting requests never manufacture GPU capacity.

This is a macroscopic continuous-service approximation. It does not reproduce individual decode iterations, chunked-prefill token budgets, kernel overlap, preemption, tensor-parallel communication, or speculative acceptance. Use measured curves for a fixed engine configuration, and add explicit model structure before drawing conclusions about those mechanisms.

## Memory and adapter admission

Every admitted request reserves `input_tokens + max_output` against `kv_tokens`. The reservation uses the hard generation limit, even when the planned or actual output is shorter. `kv_tokens` is a configured budget after weights and other persistent allocations; the engine does not calculate bytes per token or verify model weights against a GPU's advertised memory.

This rule intentionally favors conservative admission. It does not model paged-cache allocation, prefix sharing, growing reservations, cache fragmentation, sliding-window attention, or different KV sizes across layers. A measured token budget must include activation, graph, adapter, and safety headroom for the exact serving configuration. Lower memory admission can reduce throughput even when the service curve predicts spare compute capacity.

`adapter_cache_slots` declares adapter capability as well as residency capacity. A zero-slot profile serves the base model only; a request selecting an adapter is compatible only with a positive-slot profile. Both scheduler planning and physical admission enforce this rule. Unsupported adapter requests add no GPU work or adapter capacity.

Supported adapters occupy a bounded cache with recency ordering. An adapter used by active work cannot be evicted. A miss adds load time before service; multiple requests can wait for the same in-progress load. The model approximates adapter residency and a mixed-adapter penalty, without estimating adapter bytes or detecting a particular engine's adapter support automatically.

## Scheduling and objectives

`Scheduler::decide` can provision a profile, route a queued request with a service weight, release an idle node, or probe a host. The engine validates actions and counts invalid decisions. Provisioning is allowed at planning opportunities, live instances are bounded per model, and routing always passes physical compatibility, readiness, slot, adapter, and memory checks.

`FleetView.provisioning_cooldowns` maps a profile index to the earliest time when another allocation may be attempted after an observed rejection. All built-in policies exclude those profiles from new placement until that time. A cooldown leaves existing hosts available for routing. The observation contains no cloud-specific status or hidden capacity prediction; a runtime translates its provider's confirmed rejection into this scheduling constraint.

The built-in policies provide reference behavior. Warm-only uses the preallocated fleet. Reactive scales after arrived demand and a configured delay. Cheapest-elastic favors rental price per predicted capacity. Planned-hybrid uses announced work, startup quantiles, service-class priority, aging, reserved warm slots, and weighted service on warm nodes. Adaptive adds host probes and slow-start cancellation to the planned policy. They are heuristics, not proven optimizers. Custom controllers can be compared through the same trait or through the provider API without changing service physics.

Fleet throughput, client first-token latency, and rental expense measure different consequences. Larger batches can increase aggregate throughput while giving each stream a smaller service share. Early provisioning can improve client TTFT but buys idle time. Reserving warm slots protects short requests while reducing the capacity available to a batch job. Report all three dimensions rather than declaring a winner from tokens per second alone.

## Results and accounting

Client TTFT is the first delivered token time minus arrival time. It includes queue or dependency delay, cold startup when relevant, request network and upload time, adapter wait, prefill service, the first generated token, and response network delay. Queue delay ends at first admission, so it does not include every later wait or retry. Class latency targets apply to full response latency rather than TTFT.

Node cost is its allocated duration, bounded below by `minimum_bill_s`, multiplied by its hourly rate. Idle cost includes billable time without modeled GPU service, including startup, upload waits, and idle retention. Storage, egress, control-plane fees, discounts, quality, and real cloud availability are outside this calculation.

Accounting distinguishes output discarded by failed attempts (`wasted_output_tokens`) from useful partial or undelivered output at the observation cutoff (`incomplete_output_tokens`). Only client-delivered completions enter successful throughput. TTFT samples also stop at the cutoff; future delivery timestamps cannot enter the reported percentile.

Job throughput divides successful output tokens by the interval from job arrival to its last delivered response, including startup and queues. A job meets its target only when all its requests finish, its completion budget is met, and its requested aggregate throughput is met. The summary's batch rate can describe a partially finished workload; read unfinished counts and per-job results before interpreting it. Empty latency samples remain absent, and class SLO denominators include unfinished arrived requests.

## Relationship to serving-system research

The physical approximations have specific boundaries. [vLLM's optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/) describes how cache size, concurrent sequences, chunked prefill, and startup optimizations affect latency and throughput. Those settings should define separate calibrated profiles; the simulator does not treat changing a token budget as a universal capacity multiplier.

[Sarathi-Serve](https://arxiv.org/abs/2403.02310) studies chunked prefill and batching under latency constraints. It motivates preserving the throughput/latency distinction, but its published improvements are not capacity constants for this simulator. The present fluid engine does not implement its iteration schedule.

[Llumnix](https://arxiv.org/abs/2406.03243) uses runtime rescheduling and request-state migration across instances to address heterogeneous demand and service objectives. The pluggable controller boundary allows related scheduling experiments, but active-request migration and KV transfer are not currently implemented.

[ServerlessLLM](https://arxiv.org/abs/2401.14351) examines checkpoint locality, loading, and startup-aware placement. It motivates treating host cache and transfer behavior as variables independent of GPU identity. The simulator's scalar hydration and probe model does not reproduce its storage hierarchy or live migration.

## Live execution backend

The live runtime supplies the same observable fleet contract to a scheduler while translating its actions into real cloud operations. Model output remains unknown until the selected serving engine emits content and usage. Provider and storage adapters have separate lifecycles, and a pod agent exposes measured hydration and engine state. See [RunPod execution](RUNPOD.md) for configuration, measurement boundaries, and recovery.

## Native inference path

The native backend executes Qwen3.5 text models in a Rust process using Candle tensors and CUDA operations. The fleet scheduler chooses hosts and routes requests; the native engine separately chooses which queued requests execute together on one GPU. Neither the simulator nor the scheduler depends on the model implementation.

An allocation downloads a small static agent, which hydrates independently published runtime, model, and GPU-cache artifacts into private directories. Runtime and cache archives use direct piped installation with shared transfer permits. Native weights are prepared dense safetensors files: hydration installs them on the host filesystem, then the engine reads capped host buffers and copies tensors into GPU memory. Incoming network chunks do not become GPU tensors directly. Cache families contain separately baked compiler seeds for exact driver variants; they do not contain an initialized GPU model.

The engine initializes CUDA and loads weights while a separate thread prepares the tokenizer and chat renderer. Fixed configuration bounds context length, batch size, queued jobs, and host weight bytes. The GPU memory minimum filters eligible hosts; it does not impose a hard allocator ceiling. The HTTP listener opens after its required dependencies finish, without a dummy inference or automatic memory-sizing pass.

Each admitted group owns its recurrent and convolution state while sharing immutable model weights and attention pages. A bounded model-local prefix cache retains exact token checkpoints with all hybrid state and final logits. Equal-length prompt rows deduplicate shared prefixes; varied-length groups join shared decoding through true-length page tables and per-row positions. Releasing rows drops their page handles and compacts private recurrent tensors. New requests wait until the admitted group finishes; continuous admission and a preallocated page slab remain work. See [attention pages and prefix caching](native-prefix-cache.md) for retention bounds and accuracy limitations.

The native backend supports bounded greedy text requests and streamed OpenAI-compatible completion/usage events for the pinned 0.8B, 2B, and 4B models. F32 mathematical gates and BF16 endpoint measurements have different scopes: passing endpoint receipts do not establish strict BF16 numerical parity. See [native fleet execution](native-live-integration.md), [model accuracy](native-engine-accuracy.md), and [RunPod artifact deployment](native-runpod.md) for their configuration and evidence boundaries.

## Native artifact preparation

Native baking is an offline build operation with its own platform requirements and resource owner. Its controller selects a configured build host, observes the actual GPU and driver, runs fixed model/request shapes to prepare compiler caches, exports the results, and confirms host release. It can share provider lifecycle and storage clients with inference deployment; it does not ask an inference scheduler to place requests or acquire bake capacity.

Compiler targets, such as `compute_89`, describe the code being built. A bake host's driver and supported CUDA version describe the platform executing that code. The bake configuration specifies compatible host requirements independently of inference scheduling policy, and receipts retain the observed platform rather than inferring driver compatibility from a cloud pool.

The published runtime, weights, and cache manifests form the boundary between preparation and serving. Inference scheduling consumes those immutable references and their compatibility labels when deploying an endpoint. A missing cache variant requires another offline bake and publication; it does not turn an inference request into a build job.
