# Shared GPU hosts

A host plan describes one physical GPU allocation and the model endpoints that reside on it. An endpoint has a model profile, a fixed GPU memory reservation and its own serving process. The scheduler routes requests to endpoints, while the provider adapter allocates and releases the physical host once.

This supports several models on one GPU and several replicas of one model. The Rust native engine still serves one base model per process. Each endpoint has independent batching, queues, prefix history and adapter state; different base models cannot enter the same tensor batch.

## Capacity configuration

`topology.host_plans` is available in simulation scenarios, simulator HTTP environments and live deployment configuration. An empty topology preserves one endpoint per allocation. The checked example in [configs/shared-hosts](../configs/shared-hosts) places two replicas of `model-a` and one endpoint for `model-b` on a hypothetical 24 GiB GPU.

```json
{
  "topology": {
    "host_plans": [{
      "id": "resident-pair",
      "gpu": "virtual-24g",
      "gpu_memory_bytes": 25769803776,
      "reserved_gpu_memory_bytes": 4294967296,
      "contention_factor": 0.8,
      "endpoints": [
        {"id": "a-0", "profile": "model-a-shared", "gpu_memory_bytes": 6442450944, "compute_fraction": 0.3},
        {"id": "a-1", "profile": "model-a-shared", "gpu_memory_bytes": 6442450944, "compute_fraction": 0.3},
        {"id": "b-0", "profile": "model-b-shared", "gpu_memory_bytes": 8589934592, "compute_fraction": 0.4}
      ]
    }]
  }
}
```

Endpoint memory reservations include GPU weights, KV and prefix caches, adapters, and temporary workspace. Their sum plus the host margin must fit the declared device memory. Replicas have separate GPU weight allocations, even when their hydrated model files are shared. Admission uses configured reservations; it does not discover an automatic KV size or impose a CUDA allocator limit.

`compute_fraction` reserves a fraction of the endpoint profile's solo service capacity. The simulator and scheduler multiply it by `contention_factor` to estimate shared-device capacity. The fractions sum to at most one and remain conservative when siblings are idle. These are planning estimates, not enforced CUDA scheduling partitions; actual contention must be measured for the model mix and hardware. Separate processes do not provide a security boundary between untrusted tenants.

Endpoint startup and hydration timings remain profile-driven. The simulator does not model cross-process CUDA kernel contention or a coupled host-wide hydration bandwidth limit. For timing calibration, use startup observations from the intended resident model mix; multiplying solo inference capacity does not establish shared-host cold-start accuracy.

A profile ID can belong to one host plan, and can repeat within that plan to create replicas. Alternative placements use separate profile IDs. Every endpoint profile must match the plan's GPU and physical hourly quote/minimum billing period. Live profile IDs have the form `model-id:GPU-name:region`, so resident profiles must also select the same provider location.

Run the synthetic example:

```sh
cargo run --locked --release -- run \
  --profiles configs/shared-hosts/profiles.json \
  --scenario configs/shared-hosts/scenario.json \
  --policy adaptive --out results/shared-hosts.json
```

The rates, costs and startup values in this example are assumptions. A warm entry for any member starts the entire plan: `copies: 1` means one physical host, including all three endpoints. Per-model quotas count endpoints, including replicas. Results retain logical `nodes`, add physical `hosts`, and attach placement to each endpoint. Physical rent appears once; `peak_nodes` continues to count endpoints.

## Scheduling and ownership

A provisioning action for a member profile requests its entire host plan. Pending allocations forecast every resident endpoint, so demand for another resident model sees capacity already being provisioned. The live controller reserves one allocation, one rent quote and one ownership intent. `limits.max_instances` limits physical allocations; `limits.max_endpoints_per_model` bounds logical endpoints.

Routing remains model-specific and accounts for endpoint occupancy, KV admission and its estimated compute share. A short request to one model does not wait for the router to finish requests to another model. The scheduler releases a shared host only when all resident endpoints are idle and eligible for release. A host fault affects every endpoint, and cleanup terminates the physical allocation once.

Provider preparation checks that the selected offering has enough GPU memory. Unsupported plans are omitted from the effective topology, leaving their individual model profiles available. `configured-topology.json` records the original plans, `host-plan-admission.json` explains admission and `configuration.json` records the effective configuration. `host-endpoints.json` links physical ownership to endpoint IDs and URLs.

## Hydration and serving

The live adapter publishes a versioned `HostTicket`. The host agent hydrates the immutable runtime and components once, shares model files by artifact identity, and launches one native engine per endpoint on local ports 8100 through 8131. Runtime, model and cache transfers share one bounded concurrency semaphore across the entire host. A budget of twenty permits means twenty transfers in total, rather than twenty per model.

Each model starts as soon as its dependencies are available. The agent exposes authenticated URLs such as `/endpoints/a-0/v1/chat/completions` and `/endpoints/b-0/v1/chat/completions`. `/state` and `/endpoints` expose the resident set and individual startup telemetry. `/endpoints/a-0/state` exposes one endpoint's observations. The agent proxies requests; the upstream scheduler chooses an endpoint. Startup observes HTTP availability without submitting inference readiness probes.

Compatible baked GPU caches remain useful. Immutable cache seeds are shared by artifact identity, then copied into each endpoint's private writable cache directory. Compatibility keys retain runtime, model and GPU architecture requirements. Endpoint cache export uses `/endpoints/{id}/cache/export` and integrates with the existing registry. A cache produced by one engine cannot mutate another engine's seed or writable cache. Prefix/attention state remains in the live engine's GPU memory and is not a hydration artifact.

Baking remains an independent artifact operation. A shared host plan does not bake models or kernels, and cache bake jobs require a standalone allocation. Existing published agents do not understand host tickets; a deployment using this feature must pin a newly built agent binary and its SHA-256. The existing standalone fleet configurations remain unchanged.

## Validation and scope

The regression suite covers multiple base models, replicas, aggregate memory admission, shared compute estimates, host faults, busy-sibling release protection and single physical rent accounting. Agent tests exercise real HTTP proxying and fixture processes, confirming that shared runtime/model downloads happen once and owned processes stop at expiry. The live adapter contract test routes 32 calls across two endpoints while journaling and terminating one allocation. The generated gauntlet includes shared resident sets in every fourth seed.

The [local CUDA smoke script](../scripts/host-local-smoke.py) packages the existing native engine and two model directories, hydrates fresh destinations through the actual agent, and runs streamed single calls followed by a concurrent mixed burst. It uses loopback HTTP, warm source files and installed CUDA libraries; its timings exclude cloud allocation and do not establish R2 or RunPod performance. Run it in an existing tmux session after building:

```sh
scripts/build-agent.sh
cargo build --release -p switchyard-agent --example prepare_host_fixture
python3 scripts/host-local-smoke.py --out results/shared-host-cuda \
  --agent target/artifacts/switchyard-agent-x86_64-linux-musl
```

The first implementation provisions a fixed resident set. Adding or removing a model requires replacing the host. It supports one GPU at CUDA ordinal zero per host plan. Dynamic residency, eviction, GPU weight sharing between replicas, work-conserving compute reclamation and hardware execution isolation are separate extensions. Fixed residency provides a usable shared warm pool without claiming those capabilities.

## Measured local validation

On the RTX 5090, two F32 Qwen3.5 engines served concurrently from fresh hydrated destinations. The same agent and engine binaries were used for an empty-cache launch and a launch with captured CUDA caches packaged as hydration artifacts. Each launch completed two initial streamed calls and sixteen mixed-burst calls. Greedy output digests matched within each model across single and batched requests. This checks repeatability for one short prompt, rather than numerical parity against another runtime.

| Model | HTTP available, empty cache | First content, empty cache | HTTP available, hydrated CUDA cache | First content, hydrated CUDA cache |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 0.8B | 7.305 s | 16.799 s | 3.199 s | 3.262 s |
| Qwen3.5 2B | 8.224 s | 17.601 s | 5.770 s | 5.836 s |

Times are measured from local agent launch. The first request is submitted when each endpoint becomes available, so the smaller model can serve while the other model starts. Hydrated caches reduced initial request TTFT from roughly 9.4 seconds to 63–66 ms in these two samples. Cache hydration is the intervention; the test does not separately attribute every cache-dependent engine operation. Both launches stopped their owned engine processes after testing.

The [validation receipt](receipts/shared-host-validation.json) records binary SHA-256 values, startup milestones, request timings and test scope. Detailed agent, transfer and GPU telemetry remains under `results/shared-host-validation/`. To test cache hydration again, pass two same-build, same-GPU CUDA cache directories with `--cache-seeds CACHE_A CACHE_B` to the local smoke script. The fixture checks RTX 5090 compute capability and selected runtime identity.
