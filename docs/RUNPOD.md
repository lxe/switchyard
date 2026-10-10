# RunPod live execution

The CLI enables this backend with `--features live`. The live runtime runs Switchyard's scheduler against RunPod pods serving real models through vLLM. It uses the same `FleetView` and `Action` contracts as the simulator. The provider adapter handles inventory and allocation; the Rust pod agent handles model hydration, startup observations, authentication, and transport to a local vLLM server. The runtime owns request admission and routing.

## Configuration and credentials

The checked-in [fleet definition](../configs/runpod/qwen-small.json) and [workloads](../configs/runpod/gauntlet.json) describe the experiment. They contain deployment settings and credential variable names. Keep credential values in your shell or an ignored file:

```text
RUNPOD_API_KEY=...
PODTRAIN_R2_ACCESS_KEY_ID=...
PODTRAIN_R2_SECRET_ACCESS_KEY=...
```

Use repeatable `--env-file` arguments to read existing credential files. Process environment values take precedence, then the first file containing each variable. The loader does not mutate the process environment.

```sh
cargo run --features live -- live discover --env-file /path/to/credentials.env
cargo run --features live -- live check --env-file /path/to/credentials.env
cargo run --features live -- live gauntlet --env-file /path/to/credentials.env \
  --model qwen3.5-0.8b --out results/my-experiment
```

`discover` queries the current RunPod REST v2 catalog. `check` validates model manifests and downloads the pinned tokenizers and chat templates to a local cache. Both commands avoid provisioning. `gauntlet` rents GPUs and executes the workload definitions. The default experiment runs one model at a time, with at most two active pods, a configured hourly rental ceiling, and separate wall-clock, startup, request, and rental budgets. Experiment-wide cost and wall-clock limits also bound the complete multi-model run. Each model gets its own ownership journal and cleanup receipt.

## Hydration and model serving

The storage adapter signs S3 requests using AWS SigV4 with R2's `auto` signing region. It reads a `shards-v1` manifest whose chunks describe independent tar shards split into ordered slices. The fleet configuration pins the archive identity, which commits to the ordered chunk hashes. Chunk sizes and SHA256 hashes are checked during transfer. Extraction permits only relative regular files and directories. Declared file sizes and optional hashes are checked before the model directory is published.

Long-lived storage credentials stay on the client. The client writes a short-lived bootstrap ticket containing signed GET capabilities for the selected model's chunks and configured vLLM arguments. A pod receives the ticket URL and a random agent token. It downloads a checksum-pinned static Rust agent, hydrates the model, then starts vLLM on loopback port 8000. The agent exposes authenticated `/state`, `/probe`, and model-serving paths on port 8080. The provider constructs the RunPod HTTP proxy address.

Agent observations include hydration progress, elapsed time, observed download rate, model-cache status, and engine readiness. Image-cache status remains unknown because a process inside the container cannot measure the provider's earlier image preparation. The live adapter models that preparation through startup priors and records the whole allocation-to-ready interval.

The initial serving configuration disables thinking and prefix caching, uses eager execution, and bounds context and engine concurrency. These choices make the first experiment easier to compare; they are configurable performance tradeoffs. Input planning renders the pinned Hugging Face chat template with MiniJinja and counts tokens using the Rust `tokenizers` library. Before admitting the first request to a pod, the runtime compares its count with vLLM's `/tokenize` result. Every completed stream must report matching input usage and obey its output bound.

## Measurements and recovery

Reports contain streamed first-content latency, response latency, output tokens per wall second, final usage, scheduler diagnostics, allocation counts, and estimated rent. A throughput target is a scheduling objective, not a guarantee. The reports measure the complete client/runtime/provider path; they are not isolated GPU kernel benchmarks. Partial generated token counts are estimates obtained by tokenizing streamed text, while final usage comes from vLLM.

The runtime journals a unique launch name before allocation and the pod ID as soon as ownership is known. It never repeats a failed allocation POST blindly. On normal completion, request failure, or Ctrl-C, it aborts outstanding requests, terminates only its own pods, confirms their termination, and deletes its bootstrap tickets. After a process or machine crash, recover the specific model directory:

```sh
cargo run --features live -- live recover --env-file /path/to/credentials.env \
  --out results/my-experiment/qwen3.5-0.8b
```

Recovery checks both the saved pod ID and exact name. Pending allocation names also let it reconcile a successful provider allocation whose response was lost. Preserve these journals until cleanup is confirmed. Agent lifetime limits stop serving, but terminating GPU rental requires the runtime or recovery command.

## Extension boundaries

Implement `switchyard_provider::Provider` for another GPU cloud. Supply offers, allocation, inventory, termination, and a serving address. Each provider stores its own catalog configuration. The runtime accepts a provider-neutral `ExecutionConfig` and submits an opaque `ContainerSpec`; only the RunPod CLI wrapper reads `ComputeConfig`. Storage uses a separate S3-compatible adapter. Implement `switchyard_core::scheduler::Scheduler` to replace the controller and pass it to `LiveRuntime::with_scheduler`. `WorkCall` carries text chat inputs, model selection, output planning bounds, arrivals, service class, and dependencies.

This release executes planned workloads through a library and CLI. It does not provide a persistent public front-door service. Tools, images, live model swapping, speculative decoding, distributed multi-GPU serving, and provider-independent billing reconciliation need additional adapters. vLLM owns its internal batching; live routing admits bounded concurrent requests, and simulator service weights do not become weighted GPU execution in vLLM.

Provider contracts and rates follow the [RunPod v2 catalog](https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types), [pod creation API](https://docs.runpod.io/api-reference-v2/pods/create-a-pod), and [storage pricing](https://docs.runpod.io/pods/pricing). Storage signing follows [Cloudflare's R2 S3 API](https://developers.cloudflare.com/r2/api/s3/api/). The model-serving configuration uses [vLLM's supported-model documentation](https://docs.vllm.ai/en/latest/models/supported_models/).

## Agent builds

`scripts/build-agent.sh` builds a static x86-64 musl binary with a digest-pinned Rust 1.98.1 Alpine image. It mounts the source read-only and keeps build outputs in ignored `target/` directories. Publish the resulting binary at the configured `agent_url` and set `agent_sha256` to its SHA256 digest before provisioning. Pods reject a bootstrap binary whose checksum does not match.
