# Native fleet execution

The live runtime routes planned OpenAI-style requests to the native Rust engine through the existing scheduler. Provider selection, billing, hydration, and simulator contracts remain independent of the model implementation. A native launch uses the same pinned artifact resolver and ownership journals as the existing serving adapters.

## Serving configuration

A model selects `engine.kind: native`, `dtype: bf16` or `f32`, `language_model_only: true`, its explicit `max_model_len` and `max_num_seqs`, and an empty `extra_args` list. The `native` block supplies `max_weight_bytes`, `minimum_gpu_memory_bytes`, `queue_capacity`, `batch_wait_ms`, and `cuda_ordinal`. Native configuration has no GPU utilization fraction. The minimum GPU memory value filters provider offerings; it is an admission threshold, not an enforced allocator cap or automatic memory sizing.

The runtime must pin its `engine_sha256`. The adapter serializes the fixed serving configuration into `--config-json` and sets the endpoint bind address to `127.0.0.1:8000`, behind the authenticated agent proxy. The native CLI accepts exactly one configuration source: inline JSON or a configuration file. It consumes prepared dense safetensors archives; the vLLM extent layout is rejected for native models.

The public [agent-v0.5.0-rc.3 bootstrap binary](https://github.com/lxe/switchyard/releases/download/agent-v0.5.0-rc.3/switchyard-agent-x86_64-linux-musl) is statically linked for x86_64 musl. Its rebuilt SHA-256 is `8c32c2bc05ca76485a8208af288fa63cbf22bcaec2b63f623b61b7196e8f2e4f`, matching both the GitHub asset digest and a downloaded copy. The [build receipt](receipts/native-agent-build.json) pins the source and builder. A fleet configuration can retain this URL and hash without embedding an expiring storage capability. The release covers the agent; native runtime, cache and fleet acceptance remain separate artifacts.

Cache preparation and reuse retain the generic cache requirements: runtime identity, model revision/repository/archive identity, engine configuration fingerprint, runtime launch fingerprint, and GPU name. Native artifacts add the exact binary, CUDA build, GPU architecture, driver, and bake configuration labels. Changing a required label demands a compatible published artifact; the adapter does not bypass these checks.

Cache publishers can derive these generic labels offline using the same Rust implementation as cache reuse:

```sh
cargo run --locked --features live -- live cache-bindings \
  --config configs/runpod/native-fleet.json \
  --gpu 'NVIDIA GeForce RTX 4090' --out results/native-cache-bindings.json
```

The [native fleet file](../configs/runpod/native-fleet.json) pins the published shared runtime, three model archives, and per-model driver cache families. The output maps each selected model ID to required compatibility labels. This command reads no credentials, performs no network access, and creates no GPU allocation.

## Request protocol

The existing live client requests streamed chat completion with `stream_options.include_usage=true` and `chat_template_kwargs.enable_thinking=false`. The native server reports final prompt and completion counts from actual token IDs in a usage event, followed by `[DONE]`. The output channel is bounded by the admitted token budget plus usage and completion framing; the GPU thread does not wait on client reception.

Before the first useful call, the client compares its locally planned prompt count with `/tokenize`. This CPU-only endpoint uses the same actual chat renderer and tokenizer as generation. It neither executes the model nor warms the GPU. Thinking mode and unknown request options remain rejected.

The [three-model local receipt](receipts/native-fleet-local-rtx5090.json) records 17 calls per model accepted through `switchyard-live::serving::Endpoint` on an RTX 5090: one single call, eight equal-prompt calls, and eight varied-length calls. All 51 responses passed input-token accounting, bounded output, streamed content and usage/DONE framing. Separate fresh-process HTTP benchmarks measured launch-to-first-token without intervening client compilation. These measurements exclude network hydration and provider allocation; filesystem residency was uncontrolled on the shared workstation.

## Workload gauntlet

[`native-gauntlet.json`](../configs/runpod/native-gauntlet.json) specifies 1,470 calls for each model: cold and hot single calls, a 1,000-call burst, three alternate policies, staggered arrivals, mixed prompt lengths and output budgets, shared system prompts, output underestimation, and dependent calls. `prompt_variants` cycles complete user prompts; an empty list retains the ordinary fixed prompt. Request identifiers remain appended independently. These workloads use the generic `standard` service class.

The [token planning receipt](receipts/native-gauntlet-plan.json) records all requests checked against the three pinned local tokenizers. The shared-prefix case has distinct user questions after 47 identical input tokens, crossing the configured 32-token cache checkpoint. Maximum input plus hard output allowance is 105 tokens, below the configured 128-token context bound. This corpus exercises the current small-model bounds; it does not establish long-context serving capacity. Desired output throughput is a measured objective, and a missed target must remain visible in the final receipt.

Run the opt-in real-endpoint contract test with an owned endpoint and local pinned assets:

```sh
SWITCHYARD_NATIVE_ENDPOINT=http://127.0.0.1:8100 \
SWITCHYARD_NATIVE_MODEL=qwen3.5-0.8b \
SWITCHYARD_NATIVE_MODEL_ROOT=data/qwen3.5-0.8b-text \
  cargo test --locked -p switchyard-live --test native_endpoint \
  native_endpoint_accepts -- --ignored --nocapture
```

The test sends useful inference and requires real streamed token accounting. It does not provision or stop the supplied endpoint. Full cloud acceptance additionally requires the final shared runtime, platform bakes, fresh allocation/hydration measurements, scheduler gauntlet receipts, and provider-confirmed cleanup.

## Current execution limits

The [scheduler startup diagnostics](receipts/native-gauntlet-startup-diagnostics.json) record a successful 0.8B cold and hot call followed by cache rejection on two additional elastic hosts. All runtime, model, and cache transfers completed, but the launch wrapper found no validated driver variant for those hosts. The family covered only driver `580.159.04`; a cloud pool does not imply one common host driver. All four pods across the recorded attempts subsequently returned HTTP 404, with estimated rental of $0.0362. The full three-model gauntlet is not established by these partial measurements. One earlier startup failure remains unexplained because engine stderr was not retained before pod deletion.

Native serving reads capped host checkpoint buffers and copies tensors to CUDA after hydration. Incoming model bytes are not loaded directly into GPU tensors. The integrated engine independently prefills equal-length groups and joins varied-length requests into shared decoding, with fixed per-row positions and state compaction. Its [three-model F32 reference gates](native-merged-decode.md) pass; endpoint and cloud throughput require separate measurements. Continuous admission, prefix reuse and paged KV allocation remain subsequent work. F32 mathematical gates pass on the recorded model workloads, while strict BF16 sequence parity with vLLM is incomplete. Quantized projection fixtures do not admit GGUF models to serving.

## F32 fleet candidate and observed execution

The [F32 fleet candidate](../configs/runpod/native-fleet-f32-prefix-candidate.json) pins the reference-tested paged-prefix engine, three native model archives and cache families validated on drivers 580.126.20 and 580.159.04. Its 64 GiB host RAM admission remains unchanged. The separately named [32 GiB candidate](../configs/runpod/native-fleet-f32-prefix-32g-candidate.json) changes only host RAM admission; it retains the engine, GPU, request, hydration and cost limits. The [read-only preflight receipt](receipts/native-f32-prefix-preflight-32g.json) validates the actual published manifests and tokenizers. Preflight does not establish 4B host-memory capacity or full cloud acceptance.

The observer below runs the Rust scheduler gauntlet and collects provider stderr for every pod in its ownership journals. Run it in the existing tmux session. It records the CLI, fleet and workload identities, emits concise phase changes and retains detailed evidence under the selected output directory:

```sh
python deploy/native/run-gauntlet.py \
  --binary target/release/switchyard \
  --config configs/runpod/native-fleet-f32-prefix-32g-candidate.json \
  --workloads configs/runpod/native-gauntlet.json \
  --env-file /path/to/provider.env --env-file /path/to/storage.env \
  --model qwen3.5-0.8b --out results/native-f32-gauntlet-0.8b
```

Omit `--model` for the three-model experiment only after host-memory and fresh-serving gates cover that fleet. Interrupting the observer forwards Ctrl-C to the owned Rust process and lets its cleanup finish. The observer never claims cleanup on behalf of the provider; inspect per-model `outcome.json`, ownership journals and the provider state. If a process is lost, use the existing `live recover` command with that exact run directory. Desired-throughput misses remain in workload receipts.
