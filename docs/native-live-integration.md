# Native fleet execution

The live runtime routes planned OpenAI-style requests to the native Rust engine through the existing scheduler. Provider selection, billing, hydration, and simulator contracts remain independent of the model implementation. A native launch uses the same pinned artifact resolver and ownership journals as the existing serving adapters.

## Serving configuration

A model selects `engine.kind: native`, `dtype: bf16` or `f32`, `language_model_only: true`, its explicit `max_model_len` and `max_num_seqs`, and an empty `extra_args` list. The `native` block supplies `max_weight_bytes`, `minimum_gpu_memory_bytes`, `queue_capacity`, `batch_wait_ms`, and `cuda_ordinal`. Native configuration has no GPU utilization fraction. The minimum GPU memory value filters provider offerings; it is an admission threshold, not an enforced allocator cap or automatic memory sizing.

The runtime must pin its `engine_sha256`. The adapter serializes the fixed serving configuration into `--config-json` and sets the endpoint bind address to `127.0.0.1:8000`, behind the authenticated agent proxy. The native CLI accepts exactly one configuration source: inline JSON or a configuration file. It consumes prepared dense safetensors archives; the vLLM extent layout is rejected for native models.

Cache preparation and reuse retain the generic cache requirements: runtime identity, model revision/repository/archive identity, engine configuration fingerprint, runtime launch fingerprint, and GPU name. Native artifacts add the exact binary, CUDA build, GPU architecture, driver, and bake configuration labels. Changing a required label demands a compatible published artifact; the adapter does not bypass these checks.

## Request protocol

The existing live client requests streamed chat completion with `stream_options.include_usage=true` and `chat_template_kwargs.enable_thinking=false`. The native server reports final prompt and completion counts from actual token IDs in a usage event, followed by `[DONE]`. The output channel is bounded by the admitted token budget plus usage and completion framing; the GPU thread does not wait on client reception.

Before the first useful call, the client compares its locally planned prompt count with `/tokenize`. This CPU-only endpoint uses the same actual chat renderer and tokenizer as generation. It neither executes the model nor warms the GPU. Thinking mode and unknown request options remain rejected.

The [local client receipt](receipts/native-live-api-rtx5090.json) records one call and eight concurrent calls accepted through `switchyard-live::serving::Endpoint` on an RTX 5090. All nine responses had matching input counts, valid usage/DONE framing, and identical answer hashes. The measured listener interval excludes network hydration. Client compilation intervened after listener publication, so that receipt does not claim launch-to-first-token latency.

## Workload gauntlet

[`native-gauntlet.json`](../configs/runpod/native-gauntlet.json) specifies 1,470 calls for each model: cold and hot single calls, a 1,000-call burst, three alternate policies, staggered arrivals, mixed prompt lengths and output budgets, shared system prompts, output underestimation, and dependent calls. `prompt_variants` cycles complete user prompts; an empty list retains the ordinary fixed prompt. Request identifiers remain appended independently. These workloads use the generic `standard` service class.

The [token planning receipt](receipts/native-gauntlet-plan.json) records all requests checked against the three pinned local tokenizers. Maximum input plus hard output allowance is 87 tokens, below the configured 128-token context bound. This corpus exercises the current small-model bounds; it does not establish long-context serving capacity. Desired output throughput is a measured objective, and a missed target must remain visible in the final receipt.

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

Native serving reads capped host checkpoint buffers and copies tensors to CUDA after hydration. Incoming model bytes are not loaded directly into GPU tensors. The integrated batch engine groups equal-length prompts; mixed-length decoding is being validated separately. Continuous admission, prefix reuse and paged KV allocation remain subsequent work. F32 mathematical gates pass on the recorded model workloads, while strict BF16 sequence parity with vLLM is incomplete. Quantized projection fixtures do not admit GGUF models to serving.
