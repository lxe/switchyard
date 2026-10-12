# Native LoRA serving

A LoRA adapter is a small set of low-rank matrices applied to a shared base model during inference. The native engine loads standard PEFT `adapter_config.json` and `adapter_model.safetensors` artifacts without merging them into the base weights. A loaded adapter has a model alias, such as `qwen3.5-0.8b+pocket-run-42`, which clients select through the OpenAI request's `model` field.

A conversation thread belongs to the application. Pocket Model can retain that alias with the thread and send its message history to `/v1/chat/completions`. The inference endpoint does not create persistent conversation resources. Adapter loading completes before the application accepts requests that use the alias. This follows [vLLM's adapter selection and management interface](https://docs.vllm.ai/en/stable/features/lora/).

## Loading and request selection

Enable explicit adapter bounds in the engine configuration. Examples are provided for [0.8B](../examples/native-qwen35-0.8b-lora.json), [2B](../examples/native-qwen35-2b-lora.json) and [4B](../examples/native-qwen35-4b-lora.json). These profiles retain the existing fixed context, batch and prefix-cache limits. Adapter serving is disabled when the `lora` section is absent.

For an adapter directory already inside the configured `lora.root`, use the vLLM-compatible management route:

```sh
curl http://127.0.0.1:8100/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name":"qwen3.5-0.8b+pocket-run-42","lora_path":"/srv/adapters/run-42"}'
```

To upload files directly to the native endpoint, use the multipart extension. The server streams each field into private staging instead of buffering the whole HTTP body:

```sh
curl http://127.0.0.1:8100/v1/adapters/qwen3.5-0.8b+pocket-run-42 \
  -F 'adapter_config=@adapter_config.json' \
  -F 'adapter_model=@adapter_model.safetensors'
```

For an adapter stored in R2 or another object store, send the PEFT config as `adapter_config` and a signed safetensors URL as `adapter_url`. These fields replace `lora_path`:

```json
{
  "lora_name": "qwen3.5-0.8b+pocket-run-42",
  "adapter_config": {
    "peft_type": "LORA",
    "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none"
  },
  "adapter_url": "https://object-store.example/adapter_model.safetensors?signed-query"
}
```

Use the adapter's actual complete config; the short example only illustrates the transport fields. Remote transport uses a bounded direct download, rejects redirects, and removes staging after parsing or cancellation. URLs do not enter adapter identities or telemetry. The cloud agent currently retains its two-MiB request-body bound, so use URL loading through that proxy for large adapters; multipart upload is available directly on the native endpoint.

Loading returns a revision, rank and process-local `lora_int_id`. `/v1/models` lists the base and loaded aliases, including each adapter's parent model and revision. Select the alias for normal or streamed inference:

```json
{
  "model": "qwen3.5-0.8b+pocket-run-42",
  "messages": [{"role": "user", "content": "Reply to this support ticket."}],
  "temperature": 0,
  "max_tokens": 16,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

`/v1/completions`, `/v1/chat/completions` and `/tokenize` accept loaded aliases. `/generate` remains the base-model diagnostic route. A thread-creation handler can publish the two PEFT files, load them on its selected base endpoint, and save the returned alias with the thread. Application thread persistence and automatic adapter placement across cloud replicas are separate integrations; this change supplies the native serving interface.

Unload through `POST /v1/unload_lora_adapter` with either `{"lora_name":"..."}` or `{"lora_int_id":1}`. A queued or active request holds a lease on the exact loaded revision. Unloading returns HTTP 409 while leases remain. Duplicate names also return 409; replacing an adapter requires successful unload followed by load. IDs increase on reload, and an old numeric ID cannot unload a later adapter using the same name. Unknown aliases return 404.

## Execution and cache ownership

For every targeted projection, native inference computes `base(x) + scale × B(A(x))`. Ordinary LoRA uses `scale = alpha/r`; rsLoRA uses `alpha/sqrt(r)`. Adapter matrices remain F32 on the device. With a BF16 base, the adapter branch and addition run in F32 before the result is cast back to the base precision, matching the unmerged [PEFT Linear calculation](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py).

Base tensor handles are shared across adapter engines. Loading does not reread the base checkpoint or create a second full set of base weights. Each adapted model has a fresh state owner and prefix cache, which prevents base-model or other-adapter attention and recurrent state from satisfying an adapted request. A revision hashes canonical JSON configuration and safetensors bytes; JSON whitespace and key ordering do not create different revisions.

One inference owner performs device attachment, generation and unloading. It batches requests with the same adapter identity; requests using different adapters execute in separate batches. Fused multi-adapter kernels and continuous admission across adapter identities are outside the current execution contract.

Adapter transport, parsing and device attachment share a single admission permit. A canceled HTTP waiter cannot release the permit while its blocking parser still owns it. The catalog publishes an alias only after attachment succeeds. Failed parsing or construction leaves the base and loaded catalog unchanged.

## Fixed resource limits

| Configuration field | Meaning |
| --- | --- |
| `root` | Allowed local adapter directory tree and private upload staging root |
| `base_model_aliases` | Accepted PEFT base-model names in addition to the configured model ID |
| `max_loaded` | Number of resident adapters, excluding the base model |
| `max_rank` | Maximum accepted low-rank dimension |
| `max_upload_bytes` | Aggregate artifact byte limit and expanded CPU F32 matrix limit |
| `max_resident_bytes` | Aggregate device-resident adapter matrix limit |
| `transfer_timeout_s` | Network/upload deadline; parsing retains its admission permit until completion |

Device matrix accounting excludes shared base weights, sampler bookkeeping and prefix state. Every adapter receives the separately configured per-model prefix-cache bound, so provisioning must include `max_loaded × cache.max_bytes` in addition to the base cache and adapter matrices. Limits are fixed; loading never invokes automatic GPU sizing.

The loader checks base-name bindings, every A/B pair, matrix rank, finite values, declared targets and projection shapes. PEFT often records a temporary training path such as `/work/model`; accepting that name requires an explicit configured alias. A base-name alias and matching shapes cannot establish the original checkpoint revision when PEFT omits it, so callers must retain the training run's base revision when selecting an endpoint.

## Supported artifacts and validation

The dense Qwen3.5 implementation supports attention `q_proj`, `k_proj`, `v_proj`, `o_proj`; MLP `gate_proj`, `up_proj`, `down_proj`; linear-attention `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`; and `lm_head`. The actual Pocket Model SFT/GRPO adapters target language attention and MLP projections. Regex, name-list and `all-linear` declarations are accepted when every supplied tensor can be applied to a supported projection.

Unsupported active features fail explicitly: DoRA, aLoRA, bias updates, replacement modules, embedding/vision adapters, rank/alpha patterns, base-mutating initializers and other specialized PEFT variants. Packed GGUF bases currently reject LoRA attachment rather than expanding their base matrices. Language-adapter execution with image input has not received its own acceptance gate.

The [local receipt](receipts/native-lora-local.json) records real trained 0.8B and 2B adapters, each with 150 A/B pairs at rank16. Both passed local GPU HTTP/SSE tests for file loading, multipart upload, object-URL streaming, mixed-length generation, repeated-prefix reuse, concurrent base/adapter requests, duplicate rejection, safe unload/reload, stale-ID rejection and unchanged base outputs. Four single-request reference traces produced exact F32 Transformers + PEFT agreement for 64 base and 64 adapted tokens per model. The 4B profile has a separately labeled synthetic-adapter functional gate; no real trained 4B adapter or corresponding accuracy claim is supplied.

Reproduce serving with [lora_http.py](../crates/switchyard-engine/benchmarks/lora_http.py), then run [lora_reference.py](../crates/switchyard-engine/benchmarks/lora_reference.py) in a compatible Torch/Transformers/PEFT environment. The independent oracle explicitly maps every Pocket VLM text-adapter key to its text-only HF module and checks all installed A/B tensors. CPU tests independently verify scalar LoRA arithmetic, BF16 rounding, full hybrid-model prefill/decode, cache separation, lease ownership and bounded transport cancellation.

Adapter serving needs no GPU bake job and does not modify simulator contracts or scheduling decisions. Optional CUDA driver caches continue to follow the [ordinary serving cache lifecycle](native-cache-registry.md). Existing published cloud runtime artifacts are unchanged; deploying this feature on RunPod requires publishing the rebuilt native executable and completing fresh cloud adapter gates.
