# Switchyard inference engine

`switchyard-engine` is a Rust text-generation server that loads local native tensors and executes a model directly through Candle and CUDA. The first adapter implements dense Qwen3.5 attention, gated delta recurrence, normalization and feed-forward layers. It does not launch Python, Torch, vLLM or a restored process.

This is an initial engine, not the default production serving backend. The [F32 numerical gate](../../docs/native-engine-accuracy.md) passes against the released Transformers implementation, including complete single and batched sequences. Strict BF16 parity with vLLM remains incomplete. The [measurement report](../../docs/rust-inference-engine.md), [pinned model configurations](../../docs/native-model-fleet.md) and [development goals](../../docs/native-engine-roadmap.md) describe measured behavior and remaining work.

## Build and launch

CPU tests need the workspace Rust toolchain. GPU execution additionally needs a CUDA toolkit and compatible NVIDIA driver. For the measured RTX 5090 build:

```sh
CARGO_BUILD_JOBS=6 CUDA_COMPUTE_CAP=120 \
  cargo build --locked --release -p switchyard-engine --features cuda
```

Set `model_root` in [the example configuration](../../configs/engine/qwen3.5-0.8b-rtx5090.json) to a local Qwen3.5 model directory containing `config.json`, native safetensors, `tokenizer.json` and `chat_template.jinja`. Relative paths resolve against the process working directory. An optional `--model-root` overrides the configured directory for an agent-published installation. Serving adapters can supply the same fixed configuration as inline JSON through `--config-json` instead of `--config`; exactly one source is required.

Bake a smaller text-only tensor directory once, then serve it:

```sh
target/release/switchyard-engine \
  --config configs/engine/qwen3.5-0.8b-rtx5090.json \
  prepare --output data/qwen3.5-0.8b-text

target/release/switchyard-engine \
  --config configs/engine/qwen3.5-0.8b-rtx5090.json \
  --model-root data/qwen3.5-0.8b-text serve --bind 127.0.0.1:8100
```

Preparation preserves source tensor precision and dense native names, removes unused vision tensors, and writes fresh safetensors and model/tokenizer metadata. It does not quantize, fuse projections, or capture GPU allocations. Failed preparation can leave an incomplete output directory; use a fresh destination for the next attempt. The generic artifact packer can package the prepared directory independently. The [native RunPod workflow](../../docs/native-runpod.md) records a fresh 0.8B baseline cloud launch. Final three-model cloud and scheduler acceptance must use the integrated binary and its own platform bakes.

```sh
curl -N http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"What is 17 plus 25?"}],"max_tokens":32,"temperature":0,"stream":true}'
```

The implemented OpenAI subset is greedy text completions and chat completions, with optional SSE streaming. `stream_options.include_usage=true` adds a final usage event before `[DONE]`; counts come from generated token IDs. Chat requests accept `chat_template_kwargs.enable_thinking=false`, while thinking mode is rejected. `/tokenize` accepts the same model/messages with `add_generation_prompt=true` and returns the actual rendered prompt token count without GPU execution. Completion prompts accept text, token IDs, or a batch of token-ID sequences. Response choices include token IDs. Sampling controls beyond `temperature=0`, tools, vision inputs, stop strings and reasoning parsing are unsupported; unknown request fields are rejected. `/v1/models` lists the configured model, `/health` reports the listening server, and `/generate` accepts native token requests.

## Offline GPU kernel bake

Candle's compiled PTX can still trigger NVIDIA driver compilation on first use. An empty private driver cache took 11.41 seconds to the first token; reusing the separately baked cache reduced that to 0.88 seconds on the measured host. Bake this work on the selected GPU and driver before launch:

```sh
target/release/switchyard-engine-bake \
  --engine target/release/switchyard-engine \
  --config configs/engine/qwen3.5-0.8b-rtx5090.json \
  --requests results/token-requests.json --output data/gpu-bake

CUDA_CACHE_PATH="$PWD/data/gpu-bake/cuda" \
  target/release/switchyard-engine \
  --config configs/engine/qwen3.5-0.8b-rtx5090.json serve
```

The bake directory must be fresh. Its receipt binds the engine binary, configuration, representative requests, GPU and driver. The cache is a separate artifact from weights; a new process successfully reused a copied cache directory. Compatibility on other GPU/driver builds is unvalidated. Baking is offline work, while serving performs only dependency loading and useful client inference.

## Execution and memory ownership

The model factory owns immutable weight tensors. Each inference cohort receives a new session with private convolution, recurrent and attention state. Rows with equal prompt length execute together as GPU tensors. Different lengths form separate cohorts because padding would modify recurrent state. The server combines independent queued jobs up to `max_batch_size`; a cohort completes before another cohort begins. Continuous batching and mixed-length decoding are not implemented.

Configuration supplies context length, batch size, maximum host weight bytes, queue capacity and the batch collection interval. No free-memory fraction or dummy inference determines these limits. Attention and recurrent buffers allocate within those sequence bounds; this is not yet a preallocated paged KV pool or a production GPU-memory admission plan. Weight loading checks native tensor structure and shape without hashing content at serve time. All safetensors files share the configured aggregate host buffer budget.

One thread owns GPU execution. HTTP admission uses a bounded job queue, where one job can contain up to `max_batch_size` sequences. Each stream has capacity for its admitted output tokens and its final event. GPU generation uses nonblocking sends, so a disconnected client cancels its rows without blocking other rows. Completed or cancelled rows remain in the tensor batch until the entire cohort stops. Queue saturation returns HTTP 429.

Tokenizer preparation runs alongside GPU initialization and weight loading. The listener opens when these dependencies have completed; there is no inference readiness probe. JSON telemetry on stderr records device initialization, weight reception from local files, tokenizer completion, model loading, listener publication, queue batches and completed cohorts. The client benchmark separately measures actual streamed token receipt.

## Validation and benchmarks

```sh
cargo test --locked -p switchyard-engine --all-targets
cargo clippy --locked -p switchyard-engine --all-targets -- -D warnings
```

Tiny CPU model tests compare cached decoding with full causal prefixes and batched rows with independent sessions. Generated workload tests exercise ordering, context admission, output bounds, EOS and per-row cancellation.

Create token requests with the model's own chat template:

```sh
target/release/switchyard-engine \
  --config configs/engine/qwen3.5-0.8b-rtx5090.json tokenize \
  --input crates/switchyard-engine/benchmarks/fixtures/chats.json \
  --output results/token-requests.json --max-tokens 32
```

The Rust benchmark owns and cleans up its serving process. It sends useful inference immediately upon listener telemetry, then measures hot single calls and a concurrent burst:

```sh
target/release/switchyard-engine-bench \
  --engine target/release/switchyard-engine \
  --config configs/engine/qwen3.5-0.8b-rtx5090.json \
  --requests results/token-requests.json --output results/engine-benchmark
```

`benchmarks/vllm_reference.py` is an offline oracle for the pinned Ubuntu vLLM runtime. It consumes the same `/output/requests.json`, loads `/model`, uses an explicit 1 GiB KV budget, and writes `/output/vllm.json`. Its worker sizing override is confined to the benchmark process. Python is used only for this external reference. Mount an owned writable cache at `/opt/switchyard/cache`; do not mutate a published artifact tree.

Pass `--reference results/vllm.json` to retain side-by-side token differences. `--strict-parity` exits unsuccessfully on any single or burst mismatch, after saving the receipt. A failed gate must not be reported as reference-equivalent serving. Receipts include the executed engine binary SHA-256, configuration, input/output IDs, per-token client timing and native phase events.

## Model adapters

`weights` owns native file loading and offline preparation. `models` owns model dispatch and the `ModelFactory` / `ModelSession` contracts. `engine` owns greedy generation and sequence limits, while `server` owns HTTP admission and transport. The simulator, fleet scheduler, provider adapter and hydration implementation do not depend on this crate.

Dense Qwen3.5 dimensions come from model configuration; 0.8B, 2B and 4B have local full-checkpoint validation recorded in the model fleet report. A new architecture needs a factory and session implementation, without changing HTTP admission or cloud provisioning. Vision will require a separate projector and position-aware embedding prefill contract; neither vision projectors nor multimodal rotary positions are implemented in this text-only adapter. Quantization, shared-prefix state, paged KV storage and fused recurrent kernels are also remaining work.

Candle is pinned to a source revision, and the initial Qwen3.5 adapter derives from an unmerged Candle contribution. See [THIRD_PARTY.md](THIRD_PARTY.md) for provenance and licensing.
