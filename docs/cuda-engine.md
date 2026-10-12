# CUDA engine

`switchyard-cuda` serves Qwen3.5 text models (0.8B, 2B, 4B) from hand-written CUDA kernels, without Candle. It exposes the same OpenAI-compatible `/v1/chat/completions` (streamed or whole) as the [native engine](native-continuous-load.md), with greedy decoding or temperature sampling.

```sh
CUDA_COMPUTE_CAP=90a cargo build --release -p switchyard-cuda --features cuda   # 120a for RTX 50-series
switchyard-cuda serve --model data/qwen3.5-2b --name qwen3.5-2b --bind 127.0.0.1:8190
```

At run time it needs the driver and cuBLAS 13 (`libcublas.so.13`, `libcublasLt.so.13`); it loads both dynamically.

## How a step runs

Each step is one forward pass over a flat batch of tokens. Every running sequence contributes its next token; prompt chunks of newly admitted requests fill the rest of the step's token budget (4,096 by default). A new request therefore starts on the next step instead of waiting for the current batch to finish, and long prompts are spread over several steps.

Per layer, a step is three or four cuBLAS GEMMs over all of its tokens plus a few kernels:

- **Projections are fused.** The linear-attention inputs (q, k, v, z, b, a) come from one GEMM, as do the attention inputs (query with gate, key, value) and the MLP's gate and up projections.
- **Linear attention** is two kernels: the causal convolution walks each sequence's tokens; the gated delta rule keeps each head's 128×128 state in registers across a sequence's tokens and finishes with the gated norm.
- **Full attention** is one kernel for the norms, rotary embedding and paged KV write, and one for paged attention with the output gate.
- **Fused residual norms and SiLU.** The residual add is folded into the next RMSNorm; the MLP activation is one kernel.

About 220 kernel launches make a step. Steps that only decode replay a CUDA graph captured at startup for the padded batch size, so a decode step costs one launch from the host. Each sequence reserves its KV blocks for prompt plus `max_tokens` on admission, so no request is preempted.

## Accuracy

`scripts/cuda-accuracy.py` generates greedy continuations with Transformers (bf16) and records the top-5 log-probabilities at each position; `switchyard-cuda score` feeds the same tokens through the engine, prompt as prefill and continuation as decode steps, and reports the same quantities.

| Qwen3.5-2B, 5 prompts, 217 positions | top-1 agreement | median, p95 difference in the reference pick's log-probability |
|---|---|---|
| RTX 5090, eager steps | 214 | 0.0014, 0.045 |
| RTX 5090, graph decode | 214 | 0.0016, 0.042 |
| RTX 5090, 64-token prompt chunks | 214 | 0.0015, 0.043 |
| H100, graph decode | 215 | 0.0016, 0.045 |

Every disagreement is a position where the reference's top two log-probabilities are exactly equal.

## Throughput

`scripts/engine-load.py` (streamed chat completions, prompts of about 60 to 200 tokens, 16 to 64 output tokens, greedy), default flags. Output tokens per second, and median time to first token. [Receipt](receipts/cuda-engine-load.json).

| In flight | H100: switchyard-cuda | H100: vLLM 0.31.0 | RTX 5090: switchyard-cuda | RTX 5090: native engine (BF16, 32 a batch) |
|---:|---|---|---|---|
| 1 | 415 tok/s, 0.044 s | 346, 0.028 s | 306, 0.043 s | 96, 0.11 s |
| 8 | 2,258, 0.046 s | 1,633, 0.060 s | 1,812, 0.043 s | 167, 1.0 s |
| 16 | 3,674, 0.045 s | 2,676, 0.077 s | 3,065, 0.043 s | 215, 1.6 s |
| 32 | 5,484, 0.046 s | 4,198, 0.10 s | 4,308, 0.045 s | 259, 2.1 s |
| 64 | 7,502, 0.056 s | 5,467, 0.13 s | 5,554, 0.051 s | 282, 5.3 s |
| burst of 64 | 4,939 | 4,299 | 4,593 | 283 |

Both H100 runs used the same RunPod H100 SXM (secure cloud, driver 580.126.09), one after the other. switchyard-cuda was ready 5 seconds after launch, vLLM after 79.

## Limits

- **Text only:** no images, no LoRA adapters, no prefix cache.
- **Sampling:** temperature only, without top-p or top-k.
- **Prompt attention** reads each sequence's cached keys once per query token. That is fine at these prompt lengths, but slows prompts of several thousand tokens.
- **Single-stream time to first token** (44 ms) trails vLLM's 28 ms: a one-sequence prefill uses only one block per linear-attention head.
