# Native engine under continuous load

How the native engine serves a steady stream of requests, and where its time goes. The engine admits a group of requests, decodes them together, and admits the next group only when that one finishes; there is no continuous admission yet ([architecture](ARCHITECTURE.md)). This measures that behaviour so the effect of continuous admission can be compared against it.

## Workload

`scripts/engine-load.py` drives an OpenAI-compatible endpoint with streamed, greedy chat completions. Prompts are 40 to 150 words (about 60 to 200 tokens with the chat template); each request asks for 16, 32, 48 or 64 output tokens. It runs closed loops at 1, 8, 16, 32 and 64 requests in flight, then a burst of 64 at once, and reports output tokens per second, time to first token and request latency.

```sh
switchyard-engine --config qwen3.5-2b.json serve --bind 127.0.0.1:8190 &
python scripts/engine-load.py http://127.0.0.1:8190 qwen3.5-2b result.json
```

## Qwen3.5-2B on an RTX 5090

Context bound 512; other settings as the fleet's [2B engine configuration](../configs/engine/qwen3.5-2b-rtx5090.json). Output tokens per second, and median time to first token. [Receipt](receipts/native-continuous-load-rtx5090.json).

| In flight | F32, 8 a batch (fleet) | BF16, 8 a batch | BF16, 32 a batch |
|---:|---|---|---|
| 1 | 91 tok/s, 0.10 s | 93, 0.11 s | 96, 0.11 s |
| 8 | 157, 1.2 s | 171, 0.9 s | 167, 1.0 s |
| 16 | 182, 2.1 s | 207, 1.9 s | 215, 1.6 s |
| 32 | 201, 4.9 s | 192, 5.0 s | 259, 2.1 s |
| 64 | 203, 10.5 s | 193, 10.8 s | 282, 5.3 s |
| burst of 64 | 203 | 191 | 283 |

Throughput levels off near 200 tokens per second with 8 a batch and 280 with 32, and time to first token grows with the queue.

## Where the time goes

The engine's own telemetry (`mixed_decode_step`, `prefix_prefill`) splits the serving time:

- **Batched decode is efficient.** A BF16 decode step takes 8.0 ms with one active row, 10.7 ms with 8 and 15.4 ms with 32: about 2,000 tokens per second at 32 rows.
- **Prefill takes most of the time:** 57% of serving time with 8 a batch, 67% with 32. Prompts are prefilled one at a time (476 of 518 prefills were single rows), about 120 ms each, and decode waits for them.
- **Batches drain.** Rows that finish are not replaced, so decode steps averaged 4.7 active rows of 8, and 8.6 of 32.
- About 7% is neither decode nor prefill.

## Against vLLM on an H100

The same load on one H100 80GB SXM (RunPod secure cloud, driver 580.126.09), with the native runtime built for `sm_90a` and vLLM 0.31.0 on its defaults (CUDA graphs, prefix caching, chunked prefill; BF16, thinking off). Both engines produced the same output tokens. Output tokens per second, and median time to first token. [Receipt](receipts/native-continuous-load-h100.json).

| In flight | Native F32, 8 a batch | Native BF16, 32 a batch | vLLM 0.31.0 |
|---:|---|---|---|
| 1 | 56, 0.21 s | 50, 0.22 s | 359, 0.03 s |
| 8 | 98, 1.5 s | 93, 1.7 s | 1,519, 0.07 s |
| 16 | 114, 3.6 s | 124, 2.2 s | 2,522, 0.09 s |
| 32 | 110, 9.0 s | 141, 3.3 s | 3,722, 0.10 s |
| 64 | 111, 19.4 s | 147, 10.3 s | 5,308, 0.13 s |

vLLM is 7 times faster for one stream and 36 times faster at 64 in flight. The native engine was ready 4 to 9 seconds after launch, vLLM 90 seconds after.

The native engine is slower on the H100 than on the RTX 5090: a one-row BF16 decode step takes 14.4 ms against 8.0 ms, and a prefill 232 ms against 120 ms. Streaming the 2B model's weights through H100 memory takes about 1.4 ms, so most of a step is host-side work: launching many small kernels one after another, without CUDA graphs, on a server CPU (Xeon 8480+) slower per thread than the desktop's. Throughput depends on the CPU more than on the GPU.

## What follows

Continuous admission fills the rows that finish and lets new prompts start without waiting for the group: it addresses the draining and the queue's time to first token. On its own it does not remove the prefill cost. Prefill needs batching across requests, interleaving with decode steps, and profiling: 120 ms for a prompt of about 150 tokens is slow for a 2B model. With both, the measured decode steps put about 2,000 tokens per second at 32 rows within reach on the RTX 5090.

Closing the gap to vLLM needs the per-step overhead removed first: capturing decode steps in CUDA graphs, and fusing the small operations of each layer. vLLM's single stream (359 tokens per second on the H100, 2.8 ms a token) shows how much of the native engine's 14 ms step is overhead rather than arithmetic.

The engine also generated a token or two after the end-of-turn token before stopping.
