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

## What follows

Continuous admission fills the rows that finish and lets new prompts start without waiting for the group: it addresses the draining and the queue's time to first token. On its own it does not remove the prefill cost. Prefill needs batching across requests, interleaving with decode steps, and profiling: 120 ms for a prompt of about 150 tokens is slow for a 2B model. With both, the measured decode steps put about 2,000 tokens per second at 32 rows within reach on this GPU.

The engine also generated a token or two after the end-of-turn token before stopping.
