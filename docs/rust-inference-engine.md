# Rust inference engine measurements

The native engine loads Qwen3.5 0.8B text tensors and generates tokens directly through Rust and Candle/CUDA operations. It provides greedy OpenAI-style completions and chat completions, actual SSE token delivery, and bounded batching across HTTP requests. Python, Torch and CRIU are absent from the candidate serving process. The fleet simulator and cloud adapters remain independent.

The implementation and commands are documented in the [engine README](../crates/switchyard-engine/README.md). The endpoint passes the [F32 mathematical accuracy gate](native-engine-accuracy.md); strict BF16 sequence parity with vLLM remains incomplete. This engine has not replaced the prepared vLLM production path. The measurements below describe the initial implementation; the accuracy document contains the corrected engine's receipt and startup measurements.

## Local startup and GPU cache control

Measurements were taken on an RTX 5090 with driver 615.71.09, CUDA toolkit 13.4 and Rust 1.98.1. Model files were resident in the OS page cache on NVMe-backed Btrfs. Every sample started a fresh serving process. These are single samples on a shared workstation, excluding container allocation, artifact reception and filesystem extraction.

The candidate binary used in all four reported cases has SHA-256:

```
17a3941d96ae2287534ea5a01051ba7f44f527e3416c7e2494b5b8b58d3f9631
```

| Case | Launch to observed listener | Launch to first streamed token | Concurrent burst output |
| --- | ---: | ---: | ---: |
| 16 requests with varied prompt lengths, existing driver cache | 0.789 s | 0.924 s | 127.1 tokens/s |
| Eight equal-length requests, empty private driver cache | 2.653 s | 11.405 s | 791.6 tokens/s after the initial call |
| Same eight requests, reused private driver cache | 0.738 s | 0.874 s | 795.6 tokens/s |
| Same eight requests, separately baked cache copied to a new path | 0.742 s | 0.882 s | 795.3 tokens/s |

All requests used a 128-token context bound, eight-sequence batch bound, BF16 weights, greedy decoding and a 32-output-token cap. The benchmark uses `ignore_eos=true` to maintain equal output work, including continuations beyond a normal stop token. The first useful inference occurs immediately after listener telemetry; there is no inference readiness probe. The listed token time is client receipt, not an internal GPU timestamp.

The empty-versus-reused driver-cache control changed only `CUDA_CACHE_PATH`. Its first-token difference was 10.531 seconds. A separate offline bake created nine cache files totalling 15,467,702 bytes. Copying those files to a different cache directory preserved the startup reduction, which demonstrates reuse without keeping a GPU process alive.

The control establishes that availability of the generated NVIDIA kernel cache causes the measured delay. It does not attribute every millisecond to an individual driver compiler function. CUDA module loading and first-use kernel work still need lower-level tracing for that finer breakdown.

## Startup phase timings

The copied-bake case is the selected native startup pattern: hydrate native weights and a compatible small GPU cache independently, then start a fresh process. The following phase values come from its internal monotonic telemetry; the client clock also includes process creation, dynamic loading and telemetry delivery.

| Phase | Empty driver cache | Reused driver cache |
| --- | ---: | ---: |
| CUDA device initialization | 0.072 s | 0.069 s |
| Local weight read and native header parsing | 0.429 s | 0.438 s |
| Model construction, weight transfer and initial CUDA module use | 2.114 s | 0.190 s |
| Tokenizer completion after process entry, overlapping the preceding work | 0.278 s | 0.283 s |
| First call to first client token, including queue/HTTP work | 8.749 s | 0.132 s |

Tokenizer preparation runs in parallel with device initialization and weight loading. The tokenizer is then shared through `Arc`; request admission does not copy its vocabulary. On the same mixed-length workload, removing tokenizer copies and overlapping preparation reduced a preliminary 1.700-second launch-to-token result to 0.909 seconds. The final checked binary measured 0.924 seconds. These observations are not a statistical latency distribution.

Native preparation reduced the source checkpoint from 1,746,942,600 bytes to a 1,504,832,320-byte, 320-tensor text file by excluding vision weights. Tensor precision and names remain native; the candidate does not consume vLLM's fused final-state layout.

## Numerical comparison

The offline reference used vLLM 0.31.0 from the pinned prepared Ubuntu runtime, the same original checkpoint and literal input token IDs, BF16 execution, eager mode, Triton full attention, disabled prefix caching, context 128 and at most eight sequences. A benchmark-only worker override selected an explicit 1 GiB KV allocation without a dummy profiling forward or automatic sizing.

An initial four-prompt, 16-output-token test matched all 64 generated IDs against both single and batched vLLM output. The final broader corpus contains 16 prompts and forces 32 output tokens per prompt:

| Comparison | Entire sequences matching | Matching output positions |
| --- | ---: | ---: |
| Final Rust HTTP single calls vs vLLM single calls | 6 / 16 | 374 / 512 |
| Final Rust HTTP concurrent burst vs vLLM batch | 6 / 16 | 378 / 512 |
| vLLM single vs vLLM batch | 11 / 16 | See reference outputs in receipt |

The first generated token matched on all 16 cases in both candidate modes. Complete-sequence equality does not pass. Some divergence happens after a normal EOS, but differences also occur before EOS. Positional agreement after the first divergent token is not a numerical accuracy metric because the models are then consuming different generated histories.

The candidate uses sequential F32 gated-delta recurrence; the reference uses chunked prefill and fused recurrent decode kernels. Known differences include intermediate rounding and matrix-operation paths. Source inspection identified F32 gate computation and BF16 normalized prefill storage in the reference, and these are reflected in the adapter. This inspection does not establish that all remaining mismatches are harmless rounding. Teacher-forced logits, layer outputs and recurrent states need comparison before claiming reference equivalence.

The Rust benchmark retains every token difference and supports `--strict-parity`, which fails after writing the receipt if any sequence differs. No mismatch is silently accepted as a passing gate.

## Batching performance and limits

The final varied-length Rust HTTP burst delivered 512 tokens in 4.029 seconds. The offline vLLM batch delivered the same nominal 512 output tokens in 0.824 seconds. That is a throughput gap, and the measurements use different transports and nonidentical resulting sequences; they do not establish equal-quality throughput.

The candidate partitions each admitted group by prompt length. Eight equal-length requests form one GPU batch and deliver roughly 792–796 tokens/s. Varied lengths produce smaller sequential cohorts. Merging independently prefetched sequence state for shared decoding, continuous admission, a paged KV pool and fused recurrent kernels are the next throughput work. The current implementation has explicit context and batching bounds, but no production GPU-memory admission profile or preallocated paged pool.

## GPU baking and deployment boundary

`switchyard-engine-bake` launches the engine in an owned subprocess with a fresh private NVIDIA cache. It records the engine binary identity, model/serving configuration, representative inputs, GPU/driver identity, compute-capability build target and cache size. The weight directory and GPU cache are separate preparation products; generic artifact tools can transport them independently.

This proof is local. Copying a cache directory excludes transfer and extraction. It does not establish cache compatibility on another driver, CUDA build or GPU, nor a complete RunPod deployment. The current CUDA 13.4 build and its dynamic libraries need a separately prepared native runtime artifact before cloud rollout. The agent's direct-stream hydration and bounded inference-weight transport have not been changed by this work.

## Evidence and remaining gates

The [checked receipt](receipts/rust-engine-rtx5090.json) includes the workload IDs, reference output, per-token client timings, full native phase events, copied-cache bake and binary identity. Detailed local files remain under ignored `results/rust-engine`, and the rollout ran in existing tmux session 0.

Five CPU tests pass, including cached causal decoding, independent batch rows, state reset, generated workload ordering/EOS/output limits, invalid admission and per-row cancellation. CPU clippy, rustdoc, the CUDA release build and the wider workspace test suite pass. The GPU endpoint and SSE delivery were exercised by the Rust client.

The F32 numerical contract and BF16 comparison are recorded in the [accuracy document](native-engine-accuracy.md). Remaining production gates include mixed-length batching and GPU-memory admission, native runtime packaging, and a full hydrated cold start. Allocation, network transport, cold filesystem pages and per-kernel compilation timings remain unmeasured in this local native-engine experiment.
