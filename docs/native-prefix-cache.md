# Native attention pages and prefix caching

The native Qwen3.5 engine stores attention history in immutable token pages and retains reusable prompt checkpoints in a model-local cache. A checkpoint includes attention pages, convolution history, F32 gated-delta recurrent state, the next absolute position and the logits at that boundary. Caching only attention keys and values would give incorrect continuations for this hybrid model.

## Request execution

Prefix matching uses exact token IDs within one loaded model instance. The engine restores the longest retained prefix and computes the remaining prompt. It saves checkpoints at a fixed interval and at the prompt end. A complete prompt hit reuses its final logits to choose the first output token. HTTP usage still reports the client's full input length.

Within each equal-length prefill cohort, identical prefixes execute once. Different suffixes join subsequent tensor batches. For eight 80-token requests sharing 64 leading tokens, an empty cache computes 64 shared tokens and eight 16-token suffixes: 192 token positions instead of 640. Token counts measure avoided input computation, rather than predicting a proportional wall-time reduction.

Completed or cancelled requests drop their private state. Cache eviction drops the cache's references; it cannot invalidate active readers. State from another model owner is rejected before cached logits can be returned. Model identity includes the loaded weights, device and precision through that owner's lifetime. This text adapter does not support LoRA or multimodal inputs; extending it requires extending the cache namespace and checkpoint contract.

## Attention page storage

Each attention layer holds an ordered list of compact K/V pages per sequence. Full pages share immutable ownership. Appending tokens replaces only the partially filled tail, with at most `page_tokens` worth of copying; existing full pages remain shared. Restoring a cohort and removing rows select page handles instead of padding or gathering complete attention histories. Recurrent and convolution tensors still gather along the batch dimension.

CUDA single-token decode uses an embedded kernel from `switchyard-engine-kernels`. It reads each row's true-length page table directly, supports grouped-query attention, and avoids creating a contiguous copy of the entire history. F32 scores, a global softmax maximum and denominator, and F32 value accumulation preserve the established mathematical contract. BF16 exponentials round before the value product while the denominator retains unrounded exponentials.

Fresh prompt prefill keeps the dense tensor operation. Multi-token cached suffixes and unsupported devices use a portable page-wise attention operation with a causal mask. Its score scratch grows with query length and history length. The CUDA decode operation uses F32 score scratch proportional to batch rows, query heads and the longest history, although K/V storage itself has no mixed-row padding.

Pages use individual Candle allocations. This is not a preallocated slab allocator. Cudarc can use the driver's asynchronous allocation pool where supported; allocation and table preparation still need profiling for larger workloads. New requests continue to wait until the admitted cohort finishes. Continuous admission remains separate work.

## Cache bounds and eviction

The example local configurations use the reference-tested F32 precision and supply these fixed values:

```json
"cache": {
  "page_tokens": 16,
  "checkpoint_tokens": 32,
  "max_bytes": 268435456,
  "max_entries": 16
}
```

`max_bytes` conservatively charges retained tensor storage and token keys. Shared pages count against every checkpoint that references them. Host object metadata is bounded by the entry and context limits but is not included in the byte counter. Active request tensors, temporary kernels and model weights have separate bounds; this is not a hard process-wide GPU allocation ceiling.

Shared prefixes enter a protected least-recently-used segment, capped at half of both retention limits. Individual request endings therefore cannot immediately evict a shared system prompt. New shared workloads demote older protected entries, and ordinary eviction removes unprotected entries first. A checkpoint larger than the entire byte budget is not retained. Zero bytes and zero entries disable prefix retention; active attention still uses pages.

Omitted cache configuration preserves the disabled retention default for existing deployments. A live native model can supply the same object under `engine.native.cache`. Those values participate in its engine fingerprint and GPU bake configuration. Published older runtimes and compiler-cache artifacts remain pinned to their original binary/configuration; enabling this engine requires a fresh compatible bake. Baking owns its build resources independently of inference scheduling.

## Telemetry and validation

The `prefix_prefill` event reports input, computed and reused token positions, retained-hit rows, charged bytes, entries, protected entries and evictions. Phase telemetry remains JSON on stderr. The cache does no serving-time content hashing, inference readiness requests or automatic memory sizing.

CPU fixtures cover divergent readers, exact hits, mixed-row release, foreign model owners, eviction with active readers, bounded shared-prefix protection, partial pages, strided rotary keys and BF16 cached suffix masks. The CUDA operation has separate F32/BF16 dense-reference tests. Released-model numerical and timing results are recorded in the [local receipt](receipts/native-paged-prefix-rtx5090.json).

Run the offline cache benchmark against one loaded model:

```sh
target/release/switchyard-engine-cache-bench \
  --config f32-config.json --requests equal-length-requests.json \
  --samples 3 --output cache-timing.json
```

It alternates uncached timing with an empty cache and its immediate repeat, retaining complete token sequences and first-token times. It excludes model loading, HTTP and queueing. F32 sequence differences fail the benchmark after preserving its receipt. BF16 differences are recorded explicitly; their timing does not establish numerical equivalence or a quality guarantee.

## Local measurements

The RTX 5090 receipt uses eight 80-token prompts sharing 64 leading tokens, with 16 generated tokens per row. Three F32 samples per case alternate the uncached baseline with an empty cache and its immediate repeat. All complete generated sequences match the uncached path. The separate released-model gate checks 52 teacher-forced positions per model against both the previously accepted independent native sessions and the pinned Transformers reference; all 156 decisions match each reference at the unchanged `0.0001` maximum expected-token log-probability error.

| F32 model | Uncached batch output tokens/sec | Empty-cache batch output tokens/sec | Repeated batch output tokens/sec | Repeated first token |
| --- | ---: | ---: | ---: | ---: |
| 0.8B | 512 | 575 | 998 | 1.05 ms |
| 2B | 399 | 460 | 747 | 1.01 ms |
| 4B | 196 | 243 | 305 | 85.80 ms |

The 4B cache holds four checkpoints within 256 MiB. Protected retention reduced its repeated-batch first-token median from 155.59 ms to 85.80 ms under otherwise identical bounds, because all eight rows could reuse their common prefix. It still computes missing suffixes. Empty-cache single calls are slower than uncached calls for all three models: staging and saving checkpoints adds work when no reuse occurs.

BF16 single/batch runs and real HTTP streams complete, but cached versus uncached BF16 sequences differ. In the single 16-token cases, matching positions were 6/16, 15/16 and 2/16 for 0.8B, 2B and 4B. These counts do not measure model quality and are not teacher-forced comparisons; they establish that sequence equivalence is unproven. The F32 numerical result must not be extended to BF16. Local HTTP launch-to-first-content was 1.011, 1.987 and 6.109 seconds for BF16 0.8B, 2B and 4B, using existing model files and a warm driver cache. Those timings exclude cloud allocation and hydration and are not comparable to older cloud receipts.

The focused kernel benchmark also records a regression: at batch one and context 1024, paged attention took 109 microseconds versus 80 microseconds for its dense reference. Paging reduces history copies and enables sharing, but this kernel needs more parallelism for that case. Continuous admission, slab allocation and a BF16 quality contract remain open work.

## Design references

[SGLang's radix cache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) implements longest-prefix matching, separate protected references and evictable state, and page-aligned allocation. This engine uses a bounded checkpoint map and immutable ownership rather than importing that runtime. A radix index can replace the current map if lookup cost becomes significant.

[llama.cpp's sequence memory interface](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h) separates sequence copy, removal and position operations. Its [KV implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp) provides the corresponding storage behavior. Stable sequence ownership and shared history are also explicit in this engine's state contract.

[Transformers' prefix-cache example](https://huggingface.co/docs/transformers/main/kv_cache#prefill-a-cache-prefix-caching) preserves a prepared prefix and creates separate continuations. The native engine shares immutable attention pages while retaining the additional recurrent state required by Qwen. [vLLM's hybrid cache design](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/) likewise treats attention and recurrent layers as different state groups.
