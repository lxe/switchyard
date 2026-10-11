# Native quantized projections and decode batching

`switchyard-quantized` provides GGUF header admission, packed projection groups and a bounded decode-row plan for the existing native Rust engine. It has no request server, provider adapter or model forward pass. The current Qwen model adapter continues to use its accepted dense weights and arithmetic. GGUF inspection does not authorize model serving.

## GGUF format admission

`gguf::Index::read` indexes little-endian GGUF version 3 without reading tensor payloads. Each tensor descriptor retains its name, GGML encoding, dimensions, absolute payload offset and packed byte count. GGML dimensions list the contiguous input width first: `[256, 512]` becomes a Candle matrix shape `[512, 256]`. Reversing this dimension list identifies the same row-major bytes; it does not transpose the data.

The initial encodings are F32, F16, BF16, Q4_0, Q8_0, Q4_K and Q6_K. Unsupported GGML type codes fail admission. A label such as Q4_K_M describes a file's mixture of tensor encodings; it does not replace the type of each tensor. The parser verifies row-level quantization block boundaries, positive dimensions, arithmetic overflow, tensor alignment, file bounds, duplicate names and nonoverlapping payloads. Big-endian data, versions 1/2 and split-file resolution are outside this foundation.

Default indexing limits are 64 MiB of wire header, 1 MiB per string, 10,000 metadata entries, 100,000 tensor descriptors, 2,000,000 metadata values and four array nesting levels. Counts and declared lengths are checked before allocation. These are representation bounds, not a measured 64 MiB RSS guarantee: strings, enums and collection nodes add overhead. Persistent caching and GPU transfer buffer ownership remain responsibilities of the existing artifact loader.

`switchyard-gguf-inspect <file>` reports the descriptors and Qwen35 recurrent layout requirements. It checks `general.architecture` rather than trusting the file name. Its successful exit means the inspected format and recurrent shapes passed; its report still sets `native_serving_admitted` to false.

## Quantized projection groups

`projection::ProjectionGroup::read` concatenates the packed output rows of up to sixteen matrices with the same encoding and input width. It requires an explicit aggregate packed-byte cap and reads only those selected payloads. Candle construction may retain an additional packed copy. The constructor uses `QMatMul::QTensor` explicitly, so `CANDLE_DEQUANTIZE_ALL` cannot silently replace these weights with a full dense matrix.

`forward` accepts contiguous F32 `[rows, width]` or `[batch, tokens, width]` input. One packed matmul produces the combined output; the returned slices preserve the requested projection order. This is suitable for candidate same-input projections such as feed-forward gate/up. It does not fuse their activation or product, change accumulation rules, or combine projections with different input tensors.

CPU fixtures compare grouped outputs with separate packed calls for Q4_0, Q8_0, Q4_K and Q6_K at a maximum absolute tolerance of `1e-5`. Matrix reconstruction has separate fixed RMSE bounds of 0.08, 0.006, 0.10 and 0.02 on the declared synthetic sine weights. These bounds establish local block behavior, not model quality. An independent hand-encoded Q4_0 fixture checks scale, nibble and row order. Its packed CPU dot oracle accounts for Candle's FP16 Q8_0 activation scale.

The offline benchmark compares individual-row calls, tensor-batched separate projections and one grouped projection using identical packed matrices and input rows. It warms the operations, alternates measurement order, records every sample and reports medians. It checks both single-versus-batch and grouped-versus-separate outputs before timing. Results are CPU projection latency, not tokens per second or model throughput.

```sh
cargo test -p switchyard-quantized --all-targets
cargo clippy -p switchyard-quantized --all-targets -- -D warnings
cargo build --release -p switchyard-quantized
RAYON_NUM_THREADS=4 target/release/switchyard-quantized-bench \
  --encoding-code 2 --rows 1,8,64 --iterations 20 --samples 5
```

No GPU feature is enabled for this foundation's CPU tests or measurements. GPU integration requires the main thread's serialized validation slot, a numerical contract against the released model and a comparable serving workload.

## Qwen Gated DeltaNet weight layout

The official llama.cpp Qwen conversion changes more than tensor names. It tiles V heads where released HF weights group them under each K head. This affects the V portion of QKV rows, z/b/a rows, dt/A vectors, V convolution channels and output-projection columns. `gdn::HeadOrder` computes the forward and inverse channel mapping without allocating a permutation tensor. CPU fixtures cover ratios 1, 2 and 3, including the 16-key/48-value-head configuration.

The converter also stores `ssm_a` as `-exp(A_log)` and folds the HF `+1` adjustment into non-GDN RMS norm weights. Applying the native HF preparation again would change the model. `Qwen35Requirements::inspect` checks the recurrent metadata and required tensor shapes, including equal K/V head widths, and exposes these remaining conversions. Quantized vector, norm and convolution representations are rejected by this initial recurrent contract.

Integration must either bake a declared native layout or teach the existing adapter to execute the declared GGUF layout. Whole packed rows can be permuted without new quantization; arbitrary column permutations can cross quantization blocks and require a defined activation mapping or a separately validated conversion. A file passing shape checks does not establish head order, norm semantics, tokenizer identity, or gate activation. Full-attention layouts, MRoPE, MTP, folded transforms and model numerical gates remain required.

Qwen3.8-27B is a released checkpoint at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its published configuration uses `qwen3_5_text`, 64 layers, hidden width 5120, 16 key heads, 48 value heads, a `swish` output gate and untied embeddings. Those fields must be checked explicitly; sharing an architecture string with Qwen3.5 does not establish compatibility. This foundation does not load that checkpoint or measure its quality or fit.

## Merged decode after varied-length prefill

The larger throughput gap is decoding requests with different prompt lengths together. The current `ModelSession::forward(tokens, position)` supplies one scalar position and owns one cohort's state, so equal-length groups cannot simply be concatenated after prefill. Quantized projection grouping does not solve that gap.

`decode_plan::DecodePlan` represents the proposed model handoff: each row carries its request ID, next token, absolute position, remaining output budget and a state lease with slot and generation. It validates row capacity, unique requests and slots, and per-row context bounds. `advance` retains independent positions and returns completed leases while preserving the remaining row order. The adapter must settle CUDA work before recycling a returned lease and reject stale generations. These plans do not own or claim to move GPU state.

The integration path keeps the current server and scheduler. First prefill each compatible prompt cohort, retain its convolution/recurrent state and attention KV ranges under individual leases, and collect the first emitted tokens. Then add an adapter method accepting `DecodePlan::rows()` with one token per row and each row's own position. The adapter gathers/scatters GDN state by lease and uses per-row RoPE positions and valid KV lengths for full attention. Finished or cancelled rows leave the next plan; new prefills may join only after their state and first-token boundary are complete. Paged KV storage and continuous admission can follow this correct state interface.

Before enabling merged decode, compare independent sessions with mixed prompt lengths 8/32/96, staggered arrivals, different output budgets, EOS, cancellation and reused slots. Use repeated and unique prompts. Preserve request order, compare teacher-forced logits at the existing F32 maximum log-probability error of `0.0001`, and require identical next-token choices. Reduced-precision modes need their own declared gate. Benchmark total output tokens/s, per-request first-token and decode latency, active rows, queue wait and memory with identical transport and sequences. A cohort-only or projection microbenchmark cannot establish this serving result.

## Official sources

- [GGUF specification](https://github.com/ggml-org/ggml/blob/ffa4e8b80930029a35991f94e7c8a93cd67730ab/docs/gguf.md), revision `ffa4e8b80930029a35991f94e7c8a93cd67730ab`.
- [llama.cpp Qwen converter](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/conversion/qwen.py), [Qwen35 model](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/src/models/qwen35.cpp), [batch allocation](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/src/llama-batch.cpp), and [CUDA MMQ](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/ggml/src/ggml-cuda/mmq.cu). Its explicit sequence positions and recurrent state inform the handoff above; its backend performance is not assumed for Candle.
- [Candle quantized operations](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/quantized/mod.rs) and [GGUF reader](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/quantized/gguf_file.rs), the revision already used by the native engine. That checkout has no quantized Qwen3.5 model module; this foundation reuses its operations rather than introducing another model.
- [Released Qwen Transformers operations](https://github.com/huggingface/transformers/blob/55c3cc9f03cd4fbf2d27d382dcc8fcc544631f8e/src/transformers/models/qwen3_5/modeling_qwen3_5.py) and [Qwen3.8-27B pinned configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json).
