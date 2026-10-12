# Native Qwen3.5 GGUF execution

The native engine can execute Qwen3.5 text models from existing GGUF quantizations. It uses the same recurrent attention, paged attention, prefix retention and mixed-length decode implementation as the safetensors adapter. GGUF is an explicit `weights` configuration, so a directory containing both formats does not select a format implicitly.

```json
"weights": { "kind": "gguf", "file": "Qwen3.5-0.8B-Q4_K_M.gguf" }
```

The model directory also contains the source checkpoint's `config.json`, `tokenizer.json` and `chat_template.jinja`. `file` is a relative path inside that directory. The loader checks GGUF architecture, layer types, head dimensions, feed-forward dimensions, recurrent dimensions, RoPE settings and normalization epsilon against the companion configuration. It requires F32 tensor activations; the pinned Candle CUDA packed matmul kernels internally quantize those activations to Q8_1. The GGUF factory uses the file's `tokenizer.ggml.eos_token_id`; it does not inherit a different EOS ID from the companion HF configuration. Weight byte limits apply to the selected GGUF file; serving performs structural checks without content hashing.

## Packed weights and model layout

GGUF per-tensor F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K and Q6_K encodings are supported by the loader. A filename such as Q4_K_M describes a mixed quantization policy: the released Unsloth 0.8B file also contains Q5_K recurrent matrices, Q6_K embeddings/output and Q8_0 gate matrices. Unsupported IQ/Hadamard variants fail admission.

Quantized projections retain packed storage and execute Candle's quantized operators. The embedding table uses packed row lookup and the tied output projection shares its storage. Dense auxiliary vectors and convolution kernels are loaded independently. The adapter does not materialize a full dense copy of quantized projection matrices or the vocabulary embedding table.

llama.cpp's converter folds the HF normalization `+1`, stores recurrent decay as `-exp(A_log)` and rearranges value heads from grouped to tiled order. The adapter consumes the folded norm and negative decay directly. It restores the original head order through activation permutations and small auxiliary tensor permutations; packed matrix rows and columns do not need a dense conversion.

## Configuration examples and validation scope

Examples are checked in for [0.8B Q4_K_M](../examples/native-qwen35-0.8b-gguf.json), [0.8B Q8_0](../examples/native-qwen35-0.8b-gguf-q8.json), [2B Q4_K_M](../examples/native-qwen35-2b-gguf.json) and [4B Q4_K_M](../examples/native-qwen35-4b-gguf.json). Model artifacts are not checked in.

Pinned existing quant sources:

| Model | Repository | Revision |
| --- | --- | --- |
| 0.8B | [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` |
| 2B | [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | `f6d5376be1edb4d416d56da11e5397a961aca8ae` |
| 4B | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | `e87f176479d0855a907a41277aca2f8ee7a09523` |

CPU tests execute a complete synthetic hybrid GGUF model through prefill, decode and mixed rows, including a 2:1 value/key-head ratio. These tests complement the existing dense model and prefix-cache tests; they do not establish released-model quality.

The [RTX 5090 receipt](receipts/native-gguf-local-rtx5090.json) records real execution for all four examples: each completed 16 requests and 512 generated tokens, including mixed prompt lengths and eight-row decode batches. Process startup used warm files and driver caches. These timings do not measure network hydration or establish a controlled cold-start improvement.

| Existing quant | Model loaded | Warm eight-row / 32-token decode batch | llama.cpp teacher-forced top-1 matches | Mean / maximum expected logprob difference |
| --- | ---: | ---: | ---: | ---: |
| 0.8B Q4_K_M | 0.500 s | 0.233 s | 254 / 256 | 0.0260 / 0.8025 |
| 0.8B Q8_0 | 0.542 s | 0.227 s | 253 / 256 | 0.0086 / 0.0809 |
| 2B Q4_K_M | 0.640 s | 0.263 s | 254 / 256 | 0.0126 / 0.1247 |
| 4B Q4_K_M | 0.911 s | 0.436 s | 256 / 256 | 0.0078 / 0.1495 |

These comparisons use the exact same GGUF artifact in both engines, F32 llama.cpp K/V state and explicit token prompts. They establish functional execution and expose numerical differences. They do not pass exact generation parity or the unchanged `1e-4` dense logprob gate. The original free-generation comparison is retained in the receipt: only 3 / 16 complete sequences matched, and two reference sequences stopped at EOG despite its `ignore_eos` flag while the native diagnostic forced 32 tokens. That stopping-contract mismatch further limits interpretation of the raw complete-sequence count.

An independent `gguf-py` decode plus Transformers eager F32 CPU oracle covers 256 positions for the 0.8B Q4_K_M artifact. Native packed CUDA matches 252 / 256 top-1 positions, with mean expected logprob difference 0.02150 and maximum 0.89764. An offline diagnostic dense expansion of the same decoded weights lets the native model execute F32 CPU arithmetic: it matches all 256 top-1 positions, with mean difference `3.46e-5` and maximum `1.56e-4`. This separates model/layout arithmetic from additional packed-kernel activation quantization. The dense diagnostic also exceeds the unchanged `1e-4` logprob threshold at some positions; no threshold was widened.

The independent oracle is [gguf_hf_reference.py](../crates/switchyard-engine/benchmarks/gguf_hf_reference.py). It can export dense weights only for offline diagnostics. That export is not a serving fallback, hydration artifact or bake format. Projection and vocabulary weights remain packed in the native GGUF serving path.

Remaining work includes a quantized numerical/performance acceptance policy, comparing kernel precision across platforms, and validating additional tensor encodings on real full models. Synthetic projection tests cover Q4_0, Q8_0, Q4_K, Q5_K and Q6_K; real model receipts cover the mixed encodings present in the selected Q4_K_M and Q8_0 files. GGUF execution is available, while exact-parity production admission remains unproven.

The [image HTTP receipt](receipts/native-vision-http-rtx5090.json) tests the final integrated binary after the EOS correction. It covers Q4_K_M plus F16 mmproj on all three sizes and Q8_0 plus F16 mmproj on 0.8B, including concurrent requests, usage accounting and a following text request. These functional checks do not change the quantized numerical acceptance limits above.

## Existing implementation references

The model layout contract derives from [llama.cpp's Qwen converter](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/conversion/qwen.py) and [Qwen3.5 graph](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/src/models/qwen35.cpp), under MIT. Packed arithmetic and row lookup use the existing [Candle quantized backend](https://github.com/huggingface/candle/tree/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/quantized), under Apache-2.0 OR MIT. No dependency revision was changed.
