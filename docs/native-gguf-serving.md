# Native Qwen3.5 GGUF execution

The native engine can execute Qwen3.5 text models from existing GGUF quantizations. It uses the same recurrent attention, paged attention, prefix retention and mixed-length decode implementation as the safetensors adapter. GGUF is an explicit `weights` configuration, so a directory containing both formats does not select a format implicitly.

```json
"weights": { "kind": "gguf", "file": "Qwen3.5-0.8B-Q4_K_M.gguf" }
```

The model directory also contains the source checkpoint's `config.json`, `tokenizer.json` and `chat_template.jinja`. `file` is a relative path inside that directory. The loader checks GGUF architecture, layer types, head dimensions, feed-forward dimensions, recurrent dimensions, RoPE settings and normalization epsilon against the companion configuration. It requires F32 activations. Weight byte limits apply to the selected GGUF file; serving performs structural checks without content hashing.

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

The first local 0.8B Q4_K_M and Q8_0 GPU executions completed 16 requests and 512 generated tokens each. Independent llama.cpp comparisons found numerical and greedy-sequence differences. They are diagnostic evidence, not an exact-parity acceptance. Final packed-embedding execution and reference attribution remain under validation. The 2B/4B examples currently specify the intended fixed configuration and do not claim a completed GPU gate.

## Existing implementation references

The model layout contract derives from [llama.cpp's Qwen converter](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/conversion/qwen.py) and [Qwen3.5 graph](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/src/models/qwen35.cpp), under MIT. Packed arithmetic and row lookup use the existing [Candle quantized backend](https://github.com/huggingface/candle/tree/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/quantized), under Apache-2.0 OR MIT. No dependency revision was changed.
