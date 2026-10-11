# Native engine numerical accuracy

The Qwen3.5 text adapter passes a numerical comparison against the released Transformers implementation in F32. On the RTX 5090, all 512 teacher-forced next-token decisions agree, and all 16 complete single-call and 16 complete batched sequences agree. Teacher forcing means that both engines consume the same prompt and continuation tokens, so a difference at one position cannot change the input to later comparisons.

The checked-in [receipt](receipts/rust-engine-accuracy-rtx5090.json) records inputs, generated token IDs, comparisons, configuration, reference versions, binary SHA-256 and fresh-process HTTP timings. The test uses 16 synthetic chats, 32 output tokens and a context limit of 128. It deliberately ignores EOS to exercise longer continuations, including positions that an ordinary completion would not reach. This corpus validates implementation behavior; it does not establish broad model quality or long-context accuracy.

## Numerical contract

| Comparison | Result |
|---|---:|
| F32 teacher-forced next-token choices against Transformers | 512 / 512 |
| Mean absolute expected-token log-probability error | 0.000000974 |
| Maximum absolute expected-token log-probability error | 0.000025894 |
| F32 complete single-call sequences against Transformers | 16 / 16 |
| F32 complete equal-length batched sequences against Transformers | 16 / 16 |
| BF16 teacher-forced next-token choices against vLLM | 507 / 512 |
| BF16 complete single-call sequences against vLLM | 11 / 16 |
| BF16 complete batched sequences against vLLM | 11 / 16 |

The F32 gate requires identical next-token choices and a maximum expected-token log-probability error of 0.0001. The BF16 comparison is reported without a passing tolerance gate. Strict BF16 sequence parity with vLLM still fails, and the five remaining teacher-forced differences have reference top-two logit margins between zero and 0.25. vLLM itself produces different complete single versus batched sequences on five cases in the original receipt.

Against the same F32 reference, the candidate BF16 mean absolute expected-token log-probability error is 0.01254, compared with 0.01484 for vLLM BF16. Maximum errors are 0.39704 and 0.27758 respectively. These measurements show the scale of reduced-precision differences on this corpus; the lower candidate mean does not establish superior quality. Matching the mathematical reference in F32 also does not prove bitwise equivalence between different BF16 kernels.

## Arithmetic and sampling

Greedy sampling selects the lowest vocabulary ID when logits tie, matching PyTorch. Candle's CUDA argmax reduction can instead retain a higher ID from a lower thread lane. The native sampler performs a maximum reduction and a minimum-ID reduction over the tied entries, with reusable GPU index and sentinel tensors. A regression test includes ties across CUDA reduction lanes.

The feed-forward activation computes SiLU and its product with the up projection in F32 before rounding once to the model dtype. Residual normalization uses the F32 sum of the two residual streams before rounding the retained residual. Gemma-style normalization weights are adjusted once at model load, and the normalization uses Candle's fused F32 kernel. A BF16 regression case distinguishes normalization before and after residual rounding.

Full attention keeps attention scores and softmax accumulation in F32. Its value product uses rounded unnormalized exponentials, with the F32 denominator applied afterward. The recurrent state remains F32. These choices preserve the reference operations without adding CPU logit transfers to serving.

## Reference implementation

The mathematical oracle uses Transformers 5.17.0 `Qwen3_5ForCausalLM`, eager attention and its PyTorch recurrent-attention fallback, with Torch 2.13.0+cu129. TF32 is disabled. The model loads only the released language tensors; the tied output projection shares the embedding weights. Nonpersistent rotary buffers are explicitly reconstructed after checkpoint loading and checked for the expected first inverse frequency. Layer inspection covers embeddings, decoder outputs and the first full-attention layer's projections and rotary outputs.

The model's architecture and released checkpoint are described by the [official Qwen model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [Transformers model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5). vLLM 0.31.0+cu129 remains a separate serving comparison. Python and Torch are used only by these offline reference scripts; the candidate endpoint remains native Rust.

Run `benchmarks/hf_reference.py` with the source model mounted at `/model` and the token requests at `/output/requests.json`. Run `benchmarks/hf_teacher_reference.py` with `/output/traces.json`, whose rows contain `prompt` and `continuation` token arrays. These scripts produce `hf.json`, `teacher-hf.jsonl` and `layers-hf.json` in `/output`. They require the pinned reference environment described above.

Generate the candidate diagnostics and enforce the F32 contract:

```sh
switchyard-engine --config f32-config.json probe \
  --input traces.json --output teacher-rust.json --layers-output layers-rust.json
python crates/switchyard-engine/benchmarks/compare_teacher.py \
  --candidate teacher-rust.json --reference teacher-hf.jsonl \
  --output comparison.json --require-argmax --max-logprob-error 0.0001
```

`probe` performs full-vocabulary CPU diagnostics offline. Serving never calls it. The comparison rejects missing or duplicate positions and mismatched histories, writes the receipt, and exits unsuccessfully when an explicit numerical contract fails.

## Local GPU cache and startup

The measured candidate binary SHA-256 is `fd4e3314bf1545e10319d8abbf96b100eb8790beb015c614415c39ddca3a00d8`, built with CUDA 13.4 for SM120 and tested on driver 615.71.09.

| Cache availability | Listener | Launch to first streamed token | Eight-call burst output tokens/s |
|---|---:|---:|---:|
| Empty private driver cache | 2.704 s | 11.855 s | 877 |
| Reused private driver cache | 0.705 s | 0.843 s | 844 |
| Separate offline bake | 0.746 s | 0.879 s | 884 |

The offline bake created ten cache files totaling 15,983,431 bytes in 14.758 seconds, including its 16-request generation workload. Each startup row uses a fresh serving process with resident prepared model files and warm filesystem pages. These are single samples, not allocation or network-hydration measurements. Cache reuse on another GPU architecture or driver requires its own platform bake and validation.
