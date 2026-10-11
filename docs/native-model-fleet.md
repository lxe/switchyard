# Native Qwen model configurations

The native text engine uses the three checkpoints in [`configs/engine/models.json`](../configs/engine/models.json). Each entry records the source repository, immutable revision, required source files and serving configuration. These are text-only adapters; vision tensors are excluded during preparation, and neither image inputs nor vision projectors are served.

| Model | Source revision | Context bound | Batch bound | Host weight buffer limit |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5 0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | 128 | 8 | 4 GiB |
| Qwen3.5 2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 128 | 8 | 8 GiB |
| Qwen3.5 4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 128 | 8 | 12 GiB |

These limits bound context, batch size and aggregate host-side checkpoint buffers. They do not specify a hard GPU allocator cap or reserve a paged KV pool. All three RTX 5090 configurations use BF16, a 64-job queue and a 2 ms batch collection interval. Platform artifact manifests must also identify the engine binary, CUDA target and compatible driver cache before cloud deployment.

## Source preparation

Download the listed source files at the pinned revision into the corresponding `data/qwen3.5-*` directory. Source data and generated artifacts are ignored by Git. With the Hugging Face command-line client, for example:

```sh
hf download Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --local-dir data/qwen3.5-2b

target/release/switchyard-engine \
  --config configs/engine/qwen3.5-2b-rtx5090.json \
  prepare --output data/qwen3.5-2b-text
```

Preparation writes a fresh text-only safetensors checkpoint and the tokenizer/configuration files. It preserves dense tensor names and source precision; it does not quantize or bake CUDA driver state. Pass `--model-root data/qwen3.5-2b-text` during serving, or use an artifact-specific configuration with that hydrated path. The generic artifact transport and platform-specific GPU cache remain separate from the model adapter.

## Numerical and deployment acceptance

The [0.8B accuracy receipt](native-engine-accuracy.md) establishes F32 equivalence with the released Transformers model on the recorded workloads. BF16 comparison against vLLM remains a separate reduced-precision test and does not currently pass complete sequence equality. The [2B and 4B receipt](receipts/native-qwen2b4b-rtx5090.json) records the same mathematical, single/batch and real SSE tests on the larger checkpoints. Both match all 16 complete F32 sequences in each execution mode, and all 128 teacher-forced argmax decisions. Maximum expected-token log-probability errors are 0.000004555 for 2B and 0.000023910 for 4B, below the 0.0001 acceptance bound. BF16 complete sequences match vLLM on 15/16 single and 16/16 batched cases for 2B, and 14/16 single and 12/16 batched cases for 4B. These differences remain reported rather than treated as passing strict BF16 parity.

A local process-start measurement with existing model files is distinct from a RunPod allocation and R2 hydration measurement. Cloud receipts must include allocation, artifact receipt/publication, GPU initialization, tensor loading, listener availability and the client's first streamed content. Driver-cache baking happens offline on a compatible platform; serving adds no inference readiness request or automatic memory sizing.

## Local baked-cache startup

All three checkpoints have been served on the RTX 5090 with the corrected CUDA 13.4 engine, driver 615.71.09 and separately baked driver caches. Each measurement launches a fresh process with resident prepared model files and warm filesystem pages. The burst contains eight equal-length requests with 32 output tokens each.

| Model | Listener | Launch to first streamed token | Burst output tokens/s |
| --- | ---: | ---: | ---: |
| 0.8B | 0.746 s | 0.879 s | 884 |
| 2B | 1.542 s | 1.677 s | 739 |
| 4B | 3.531 s | 3.679 s | 435 |

The startup values above come from the client clock. Engine telemetry has its own process clock and records CUDA initialization, local file reading, GPU tensor construction and listener publication. The larger checkpoints spend most of their pre-listener interval reading the prepared weight file; Python imports and vLLM engine construction are absent. Cache files contain compiled driver code, not saved GPU tensors, so CUDA allocation and tensor transfer still occur on each launch. These single samples were not isolated from background CPU builds and do not include network hydration, provider allocation or cold filesystem reads.

The 2B and 4B teacher traces each cover eight prompts and 16 continuation positions, while full-sequence tests cover all 16 prompts with 32 generated positions. The receipt preserves literal input and output IDs, source/configuration identities, bake metadata, timeline events and hashes of detailed local evidence. The raw logs remain in ignored `results/rust-engine/three-model-01/` directories. Cloud acceptance still requires independent platform bake and fresh hydration measurements for each model.
