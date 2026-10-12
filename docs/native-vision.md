# Native Qwen3.5 image inference

The native vision module executes Qwen3.5's image encoder and patch merger using Candle tensor operations. It accepts the released Hugging Face vision tensors or a llama.cpp `qwen3vl_merger` GGUF mmproj. An mmproj includes the encoder's patch embedding, learned position embeddings, attention/MLP blocks and final merger; loading only its projection matrices would not produce usable image embeddings.

## Image preparation and model positions

The processor admits bounded PNG/JPEG images and calculates a deterministic 32-pixel aligned size within the caller's fixed pixel budget. It normalizes RGB channels with mean 0.5 and standard deviation 0.5, repeats the static frame twice, and orders 16-pixel patches in spatial merge blocks. Four patch outputs merge into one text-width embedding. The released 0.8B model uses twelve vision blocks and produces 1024-wide embeddings.

`qwen35::ModelForCausalLM::prefill_images` replaces already expanded image-placeholder embeddings at explicit sequence offsets. It applies Qwen3.5's interleaved temporal/height/width rotary frequencies. Following text positions advance by the image's longest merged grid dimension, independently of physical sequence length. Decode uses the returned rotary-position delta while retaining the physical KV length. Initial image prefill supports one row; image requests must bypass the text-token prefix cache.

## Numerical validation

[The CPU receipt](receipts/native-vision-mmproj-cpu.json) identifies a released 0.8B F32 mmproj and an actual encoder/merger comparison against the full official Hugging Face vision weights. For a 256x256 RGB coordinate fixture, both implementations produce 64x1024 embeddings; maximum absolute difference is 0.000064 and mean absolute difference is 0.0000014. The initial fixture requires no resize. This validates the encoder, merger, temporal weight mapping and patch order. It does not establish complete HTTP image generation or arbitrary image resizing parity.

The implementation is adapted from Candle's Apache-2.0/MIT Qwen3-VL vision module at commit `c68b24997319b8d76e5a4e775dab197c7f26322a`, checked against the official Transformers Qwen3.5 model and released configuration. The merger uses exact GELU, as prescribed by Transformers; the block MLP uses the model's tanh approximation. Source revisions and artifact identity are retained in the receipt.

## Offline encoder probe

Run `cargo run -p switchyard-engine --example vision_probe -- MODEL_ROOT MMPROJ IMAGE OUTPUT_DIR`. Use `-` for MMPROJ to read the original Hugging Face vision tensors. With the `cuda` feature, append `cuda` to use device zero. The probe writes patch and embedding NPY arrays and timing JSON for comparison with an independent oracle. It does not allocate cloud machines or route requests.
