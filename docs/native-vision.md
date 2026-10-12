# Native Qwen3.5 image inference

The native vision module executes Qwen3.5's image encoder and patch merger using Candle tensor operations. It accepts the released Hugging Face vision tensors or a llama.cpp `qwen3vl_merger` GGUF mmproj. An mmproj includes the encoder's patch embedding, learned position embeddings, attention/MLP blocks and final merger; loading only its projection matrices would not produce usable image embeddings.

## Image preparation and model positions

The processor admits bounded PNG/JPEG images and calculates a deterministic 32-pixel aligned size within the caller's fixed pixel budget. The processed image is bounded to 262144 pixels; the default validation budget is 65536. If the processor's soft minimum resolution would exceed the configured maximum, the maximum remains a hard bound. Fixed-point CatmullRom resampling matches the independent Pillow/Torchvision fixture pixels. It normalizes RGB channels with mean 0.5 and standard deviation 0.5, repeats the static frame twice, and orders 16-pixel patches in spatial merge blocks. Four patch outputs merge into one text-width embedding. The released 0.8B model uses twelve vision blocks and produces 1024-wide embeddings.

`qwen35::ModelForCausalLM::prefill_images` replaces already expanded image-placeholder embeddings at explicit sequence offsets. It applies Qwen3.5's interleaved temporal/height/width rotary frequencies. Following text positions advance by the image's longest merged grid dimension, independently of physical sequence length. Decode uses the returned rotary-position delta while retaining the physical KV length. Initial image prefill supports one row; image requests must bypass the text-token prefix cache.

## Numerical validation

[The CPU receipt](receipts/native-vision-mmproj-cpu.json) identifies a released 0.8B F32 mmproj and an actual encoder/merger comparison against the full official Hugging Face vision weights. For a 256x256 RGB coordinate fixture, both implementations produce 64x1024 embeddings; maximum absolute difference is 0.000064 and mean absolute difference is 0.0000014. The initial fixture requires no resize. This validates the encoder, merger, temporal weight mapping and patch order. The [RTX 5090 generation receipt](receipts/native-vision-generation-rtx5090.json) additionally compares the native image/text path with the official full conditional-generation model: all sixteen greedy tokens match, the rotary delta is -56 in both, and maximum absolute argmax-logit difference is 0.000029. The displayed answer is “blue, cyan, purple, pink.” The offline trace deliberately continues beyond EOS to compare the full requested token sequence. HTTP integration requires a separate endpoint receipt. Resize fixtures cover unchanged geometry, downsampling and upsampling; this does not establish every possible JPEG decoder or image size.

The implementation is adapted from Candle's Apache-2.0/MIT Qwen3-VL vision module at commit `c68b24997319b8d76e5a4e775dab197c7f26322a`, checked against the official Transformers Qwen3.5 model and released configuration. The merger uses exact GELU, as prescribed by Transformers; the block MLP uses the model's tanh approximation. Source revisions and artifact identity are retained in the receipt.

## Image serving configuration

An optional `vision` object in the engine configuration selects image serving. Omitting it retains text-only loading. Set `mmproj` to a GGUF path relative to `model_root`, or omit that field to load the original Hugging Face `model.visual` tensors. Text-only prepared safetensors omit those tensors and therefore require a separate mmproj. The vision loader expands packed mmproj weights once into dense serving tensors; it currently does not execute a packed vision encoder.

```json
"vision": {
  "mmproj": "mmproj-F32.gguf",
  "max_encoded_bytes": 8388608,
  "max_source_pixels": 16777216,
  "max_pixels": 65536,
  "max_images": 2
}
```

The HTTP chat endpoint accepts string content and ordered `text`/`image_url` parts. Images must use inline PNG or JPEG base64 data URLs; remote downloads and video are unsupported. The fixed pixel budget controls resolution. Image bytes and dimensions are bounded before queue admission, and `/tokenize` returns the expanded image-placeholder count used for context and usage accounting. Requests exceeding the configured context plus output allowance fail before GPU work.

The inference owner decodes image bytes, computes embeddings and begins an independent multimodal sequence. Text requests retain existing tensor batching. Multiple image requests are accepted in the same bounded queue, but their model sequences currently execute individually. Image generation bypasses text-prefix caching, retains its own rotary-position delta, and uses the ordinary streaming, cancellation and completion frames. Full HTTP and GPU image generation acceptance remains pending the integrated test receipt.

## Offline encoder probe

Run `cargo run -p switchyard-engine --example vision_probe -- MODEL_ROOT MMPROJ IMAGE OUTPUT_DIR`. Use `-` for MMPROJ to read the original Hugging Face vision tensors. With the `cuda` feature, append `cuda` to use device zero. The probe writes patch and embedding NPY arrays and timing JSON for comparison with an independent oracle. It does not allocate cloud machines or route requests.
