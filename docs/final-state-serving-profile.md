# Final-state tensors and fixed serving profiles

The selected engine loads vLLM's final named parameter and buffer state and allocates a fixed KV budget. HF key mapping, projection fusion, dtype conversion and native sharded-state export run during the GPU bake. Serving uses the resulting tensor names and destination byte layout, without calling the model's HF `load_weights` method.

The initial profile is Qwen3.5 0.8B, BF16, language-only, RTX 5090, eager execution, context 64, one sequence and TP/PP/DP sizes of one. Its KV allocation is explicitly 1 GiB, with a separate 1 GiB runtime-headroom admission requirement. Other GPU/model combinations require their own profile; this initial profile does not measure them.

## Final-state artifact

The preparation helper calls native `ShardedStateLoader.save_model` after native loading, warmup and one representative decode. `switchyard-final-state.json` records the runtime ID, source model content ID, native model class, vLLM/Torch versions, rank, partition size, final tensor names, shapes, dtypes, strides, layout hash and completed post-load bake.

The ordinary raw-extent packer packs these safetensors plus tokenizer/config metadata. Use 4 MiB extents for the bounded transport consumer. Safetensors still supplies byte offsets and lengths; the final-state receipt supplies destination layout and alias filtering. Serving verifies the native initialized model's filtered state against that receipt before copying. The initial path rejects quantization, MoE, noncontiguous state and non-single-worker partitions.

`CompletedWeights.destinations` is a dictionary from final tensor name to the actual contiguous GPU destination tensor. Direct staging writes to its `uint8` view, so no full temporary GPU tensor or second device-to-device assignment is needed. A bounded consumer can use the same dictionary to intersect incoming byte intervals with destination slices and acknowledge chunks after CUDA events. CPU-to-GPU DMA remains necessary.

Native dense CUDA post-load initialization remains in `BaseModelLoader`. Unquantized CUDA linear postprocessing performs no tensor repacking; this contract does not extend that claim to quantized kernels. Native tied-weight alias filtering is shared by export and load.

## Serving profile

`VLLM_ENGINE_STARTUP_POLICY.serving_profile` names a normalized relative file below the selected cache root. The [initial policy](../configs/serving-profiles/qwen3.5-0.8b-rtx5090.policy.json) requires strict native caches and references `compiled/serving-profile.json`. Engine arguments must also set `--kv-cache-memory-bytes 1073741824` (the pinned native CLI alias may be `--kv-cache-memory`).

The profile has schema version 1 and purpose `fixed_serving_budget`. It binds the boot runtime ID, final model content ID, final layout hash, native library builds, GPU name/capability/total memory, dtype, context/batch limits, worker partition, attention backend, KV/cache settings, and native compilation/graph modes. Its bake receipt identifies completed native warmup and a sampling binary/hash in the selected cache.

`Worker.determine_available_memory` checks these identities, the explicit KV value, the sampling binary and current available memory. It returns the fixed budget through native multimodal IPC reservation without invoking `profile_run`, automatic memory sizing or native startup-plan selection. A mismatch fails; it does not silently return to profiling. Actual device/model initialization, KV allocation and native execution warmup still run.

## Preparation and integration

Root dispatches `scratch/engine-startup/bake-production-engine.sh` in existing tmux `0`. It writes native final model files, warmed cache files and `serving-profile.unbound.json` under `results/engine-startup/production-engine-bake`. Callable worker serialization is enabled only in this owned preparation process. It captures no GPU process checkpoint.

Pack `production-engine-bake/model` with `switchyard-weights pack SOURCE OUTPUT --extent-bytes 4194304`. Set the profile's `model_content_id` to that new manifest's content ID, then place the bound file at `cache/compiled/serving-profile.json` and pack the cache. Cache compatibility must use the selected final model content ID, current loader v3 identity and serving arguments.

The loader bundle includes `switchyard_final_state.py` and `switchyard_serving_profile.py`; both participate in its v3 source identity. CPU preparation must import this same bundle. The agent injects `SWITCHYARD_MODEL_CONTENT_ID` into the engine child so the profile validates the actual artifact, including after a CPU process restore.

The single consolidated CUDA rollout belongs to root. Focused CPU checks cover budget admission, layout mismatch, direct named copies and incomplete-state rejection. They do not establish CUDA correctness; that requires the final-artifact rollout.
