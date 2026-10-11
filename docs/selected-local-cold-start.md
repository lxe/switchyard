# Selected local cold-start rollout

The current CPU snapshot and final tensor rollout is documented in [CPU snapshot and final tensor cold start](cpu-snapshot-local-cold-start.md).

This historical Qwen3.5 0.8B rollout produced its first streamed content, `Paris`, **38.988 seconds after local container launch**. The streaming request itself took **81.4 ms** after an internal inference call had already warmed the engine. The 81.4 ms figure is therefore a warm request measurement. This was one successful local RTX 5090 rollout through the actual Switchyard agent, with timestamped native startup instrumentation. The [sanitized timing receipt](receipts/selected-local-cold-start.json) retains the identities, decisions, and measured operations.

## Measurement conditions

The container and installation were fresh. Boot, build, model, and compiler-cache artifacts came from a local HTTP fixture; this measures local hydration rather than R2 delivery or provider allocation. The base container image was already present, and the host page cache was not flushed. Sampling compilation, cache packing, and agent building occurred before the measured clock. Native startup instrumentation was enabled; its overhead was not measured separately.

The fixture used BF16, eager execution, a 64-token context, one sequence, 64 maximum batched tokens, a 0.25 GPU memory fraction, and the Triton attention backend. The GPU was an RTX 5090 with driver 615.71.09. These small eager-engine settings do not establish production startup or serving throughput.

Early launch was enabled. The model, build component, and compiler cache hydrated concurrently with boot. Boot was the last dependency to finish, so the gate was already releasable when Python started; this run did not observe imports overlapping unfinished supplemental dependencies. No artificial component delay was applied.

## Startup timeline

The following times are seconds after the host started container launch. Native Python timestamps use the shared Linux monotonic clock. Agent milestones use a pre-exec clock estimate with a 10 ms rounding bound; artifact polling observations are distinguished from native durations in the receipt.

| Event | Time or duration |
| --- | ---: |
| Container launch returned | 3.758 s |
| Model metadata ready | 4.579 s |
| Model hydration finished | 7.445 s |
| Boot and all dependencies ready | 11.011 s |
| Agent released engine prerequisite gate | 11.012 s |
| Torch preparation | 0.717 s; finished at 11.773 s |
| Transformers preparation | 0.515 s; finished at 12.288 s |
| vLLM serving module preparation | 2.527 s; finished at 14.815 s |
| vLLM GPU worker module preparation | 0.433 s; finished at 15.249 s |
| Strict native sampling provider accepted | 15.959 s |
| Native process guard selected spawn | 18.445 s |
| First tensor delivered | 23.742 s |
| All tensors assigned | 24.551 s |
| Native memory-plan reuse declined | 25.258 s |
| GPU model profiling | 6.986 s |
| Native model warmup | 0.531 s |
| Native API startup log observed | 33.800 s |
| Agent inference verification finished | 38.877 s |
| Streaming request submitted | 38.906 s |
| First streamed content | 38.988 s |
| Completed response validated as `Paris` | 38.995 s |
| Owned container removed | 50.293 s |

Operations can nest and overlap. In particular, `Worker.load_model` and `GPUModelRunner.load_model` describe the same loading interval at different call levels; their durations must not be added.

## Artifact reception and weights

Each download and extraction duration below is measured from that artifact's own reception start, rather than from container launch. Download and extraction run concurrently, so extraction completion includes time spent waiting for arriving compressed data.

| Artifact | Network bytes | Download complete | Extraction complete |
| --- | ---: | ---: | ---: |
| Boot runtime | 4,562,590,833 | 3.739 s | 6.662 s |
| Build component | 682,116,732 | 1.946 s | 4.244 s |
| Native compiler cache | 19,890,845 | 0.047 s | 0.228 s |

The model received 1,769,980,465 bytes. The extent loader delivered 488 tensors totaling 1,746,882,752 bytes and completed GPU assignment at agent elapsed 20.800 seconds. Its recorded pinned-buffer staging time was 0.614 seconds and DMA time was 0.077 seconds; these counters are not additive phase durations. The native worker's complete model-loading operation took 2.303 seconds.

## Process and cache decisions

The policy requested guarded fork, but the actual worker used **spawn**. Importing `flashinfer.gdn_kernels.gdn_decode_bf16_state` called `torch.cuda.get_device_properties` at module initialization and initialized CUDA in the API process. vLLM's existing safety guard detected that state and selected spawn. The worker independently reported `actual_start_method="spawn"` and `fork_inheritance=false`; no native guard was bypassed.

The native sampling binary was compiled and executed on the RTX 5090 before this run, then registered through FlashInfer's provider API for `sm120f`. Required-cache policy accepted it with FlashInfer JIT disabled. Eager configuration selected `CompilationMode.NONE`, so Torch AOT files were not applicable. This does not establish complete compiled-mode cache acceptance.

Other native kernel files originated in the earlier diagnostic cache. Their provenance remains in the cache manifest, which explicitly accepts them for this local diagnostic adapter without presenting them as a rebake under a new production identity. vLLM declined the previous memory plan and performed native GPU profiling; its free-memory baseline and fingerprint checks were retained.

## Agent readiness verification

In this historical run, agent inference verification followed the native API startup observation by 5.077 seconds. That agent build polled health at one-second intervals, then required successful tokenization and a non-streaming one-token decode before reporting ready. This check preceded the measured streaming request; the benchmark did not add another tokenization request. Current agent builds remove both inference checks and poll HTTP availability every 100 ms. The first client request now pays any remaining first-decode initialization cost.

Two CUDA driver-cache files were written during that interval, at 15:45:27.947 and 15:45:31.556 UTC. Those writes are consistent with first-decode initialization contributing to the interval. The recorded spans do not separate the exact costs of health polling, tokenization, and the first decode, so the whole interval cannot be attributed to one operation. Current builds emit `engine_health_polling_started` and `engine_health_ready`; they do not emit tokenization or verification-decode milestones.

## Reproduction and evidence

The prepared local inputs are selected by [the runnable fixture script](../crates/switchyard-weights/benchmarks/run_selected_cold.sh). It accepts an optional output directory and otherwise creates a timestamped directory. It preserves console output with `tee`, propagates failures with `pipefail`, and removes its owned container. The script uses existing local artifacts; it does not allocate a cloud GPU.

Set `COLD_START_PARTITION` to the prepared runtime-set directory, then run the script from the checkout containing the local fixture inputs:

```sh
COLD_START_PARTITION=/path/to/runtime-set \
  bash crates/switchyard-weights/benchmarks/run_selected_cold.sh
```

The completed run is under `results/engine-startup/05-selected-cold-rollout/extents`. Its `timeline.jsonl` contains 2,431 events, `startup-profile.jsonl` contains 1,888 native events, and `engine-startup.jsonl` contains 18 structured decisions. `receipt.json` preserves the final agent state and object-transfer receipts. The console is `scratch/engine-startup/selected-cold-console-final.log`.

All owned rollout containers were removed.
