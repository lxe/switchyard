# Raw model file hydration

Raw model hydration downloads independent HTTP byte ranges into private, pre-sized files, then atomically publishes the complete model directory. Each model file is stored as one immutable raw object. Safetensors headers, tensor bytes, tokenizer files and model configuration retain their original representation. The engine loads the published files through its existing bounded weight loader.

A `switchyard-extents-v1` manifest describes destination files and their byte ranges. The agent's `model_source.kind = "files"` selects ordinary file publication without a vLLM loader or a Python bridge. The live fleet uses `model_artifact.kind = "files"`, with the immutable object prefix, manifest layout ID and canonical content ID. The same transport can serve different engines because it does not interpret tensor formats.

The selected L4 configuration uses 128 MiB ranges and a shared limit of twenty active workers across runtime and model hydration. Each range worker has a 64 KiB copy buffer, requests its exact interval, and writes it at the corresponding file offset. Runtime and compiler caches retain direct piped download and decompression. There is no whole-artifact compressed reception spool. Cancellation retains staging ownership until workers stop; structural checks enforce paths, HTTP status, byte boundaries and complete files. Trusted startup does not hash file contents or fsync them.

## Preparation and configuration

Preparation computes file and range hashes before publication. Supply artifact metadata through a JSON file, then publish the raw objects and manifest:

```sh
switchyard artifacts pack-file-ranges \
  --source MODEL_DIRECTORY --out OBJECT_DIRECTORY \
  --metadata METADATA_JSON --range-bytes 134217728
switchyard artifacts publish-extents \
  --config STORAGE_CONFIG --env-file CREDENTIAL_FILE \
  --directory OBJECT_DIRECTORY --prefix IMMUTABLE_PREFIX \
  --receipt PUBLICATION_RECEIPT
```

The [L4 raw-range fleet configuration](../configs/runpod/native-fleet-f32-l4-raw-ranges-candidate.json) pins the published 0.8B, 2B and 4B files, frozen native runtime, static agent binary and concurrency budget. It retains F32 inference, fixed context and batch limits, and a fixed GPU admission floor. Its optional GPU cache registry is independent of model transport. A missing cache does not initiate a bake job.

The [RTX 4090 raw-range candidate](../configs/runpod/native-fleet-f32-4090-raw-ranges-candidate.json) retains those model files, runtime, twenty-worker budget and fixed memory/request limits. It changes the provider GPU selector and the runtime GPU identity check to NVIDIA GeForce RTX 4090. Both GPUs use compute capability 8.9; the existing SM89 executable is reused. GPU cache lookup remains keyed by the actual GPU and driver. Rental quotes are discovered rather than copied from L4 pricing. Measurements on this candidate are separate hardware results, not an L4 observer A/B comparison.

## Fresh serving measurements

Two fresh NVIDIA L4 pods in EUR-IS-1 served the unchanged native SM89 executable through the actual agent. Neither pod received a GPU cache seed. Runtime hydration overlapped weight hydration. The first useful request asked for the word Paris; subsequent requests exercised hot single inference, equal-length batches and mixed-length batches. Both pods completed eighteen requests and were deleted.

| Phase duration | 2B | 4B |
| --- | ---: | ---: |
| Container start after controller launch, estimated | 3.758 s | 3.684 s |
| Agent download | 1.132 s | 0.842 s |
| Runtime hydration, overlapping weights | 6.692 s | 8.868 s |
| Model file hydration | 6.398 s | 16.354 s |
| Model hydration throughput, decimal MB/s | 590.3 | 515.1 |
| CUDA initialization | 0.252 s | 0.290 s |
| Read safetensors into the engine | 2.374 s | 5.117 s |
| Construct model, cast and copy tensors | 3.744 s | 4.738 s |
| Publish HTTP listener | 0.114 s | 0.114 s |
| Total engine process to HTTP listener | 6.484 s | 10.259 s |
| HTTP listener to first batch | 4.772 s | 1.248 s |
| First prefill, including uncached kernel work | 9.392 s | 9.456 s |
| Client request to first streamed content | 10.179 s | 10.885 s |
| **Controller launch to first streamed content** | **33.828 s** | **43.539 s** |

Rows overlap or describe subdivisions; they must not be summed. Launch-to-content uses the controller's monotonic clock. Engine and agent phases use their own monotonic clocks. Container launch estimates compare host and controller wall clocks; clock skew was not measured. The launch timer includes approximately 0.7 seconds of controller capability preparation before submitting the allocation.

Earlier archive measurements reported 43.637/108.302 seconds for model hydration and 71.586/146.736 seconds to first content. Those runs used a different initial prompt and scheduler controller, and the earlier 2B pod was in EUR-IS-2. They provide historical context rather than a controlled A/B result. The new 4B launch increased only the host-buffer byte cap from 12 to 16 GiB; model bytes, F32 math, context, batch and prefix retention limits stayed fixed.

The [measurement receipt](receipts/native-raw-range-l4-rollout.json) records artifact and binary identities, exact phase values, workload scope, telemetry hashes and confirmed pod deletion. Detailed range observations and native engine events remain in `results/native-raw-serving-01/`.

## Remaining timing limits

First prefill still takes about 9.4 seconds on these unseeded GPU caches. Model construction, casts and GPU copies remain part of engine startup. The 4.8/1.2-second listener-to-first-batch intervals combine proxy discovery, controller polling and request transport; the individual contributions were not measured.

Twenty concurrent ranges remove the single compressed weight stream, but cannot establish consistent provider bandwidth. A separate EUR-IS-2 isolation host had a 145-second body-await outlier on one 128 MiB range, with 0.293 seconds spent writing it. The specific R2, network-route or host-network cause remains unresolved. No retry or stall-recovery mechanism conceals that result.
