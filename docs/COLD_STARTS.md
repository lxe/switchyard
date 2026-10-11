# Cold-start measurements

These measurements describe the latest fresh-pod Qwen3.5 0.8B, 2B and 4B cache-reuse smoke tests on an L40S. Each test used the complete unpatched vLLM runtime, raw safetensor extents, required native Torch AOT loading, and disabled FlashInfer JIT. Each model completed 34 calls; these runs do not establish full scheduler gauntlet acceptance.

## Critical startup stages

Client cold TTFT runs from scheduled request arrival to first useful streamed content. Agent phases use their own epoch. The outside-agent residual is the difference after subtracting agent elapsed time and controller allocation time; it includes unseparated container/bootstrap and client readiness/routing work. It is not a measured container-start duration.

| Duration | 0.8B | 2B | 4B |
| --- | ---: | ---: | ---: |
| Provider allocation | 3.87 s | 3.78 s | 3.64 s |
| Agent start to runtime hydration | 1.87 s | 1.28 s | 1.55 s |
| Runtime hydration | 29.74 s | 37.41 s | 60.83 s |
| Engine spawn to first tensor | 20.65 s | 20.63 s | 20.56 s |
| Tensor assignment | 0.25 s | 0.40 s | 0.76 s |
| Remaining initialization and verification | 7.06 s | 7.07 s | 8.07 s |
| Outside-agent residual | 4.22 s | 4.03 s | 5.34 s |
| Client cold TTFT | 67.66 s | 74.60 s | 100.74 s |

Runtime hydration is the critical transfer in all three runs. Model hydration overlaps it and completed before the runtime became ready. Model hydration spans were 9.04 seconds, 32.04 seconds, 56.28 seconds for 0.8B, 2B and 4B respectively. The table describes the measured critical path, with the outside-agent component obtained by subtraction. The roughly one-millisecond runtime-ready-to-engine-spawn transition is below its displayed precision.

The approximately 20.6 seconds before first tensor delivery is distinct from GPU weight assignment, which completes in less than a second. The current measurements do not divide that interval into Python imports, CUDA initialization and model construction. The final initialization interval also includes the health, tokenization and nonstreaming inference checks.

## Runtime reception and extraction

Every runtime received 5,432,616,775 compressed bytes and wrote approximately 20 GB of files. Reception and extraction operate concurrently. The rates below divide compressed bytes by the recorded reception span and include any pauses in that span; they are not peak link rates.

| Measurement | 0.8B | 2B | 4B |
| --- | ---: | ---: | ---: |
| Reception span | 20.669 s | 33.446 s | 56.378 s |
| Extraction tail after reception | 4.950 s | 0.026 s | 0.041 s |
| Other runtime hydration work | 4.120 s | 3.940 s | 4.407 s |
| Average reception | 263 MB/s | 162 MB/s | 96 MB/s |

The remaining hydration work lies outside the timed receiver/extractor region and is not separately instrumented here. Concurrent body-wait and consumer-wait counters sum worker time; adding them to elapsed startup would count overlapping time repeatedly.

The [machine-readable timeline](../experiments/qwen-extents-cold-start-timelines.json) contains exact milestones, reception counters, tensor-loading measurements and links to each fleet configuration and smoke receipt.
