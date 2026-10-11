# CPU snapshot and final tensor cold start

Qwen3.5 0.8B returned its first streamed content, `Paris`, **49.954 seconds after local container launch** on an RTX 5090. The first client request took **343.6 ms** to produce content. No tokenization request or inference readiness probe preceded that request. The [timing receipt](receipts/cpu-snapshot-local-cold-start.json) preserves artifact identities, memory measurements and selected native spans.

## Measurement conditions

This run used a fresh Ubuntu container and empty installation, with local HTTP hydration. The base image was already present and the host page cache was not flushed. It does not measure R2 delivery, cloud allocation or image pulling. Native startup instrumentation was enabled; its overhead was not separately measured.

The serving profile uses language-only BF16 weights, eager execution, a 64-token context, one sequence, 64 maximum batched tokens and Triton attention. Its fixed KV allocation is 1 GiB, with a separate 1 GiB headroom requirement. The native memory-profiling forward pass was bypassed. Device initialization, KV allocation and native execution warmup still run.

The CPU snapshot preserves imported Python modules before GPU initialization. Restore used local privileged Docker with host PID visibility and a matching Ubuntu runtime. This proves local restoration, not portability across hosts or permission to restore on a cloud provider. The GPU worker still uses spawn and imports its own modules. The restored process rebinds its main script to the current guarded serving entrypoint before spawning.

## Startup timeline

Times below are seconds after container launch began. Agent milestones use its estimated epoch; restored Python timestamps include a calibrated monotonic-clock offset with 0.854 ms uncertainty.

| Event | Elapsed time |
| --- | ---: |
| Container launch returned | 3.839 s |
| Runtime hydration started | 4.107 s |
| Model metadata ready | 4.641 s |
| Model network reception finished | 6.789 s |
| Serving process spawned | 34.617 s |
| CPU restore started | 34.774 s |
| CPU restore completed | 34.923 s |
| Final tensor copies finished | 46.834 s |
| HTTP endpoint available | 49.570 s |
| First client request submitted | 49.610 s |
| First streamed content | 49.954 s |
| Response completed | 49.967 s |

CPU restore itself took **148.6 ms**. Native worker device initialization took **316 ms**, complete model loading **1.928 s**, KV initialization **2.7 ms**, and execution warmup **1.318 s**. These are nested native operations and must not be added to overlapping parent intervals.

The final tensor consumer delivered 272 tensors directly into their GPU destinations. Recorded pinned staging took 510 ms and CUDA DMA 69.8 ms; these counters overlap other work and are not separate wall-clock phases. CPU-to-GPU transfer remains necessary. The configured transport budgets were 64 MiB RAM, 20 MiB pinned memory and 4 GiB NVMe overflow.

## Hydration and remaining bottlenecks

Each completion time below is measured from that artifact's own reception start. Reception and extraction overlap, so the columns are not additive.

| Artifact | Network bytes | Reception complete | Extraction complete |
| --- | ---: | ---: | ---: |
| Runtime | 4,562,590,833 | 3.866 s | 30.191 s |
| Build component | 682,116,732 | 2.441 s | 13.082 s |
| CPU snapshot | 97,277,747 | 0.050 s | 5.115 s |
| Compiled cache | 27,744,385 | 0.169 s | 0.459 s |

The largest measured delay is runtime extraction, after network reception has finished. Runtime and build artifacts write approximately 19.36 GB in total. The installation resides on NVMe-backed Btrfs with transparent `zstd:3` compression. Decompression, filesystem compression, metadata work and writeback have not yet been independently attributed; transparent filesystem compression is a hypothesis, not a confirmed root cause. No filesystem setting was changed for this measurement.

The container had a 32 GiB cap and swap disabled. Its peak charged memory was **22,040,424,448 bytes**, with observed file-cache usage up to 18.48 GB and dirty file pages up to 12.08 GB. These maxima occur at different samples. Weight-buffer limits do not bound the runtime installation's filesystem page cache. Native worker spawning and imports remain another startup cost after CPU restoration.

## Local evidence

Raw console output, native spans, per-second memory samples and the complete receipt remain under `results/engine-startup/consolidated-cold-03/`. They are ignored by Git. The owned rollout container was removed after measurement; tmux session `0` remains available.
