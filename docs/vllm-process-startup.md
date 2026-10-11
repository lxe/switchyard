# vLLM process startup

The local startup experiment measures the interval between the agent spawning vLLM and the weight loader receiving its first tensor request. On the RTX 5090, the unpatched serving process initializes CUDA while importing the Flash Linear Attention (FLA) utilities. vLLM then selects a fresh engine process using `spawn`, which imports the serving stack again before loading weights.

## CUDA initialization cause

The recorded control call stack is `FLA utils._check_platform → get_available_device → Triton get_current_target → torch.cuda.current_device → _lazy_init → _cuda_init`. The control subsequently logs that its native safety guard overrides the requested `fork` policy because CUDA is initialized. This establishes a concrete import-time cause; the CUDA initialization operation itself takes tens of milliseconds on this GPU, while the resulting fresh interpreter repeats seconds of imports and model setup.

The [pinned FLA patch](../deploy/hydration/patches/vllm-0.31.0-fla-nvml.json) obtains the platform and Hopper classification through NVML only when a single visible GPU has an uninitialized CUDA driver and its physical identity is unambiguous. Every other case follows the original queries. The patch retains vLLM's native CUDA, Ray, NUMA and WSL process guards. With this patch, the local engine inherits the already imported modules through native fork and initializes CUDA in `Worker.init_device`.

## Local fixture

These measurements use Qwen3.5-0.8B BF16, Python 3.12.14, Torch 2.13.0+cu129 and vLLM 0.31.0+cu129 on an RTX 5090 with driver 615.71.09 and an Intel i7-13700. The actual static agent hydrates the runtime and raw weight extents into a fresh owned container. The compiler cache is copied from the same immutable local seed before each request epoch; this is a diagnostic cache reuse fixture rather than a published patched cache.

The engine uses a 64-token context, one active sequence, 64 batched tokens, GPU memory utilization 0.25, eager execution, TRITON attention and seed 123. Its GPU, runtime artifact and engine configuration differ from the L40S fleet's 4,096-token, 32-sequence graph-enabled configuration. The [L40S cold-start timings](COLD_STARTS.md) still measure approximately 20.6 seconds before tensor loading; the local experiment cannot partition or replace that number.

The final comparison stages both pristine and patched FLA source through the same module-overlay mechanism and uses identical profiler bytes. Both sides request native fork and disable the optional persisted memory plan for this experiment, leaving full native memory profiling enabled on both. This removes variation from the desktop GPU's free-memory gate. Earlier diagnostic runs retain their original plan settings and harness differences in the numeric receipt; they are supporting observations rather than the strict comparison.

## Timing and inference checks

The final matched pair measured the following intervals. The phase boundaries below are consecutive, so they can be added; individual nested import spans cannot.

| Startup phase | Control | Patched |
| --- | ---: | ---: |
| Engine spawn to Python profiling entry | 0.03 s | 0.05 s |
| CLI imports and argument setup | 4.29 s | 4.34 s |
| Engine configuration | 0.89 s | 0.90 s |
| Parent engine-client setup | 2.05 s | 2.07 s |
| Child start, imports and worker construction | 4.17 s | 0.16 s |
| GPU device initialization | 0.39 s | 0.39 s |
| Model setup to first tensor request | 0.85 s | 0.33 s |
| **Engine spawn to first tensor** | **12.68 s** | **8.23 s** |
| Request to first useful token, including hydration | 30.37 s | 25.83 s |
| Serial fixed32 output rate | 78.66 tokens/s | 78.56 tokens/s |
| Queued fixed32 output rate | 78.23 tokens/s | 78.78 tokens/s |

The pre-tensor interval falls by 4.45 seconds, or 35.1%. The largest reduction is the child-start interval: native fork inherits imports instead of executing them again. Model setup also reuses the tokenizer already constructed in the API process. The Python-entry boundary is bounded by the agent epoch samples; displayed subphase values use the midpoint, with less than 10 ms of epoch uncertainty. The measured total uses the agent's exact milestones. One matched pair establishes a local improvement, while the supporting launches show its repeatability; these samples do not establish a statistical throughput equivalence bound.

See the [numeric receipt](../experiments/vllm-local-process-startup.json) for all runs, the final matched comparison, source hashes and phase boundaries. Import spans can contain other imports, so their durations must not be added. Agent and Python events share Linux monotonic time; the agent epoch is bounded by the pre-exec clock sample and its first ticket request.

Each launch checks tokenization, a streamed `Paris` response and 24 further deterministic `Paris` calls with at most eight clients in flight. The decode comparison uses 16 serial and 16 queued requests per endpoint, each forced to produce 32 tokens. It records completion-token counts, content hashes and direct HTTP durations. With one active sequence, the queued phase measures waiting clients behind serialized decoding; it does not calibrate a concurrent production fleet.

## Reproduction

Run `crates/switchyard-weights/benchmarks/run_agent.py` with a fresh output directory and `--objects`, `--legacy-ticket`, `--runtime-objects`, `--agent` and `--warm-cache-from` pointing to the pinned local inputs. Select `--mode extents --profile-startup --runtime-env VLLM_WORKER_MULTIPROC_METHOD=fork --runtime-env VLLM_ENABLE_STARTUP_PLAN=0 --verify-concurrency 24 --concurrency 8 --decode-benchmark 16`.

For each side, provide `--module-source vllm.third_party.flash_linear_attention.ops.utils=<source.py>`. The control source must match the patch descriptor's `before_sha256`; applying each declared replacement exactly once must produce its `after_sha256`. The harness stages and hashes the source before the request epoch and records both identities. An overlay is private diagnostic input: it does not turn the original runtime or copied cache into a published compatible derivative.

The profiler records process starts, native startup operations, imports and the caller stack for CUDA initialization. Caller frames contain file, line and function identifiers, without arguments or error messages. Its source overlays also work without timing hooks for unprofiled comparisons.

## Deployment scope

The patch is an engine-specific optimization and belongs in runtime preparation. It does not change simulator behavior, routing policies or hydration transport. These local results support avoiding duplicate interpreter imports on this accepted fixture. Publishing a selected derivative requires its own runtime identity, cache provenance and matching native acceptance on the production GPU and engine configuration. The canonical three-model fleet remains unchanged by this diagnostic.
