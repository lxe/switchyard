# vLLM prepared startup

The prepared launcher imports the pinned vLLM serving stack while model metadata and compiler caches arrive. The agent starts it after the boot runtime is atomically complete. The launcher consumes `switchyard-metadata-ready-v1\n` only in the original API process; the agent releases that gate after model metadata, supplemental runtime components, and compiler caches are complete. Spawned workers import the launcher without consuming the gate.

## Agent protocol

The generic agent supplies `SWITCHYARD_RUNTIME_ROOT`, `SWITCHYARD_COMPONENTS_ROOT`, `SWITCHYARD_CACHE_ROOT`, and `SWITCHYARD_RUNTIME_ARTIFACT_ID` from the selected ticket. Supplemental compiler and cubin files live at `/opt/switchyard/components/build`, a sibling of the immutable boot runtime. `FLASHINFER_CUBIN_DIR`, `CUDA_HOME`, and compiler paths can point into this component before it exists, because their import consumers wait for the gate. The artifact ID must identify the complete configured runtime, including supplemental components. `VLLM_ENGINE_STARTUP_POLICY` is a JSON configuration included in the launch fingerprint. No file polling or second agent readiness pipe is required.

```json
{
  "schema_version": 1,
  "strategy": "prepared",
  "worker_method": "guarded_fork",
  "fork_runtime_id": "<selected patched runtime SHA-256>",
  "compute_capability": "8.9",
  "tokenizer_preload": true,
  "cache_policy": "required",
  "memory_plan": "native",
  "telemetry": true
}
```

The default strategy retains the native launch. Prepared spawn is available by selecting `worker_method="spawn"`. vLLM 0.31.0 accepts only native spawn and fork; forkserver is unsupported by its environment contract. Guarded fork requires the selected runtime ID to match the configured pin and its baked patch receipt to identify the exact NVML FLA descriptor `5b935fc5f4eacaff29876d2c9304df0b54ebe69828c6c3da8916ba36e54135f5`. The launcher retains vLLM's CUDA, NUMA, Ray, and WSL safety guards. A native fallback to spawn is recorded rather than bypassed.

## Import and cache boundaries

Preparation runs in an owned thread. Cache-dependent FlashInfer imports wait for the combined gate, preventing workspace creation or provider discovery against an incomplete cache. After the gate, the launcher creates any configured checked cache aliases and invalidates Python import caches before releasing blocked imports. It joins preparation and refreshes the pinned native FlashInfer provider inventory before constructing an engine. Required-cache policy checks the strict native flags and sampling-provider presence; native AOT loading still decides binary compatibility during execution.

`compute_capability` supplies the architecture already established by the GPU/cache compatibility check. It pins FlashInfer's architecture-list environment before imports, avoiding a CUDA initialization solely to discover compilation targets. It must match the selected GPU. The native device checks still run in the engine worker.

## Tokenizer preparation

After native `EngineArgs.create_model_config()` returns, tokenizer preparation uses `cached_tokenizer_from_config()` from vLLM 0.31.0. That function supplies the exact native key: tokenizer path, runner type, tokenizer mode, revision, trust-remote-code flag, and any caller keyword arguments. A lock around the existing native LRU prevents duplicate concurrent misses without changing key construction or object identity. Spawn mode can prepare the tokenizer while engine configuration continues. Guarded fork uses synchronous preparation so no tokenizer preparation thread survives into fork.

## Memory plans and telemetry

Memory-plan policy `native` retains the existing fingerprint and free-memory checks. `required` fails if the native loader declines the plan; it never lowers the recorded free-memory baseline or invents a KV capacity. Native profiling remains the fallback under `native` policy.

Structured `SWITCHYARD_ENGINE_STARTUP` records identify import preparation, combined-gate release, provider discovery, tokenizer preparation, native process-method selection, worker start, and memory-plan decisions. They contain process IDs and monotonic times, without request text, environment values, credentials, or raw exception messages. Native span durations can be nested and must not be summed.

## Compatibility and acceptance

The prepared implementation is pinned to vLLM 0.31.0 and FlashInfer 0.7.0.post1. Loader source identity and launch configuration change when it is selected; model-specific compiler caches must be rebaked or explicitly accepted under that identity. A successful spawn fallback does not establish elimination of duplicated engine imports. That claim requires the actual worker to report fork and matched GPU tests to retain correctness, strict cache reuse, and serving throughput. The unpatched canonical runtime remains eligible for prepared spawn.

CPU acceptance covers the bounded prerequisite token, cancellation, import errors before readiness, exact tokenizer keys and concurrent reuse, pinned native spawn guards, and unchanged memory-plan baselines. `benchmarks/prepared_runtime_probe.py` also passes in the actual Python 3.12 runtime with a provider path missing at interpreter startup: native imports pause, the late cache alias appears, the importer cache is invalidated, and FlashInfer discovers one sampling provider without CUDA initialization. This CPU probe does not load a GPU binary.
