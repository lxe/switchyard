# Native CUDA cache lifecycle

The native runtime consists of a Rust executable, its CUDA library dependencies and a launch wrapper. Model artifacts contain pinned weights and tokenizer assets. Fixed serving configuration controls context length, batching, memory admission and prefix retention. These are build and packaging inputs; they do not require a serving GPU to prepare Python imports, vLLM startup state or a CPU process snapshot.

CUDA driver caches are optional acceleration artifacts. They contain compiled kernel code, not weights, live tensors, KV pages or saved requests. The driver can compile missing code during ordinary inference. A cached launch still initializes CUDA, constructs the model, transfers weights and allocates its serving state.

## Cache lookup and collection

A native deployment can enable `bootstrap.native_cache_registry`. Its lifecycle integration looks up a family using the complete runtime, launch, model, engine configuration and GPU bindings. The family stores separate directories for exact driver versions. The launch wrapper selects only the observed GPU and driver. If the matching directory is absent, the opportunistic policy creates a private writable cache and starts the engine normally. GPU or compute capability mismatches and malformed existing identities remain errors.

The scheduler continues to decide where requests run and when capacity is released. It does not submit cache warmup jobs. At retirement, the lifecycle integration can drain useful requests, stop the engine and collect its compiler cache using the existing authenticated agent export. It downloads that bounded artifact, confirms release of the owned pod, then publishes the artifact and its registry reference. Publication therefore does not extend the paid host's lifetime.

Lookup, collection and publication have separate time and byte bounds. A missing registry record allows cold execution. Errors in those operations are recorded and do not turn a successful inference workload into a failed workload or prevent owned-pod cleanup. Explicitly pinned caches retain their existing strict semantics.

```json
"native_cache_registry": {
  "name": "gpu-kernels",
  "registry_prefix": "switchyard/native-engine/cuda-registry/v1",
  "artifact_prefix": "switchyard/native-engine/observed-cuda-caches/v1",
  "compatibility": {
    "native_driver_cache_layout": "native-cuda-driver-family-v1"
  },
  "lookup_timeout_s": 3,
  "collection_timeout_s": 30,
  "publication_timeout_s": 60,
  "max_artifact_bytes": 536870912,
  "transfer_concurrency": 4
}
```

The selected runtime must set `NATIVE_ENGINE_CACHE_POLICY=opportunistic`, and its cache root must use the same configured name. A deployment that starts with no seed uses an empty model `caches` array. Registry artifacts use the same bounded direct hydration as other caches. They remain separate from inference-weight transport. Registry events record lookup, collection and publication phases and elapsed time in `native-cache-registry.ndjson`; the gauntlet observer prints concise phase summaries in the existing tmux pane.

## Validation and limitations

[The cache smoke workload](../configs/runpod/native-cache-smoke.json) makes ten useful inference calls: one cold request, one hot request and eight requests with a shared system prefix. Running it twice against fresh owned pods tests cache collection and reuse without dedicated bake requests or inference readiness probes. Driver identities, hit/miss markers, exact executable and configuration hashes, exports, publication and cleanup receipts are required to establish reuse. Two successful launches with different drivers do not demonstrate a cache hit or a cold-start improvement.

The complete [native gauntlet](../configs/runpod/native-gauntlet.json) remains the scheduling and throughput gate. The smoke workload does not replace its 1,470 requests per model. Throughput targets remain goals and any missed target must be reported.

A cache learned from real traffic covers the kernels that traffic used. A later request shape or image workload may require additional compilation. Cache presence does not establish numerical accuracy: the engine's existing reference gates and the separate quantized-model limits still apply. Concurrent registry publication can replace a previous family reference; a lost variant causes another compilation, never reuse of a mismatched driver cache.

NVIDIA documents the driver cache path and bounded eviction in its [CUDA environment-variable reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html). Its [JIT cache explanation](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/) describes invalidation after a driver upgrade. Exact per-driver selection keeps those opaque cache files separate.
