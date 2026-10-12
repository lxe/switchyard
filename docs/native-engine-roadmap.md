# Native inference engine development goals

The native engine is a Rust serving backend for the scheduler and fleet adapters. Its first milestone loads Qwen3.5 0.8B, serves bounded single and batched requests, and compares token generation against published model mathematics and vLLM. The [accuracy receipt](native-engine-accuracy.md) passes the F32 mathematical gate; strict BF16 sequence parity with vLLM remains incomplete.

## Three-model local and RunPod fleet

Configure Qwen3.5 0.8B, 2B and 4B with [pinned source checkpoints](native-model-fleet.md) and explicit context, batch, queue and memory plans. Test each model locally with single calls, batches and numerical references. Build a minimal native runtime, publish runtime, tensor and compatible GPU-cache artifacts to R2, and validate fresh RunPod launches with first-streamed-token timelines. Integrate the resulting fleet with the existing scheduler and run the live workload gauntlet. Simulator abstractions remain independent of RunPod provisioning and artifact packaging.

This is the current implementation priority. RunPod build and rollout preparation can proceed independently while local numerical and model tests finish. Provisional cloud artifacts must be rebaked against the final engine and model configuration before acceptance.

## GPU architecture build matrix

Produce compiled CUDA kernels for a configured set of GPU architectures in one build job, using architecture-specific binaries or a multi-architecture fat binary where the backend supports it. Record the compiler, architecture targets, engine version and required driver/CUDA compatibility in an artifact manifest. Verify that selected platforms load the intended compiled kernels without recompiling them during a cold launch.

Separate compiled kernel artifacts from driver JIT caches and other hardware-dependent runtime plans. A multi-architecture kernel build must not claim that every driver's cache or every GPU execution plan can be produced without target-platform validation. RunPod platform baking supplies that validation where necessary. Coordinate this implementation with the native runtime packaging so it does not create a second artifact-selection policy.

## Batch throughput and quantized weights

Study the kernel, batching, memory-layout and quantization implementations in llama.cpp and suitable Rust runtimes. Add optimizations through the existing model/backend interfaces, with measured throughput and startup changes. Establish a precise quality contract for each reduced-precision mode using the released model reference, next-token probabilities and representative generated workloads.

The [quantized projection foundation](native-quantized-batching.md) now indexes bounded GGUF headers and executes grouped packed matrices, with CPU tests and a passing RTX 5090 grouped/separate fixture. It does not admit GGUF models to serving. The separate dense model adapter now supports [reference-tested mixed-length decode](native-merged-decode.md), with independent prefills and owned state compaction. The [paged attention and prefix cache implementation](native-prefix-cache.md) now shares immutable history, caches complete hybrid-state checkpoints and executes direct CUDA paged decode. Three-model F32 reference gates pass; BF16 cached versus uncached token differences still require a quality contract. Continuous admission, slab allocation and long-context kernel optimization remain work.

Add GGUF model support if its weight layouts and quantized GPU kernels provide a measured advantage. GGUF parsing alone is not the throughput optimization: the backend must execute the quantized tensors efficiently. Prefer a bounded reusable Rust loader and existing suitable kernels over importing an unrelated serving stack. Keep dense safetensors support, OpenAI request handling, routing and provider code independent of the weight format.

Correctness tests must cover format/layout errors, quantization boundaries, cached decode, independent batch rows and state reuse. Performance tests must include repeated prompts, varied prompt lengths, concurrent arrivals and realistic context/output lengths. Optimization claims require comparable workloads and an explicit accuracy tradeoff; speculative speedup estimates are not acceptance evidence.

## Qwen3.8 27B support

Add the released Qwen3.8 27B checkpoint after checking its model card, architecture, tokenizer, tensor layout and license. Pin the source revision and implement only the adapter differences needed by that release. Select hardware or validated quantization that fits the fixed memory plan, then run the same mathematical, single/batch, cold-start and scheduler tests used for the smaller models.

The [published checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B) currently resolves to revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its text configuration still identifies `qwen3_5_text`, with 64 layers, hidden width 5120, 48 GDN value heads and 16 key heads. It declares `output_gate_type: swish` and untied embeddings. The current smaller-model adapter uses sigmoid attention output gating; architecture naming alone therefore does not establish compatibility. Add a reference-tested gate implementation and untied-output tests before claiming support. The dense BF16 weights exceed the local 32 GiB GPU capacity, so GPU acceptance requires a larger platform or a validated quantized backend.

Architecture investigation and fixture preparation can proceed in parallel. Runtime integration and final GPU validation depend on the accepted model/backend and quantization interfaces above.

## Work coordination

Keep numerical/model changes in the main engine checkout. Use isolated branches for RunPod packaging, architecture build support and throughput/format work. Reserve the local GPU for one measured workload at a time; use owned cloud pods for platform-specific baking. Integrate committed changes only after their reported tests and interfaces are reviewed, then rebake affected artifacts using the integrated binary SHA-256. Shut down owned pods after their measurements.
