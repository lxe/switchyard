# SGLang and vLLM for Pocket Model

This comparison evaluates the serving process used by Switchyard for Qwen3.5-0.8B, 2B, and 4B. Switchyard hydrates a complete Python/CUDA runtime and a pinned model archive onto each fresh worker, launches an engine behind the agent, and routes OpenAI-compatible streaming requests through the existing scheduler. Engine startup, prefix reuse, and batching affect the latency and rental cost seen by that scheduler.

## Model and engine scope

The checked configurations use the same model revisions and archive hashes, BF16 weights, text-only loading, 4,096-token context, 32 active requests, and one NVIDIA L40S in US-MO-1. The initial successful vLLM bake used US-IL-1; its subsequent fresh-host tests used US-MO-1 after the original region ran out of eligible capacity. Thinking is disabled through the chat template. Output bounds and prompts are identical across engines. The three models are dense hybrid attention models; results for the 397B mixture-of-experts model do not establish their performance. The pinned 0.8B configuration has 24 text layers, with full attention every fourth layer and Gated DeltaNet layers between them. [Pinned Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)

vLLM is pinned to 0.31.0 with CUDA 12.9. SGLang is pinned to 0.5.21 with CUDA 13.0, following its release build requirements. Both set the engine memory fraction to 0.8. Each engine interprets this setting through its own allocator, so actual cache capacity must be observed. Both use Python 3.12.14 and PyTorch 2.13.0. vLLM uses Transformers 5.17.0, Tokenizers 0.23.3, FlashInfer 0.7.0.post1 and its default FlashAttention backend. SGLang uses Transformers 5.12.1, Tokenizers 0.22.2, FlashInfer 0.6.18 and its default FlashInfer backend. The CUDA and attention stacks therefore differ; these measurements compare deployable release configurations rather than isolating only the engine's Python code. Both placement configurations require CUDA 13-capable hosts. All six matched serving runs observed driver 580.159.03; the original vLLM bake observed 595.91.07. Driver eligibility is matched, but the cold-to-baked vLLM startup comparison also changes region and driver. CUDA 13 requires the 580 driver family or newer. [SGLang release Dockerfile](https://github.com/sgl-project/sglang/blob/v0.5.21/docker/Dockerfile), [NVIDIA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

The engine selector lives in the serving adapter. Scheduling contracts, policies, and the simulator do not depend on either engine. Older configurations omit `kind` and retain their existing vLLM cache fingerprint. For Qwen3.5, SGLang uses its native JSON model override to set `language_model_only=true`; the release CLI flag only admits Muse/Cosmos architectures. Its kernel JIT cache needs `SGLANG_JIT_CACHE_DIR` explicitly, because that cache does not follow the ordinary cache-root setting. SGLang maps the generic model length, sequence limit, dtype, memory fraction, and model identity to its own launch arguments. Its native `/tokenize` endpoint and streaming chat endpoint satisfy the existing agent protocol. [Pinned SGLang HTTP server](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/entrypoints/http_server.py)

## Prefix reuse for hybrid attention

A repeated prompt can reuse both full-attention KV tensors and the recurrent state produced by its linear-attention layers. An engine must restore compatible state at the same prefix boundary before processing the new suffix. Reusing attention tensors alone is insufficient for these models.

vLLM's `--mamba-cache-mode align` saves recurrent state on its block grid. A cache hit can resume at those boundaries. The additional shared-prefix checkpoint feature has conditions involving speculative decoding and a smaller prefix-match unit; this experiment uses ordinary aligned caching without speculative decoding. The actual serving block size must be read from the startup log. [vLLM 0.31 prefix caching](https://docs.vllm.ai/en/v0.31.0/features/automatic_prefix_caching/)

SGLang's checked configuration uses `extra_buffer` and page size 64. Its Qwen recipe describes this strategy as enabling overlap scheduling and branching-point caching on NVIDIA with the FLA backend. Extra recurrent-state buffers consume memory, so a throughput improvement can come with lower request capacity. The recipe also describes a lower-memory `no_buffer` strategy. The experiment pins the strategy explicitly instead of relying on a changing default. [SGLang Qwen recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)

The corpus first primes a shared system prompt, then submits serial calls, a 32-call burst, and a 64-call queue. Matching control cases put a unique identifier before the system content, preventing reuse of the long prefix. Their input lengths remain similar. The 64-call cases cycle output bounds of 8, 16, 32, and 96 tokens. Per-response cached-token counts, when supplied, and engine metric snapshots establish whether reuse occurred. A faster shared-prefix case without a cache observation is suggestive and must not be reported as a proven cache hit.

Prefix reuse saves prefill work. It does not remove the compute required to generate a new answer. Interactive decode rate, aggregate output rate, and prefix-sensitive time to first token must therefore be reported separately.

## Runtime artifacts and compiler caches

A runtime artifact contains the interpreter, installed packages, native libraries, compiler, CUDA headers and libraries, and timestamp-independent Python bytecode. It can be prepared without a GPU. The offline smoke check imports the serving stack, compiles C++ and an Ada CUDA kernel using `curand.h` and `curand_kernel.h`, and exercises audio decoding. Successful compilation verifies the header/toolchain closure; it does not prove GPU execution.

A compiler-cache artifact contains generated files in the configured cache directory. Triton kernels, FlashInfer builds, and engine compilation caches may be reused when the runtime, model revision, engine arguments, GPU architecture, and relevant environment match. CUDA graph capture still creates process-local execution state with live device addresses. Copying compiler files does not restore a running model, allocated KV tensors, or an initialized graph execution object. [vLLM compilation design](https://docs.vllm.ai/en/v0.31.0/design/torch_compile/), [NVIDIA graph runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)

vLLM 0.31 also offers experimental initialized engine snapshots using CRIU and CUDA checkpointing. Its documented restore constraints include the same host, GPU, driver, kernel, installed software, model revision, arguments, and filesystem paths, with privileged capture support. Only dense FP16 TP1 is validated, and local model directories are unsupported. This feature is a distinct mechanism and does not provide a portable artifact for Switchyard's arbitrary fresh pods. It is not benchmarked here. [vLLM initialized snapshots](https://docs.vllm.ai/en/v0.31.0/features/initialized_snapshots/)

SGLang HiCache adds host-memory and storage tiers for prefix state. That requires a cache hierarchy and a storage protocol beyond exporting compiler files. Durable prefix restoration for Qwen's hybrid state must be validated for the chosen version and backend, including restoration of both recurrent and attention state. It cannot be inferred from successful hydration of the ordinary cache directory. [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)

## Measurement boundaries

The first request on a fresh worker includes allocation, bootstrap, runtime/model/cache hydration, Python imports, weight loading, compilation, graph capture, and readiness verification. Agent milestones and sanitized engine logs separate the visible stages. Runtime and model hydration overlap, and Python preimport begins after runtime publication while model hydration may still be running.

The current model hydration path connects bounded network reception to sequential archive consumption. Consumer stalls can propagate back to reception. Reported bytes divided by hydration duration describe pipeline throughput and cannot establish an R2 bandwidth ceiling. A download-only object probe is the separate test for network capacity. The comparison uses the same hydration path for both engines; it does not rely on an unfinished parallel model-loader implementation.

Workload reports measure time to first content from scheduled arrival, including queueing and dependency waits. The derived service measurement subtracts `started_s` from `first_content_s` to describe time after dispatch. Aggregate output rate is actual output tokens divided by case elapsed time. Results must retain completed, failed, unfinished, invalid-action, and budget-exceeded counts. The configured capacity priors are scheduler inputs, not measured engine capacity.

A four-call serial case asks for longer explanations with a 96-token bound, providing a sustained single-request measurement alongside the short literal checks. The literal-response cases request `POCKET_MODEL_OK`; response digests permit parity checks without retaining raw text. Digests for free-form explanations are observational. Different valid explanations do not by themselves indicate incorrect inference, and these tests do not establish general model quality.

## Packaging failure receipt

The first paid vLLM cache preparation hydrated its runtime and model but exited before readiness. FlashInfer sampling compilation failed because the runtime's CUDA include directory lacked `curand.h`. Model hydration completed in 27.140 seconds; runtime publication completed at agent elapsed 22.604 seconds and model publication at 28.508 seconds. No serving performance is attributed to this failed run.

The run's estimated rental cost was $0.0482498382997339. Its pod was deleted and independently confirmed absent with GET 404. `results/cloud-vllm-0.8b-bake/independent-receipt.json`, `rental-receipt.json`, `agent-posthydration-state.json`, and sanitized `container-events.ndjson` preserve the evidence. The corrected builder includes the pinned NVIDIA CURAND redistribution and exercises its headers before another paid allocation. CUDA 13 also requires separately packaged CRT and NVVM components; both are included in the SGLang source lock. A subsequent US-TX-3 attempt was rejected before allocation and cost $0.00. Successful matched serving runs use US-MO-1.

The first SGLang bake also failed before hydration with the older agent's generic `invalid ticket` error. Its $0.031368770806214384 estimated rental cost is retained in the ledger. A controlled slow-body test reproduced that generic error when the old 30-second client deadline expired. The corrected agent separates body-transfer failures from JSON/schema failures and allows 120 seconds. The successful SGLang ticket transferred in 4.317 seconds, so the original cloud failure's precise cause remains unconfirmed.

## Reproduction and results

The checked fleet configurations, workload corpus, local functional smoke test, bounded experiment wrapper, and runtime source locks are under `configs/runpod/serving-comparison`, `scripts/serving-comparison`, and `deploy/hydration/sglang`. Credential files are supplied locally to the controller; pods receive scoped download capabilities. Results and local caches are ignored by Git. The wrapper enforces the remaining cumulative reservation, stops an engine failure without reprovisioning, captures sanitized telemetry, and independently confirms termination of allocations recorded in its own journal.

### Matched serving measurements

Each engine completed the same 222-call corpus for each of the three models. The table records the original matched runs; later vLLM startup experiments are reported separately. All six runs had zero failed or unfinished requests, zero invalid actions, no budget-exceeded flag, and nine correct literal responses. These are single-run measurements of bounded workloads, with no confidence intervals. Actual output lengths and free-form answers differ between engines; throughput uses each response's actual token count. They establish serving behavior and literal parity, without establishing general answer quality.

Output rates below are actual output tokens per second. The serial column is four successive longer answers with a 96-token bound. Burst cases submit 32 requests, and queue cases submit 64 requests against a 32-request engine limit. The queue cases cycle bounds of 8, 16, 32 and 96 tokens.

| Model | Engine | Serial longer answers | Shared burst 32 | Unique burst 32 | Shared queue 64 | Unique queue 64 |
|---|---|---:|---:|---:|---:|---:|
| 0.8B | vLLM | 177.4 | 1,979.5 | 1,285.9 | 1,339.4 | 680.4 |
| 0.8B | SGLang | 226.1 | 2,127.9 | 1,550.0 | 1,673.9 | 944.3 |
| 2B | vLLM | 112.3 | 1,139.4 | 717.2 | 1,009.5 | 606.8 |
| 2B | SGLang | 128.3 | 1,045.8 | 667.5 | 1,122.7 | 569.7 |
| 4B | vLLM | 67.5 | 653.4 | 318.7 | 563.6 | 251.3 |
| 4B | SGLang | 72.5 | 792.6 | 282.0 | 709.1 | 240.5 |

SGLang led every listed 0.8B case, by 7.5% to 38.8%. The larger models produced mixed results. SGLang led serial decode by 14.2% on 2B and 7.4% on 4B; vLLM led unique-prefix burst throughput by 7.5% and 13.0%, respectively. SGLang led the 4B shared queue by 25.8%, while vLLM led its unique queue by 4.5%. The request mix therefore matters more than a single engine ranking. Later vLLM 0.8B startup experiments measured 218.5–224.7 tokens/s in the serial case, versus 177.4 in the original run. These use different fresh hosts and cache loading conditions. This variability makes the table useful as workload evidence, while preventing a confident universal throughput ranking from one run per engine.

Native counters prove shared-prefix reuse. On 0.8B and 2B, the eight serial shared calls submitted 16,744 input tokens: vLLM reported 13,056 prefix-cache hit tokens and SGLang reported 16,384 cached tokens. Unique control calls reported zero hit tokens. vLLM's aligned block size was 544 tokens, making the reusable prefix 1,632 tokens per request; SGLang's 64-token pages reused 2,048. On 4B, vLLM used 528-token blocks and reused 1,584 tokens per request. These differences explain part of the shared-prefix result without implying a missing vLLM cache implementation.

Service time to first content also improved under shared prefixes. The 64-call queue's service p95 was 0.467 versus 1.426 seconds for vLLM 0.8B, and 0.251 versus 0.777 seconds for SGLang 0.8B. On 4B it was 0.938 versus 3.641 seconds for vLLM and 0.441 versus 3.357 seconds for SGLang. Scheduled-arrival p95 includes the additional queue wait and is retained separately in the machine-readable report.

### Fresh-host cache observations

The full 222-call 0.8B bake exported each engine's cache only after successful native GPU inference. The resulting artifacts were then hydrated onto different fresh pods, together with the complete runtime and pinned model. The 2B and 4B measurements were cold runs without model-specific compiler-cache artifacts; they do not establish baked startup time for those models.

| 0.8B launch | Model gate to verified inference | Native observation |
|---|---:|---|
| vLLM cold bake, IL-1 | 161.282 s | Torch compilation 13.82 s; initial profiling/warmup 51.49 s; sampling compilation interval about 60.9 s |
| vLLM restored cache, MO-1 | 89.176 s | Native Torch AOT load 0.53 s; initial profiling/warmup 0.69 s; repeated sampling build remained |
| vLLM strict AOT diagnostic, MO-1 | 88.175 s | Native Torch AOT load 0.41 s; v2 warmup gap 58.826 s; startup plan safely rejected |
| vLLM native sampling provider, MO-1 | 30.119 s | Torch AOT required; FlashInfer JIT disabled; sampling module load 0.0022 s |
| vLLM requested fork policy, MO-1 | 29.085 s | Native CUDA guard switched back to spawn; no import reduction established |
| vLLM scoped runtime, new agent, MO-1 | 26.097 s | Torch AOT and native sampling reused; 222 calls accepted |
| SGLang cold bake, MO-1 | 78.191 s | Complete native startup and kernel generation |
| SGLang restored cache, MO-1 | 52.145 s | Some compiled state reused; first long-prefix case still expensive |

The model gate is an agent milestone: it permits model loading after publication. It can occur while Python preimport is still running. Its interval includes imports and process initialization, not just engine constructor time. Cold and restored vLLM rows also differ in region and driver, so the explicit AOT load messages and native stage timings provide firmer evidence of cache reuse than the total difference.

vLLM's persisted startup plan was present in the artifact. The strict diagnostic measured 47,210,430,464 free bytes against the bake baseline of 47,212,462,080, a difference of 2,031,616 bytes (1.94 MiB). The native exact-baseline guard rejected the plan and ran its safe profiling path. This was a compatibility miss, not a missing export. No baseline or capacity metadata was altered to force acceptance.

The larger repeated cost had a separate cause. A local extraction through Switchyard's actual artifact hydrator reproduced Ninja's rejection of the restored sampling objects: its stored dependency timestamps differed from the files created on the new host. A dry run scheduled all three CUDA compilation commands and linking even though `sampling.so` was present. FlashInfer 0.7's native loader first searches registered precompiled providers; absent a provider, it runs Ninja. Its normal log can omit a build announcement when a `.so` already exists, so absence of that announcement does not prove reuse. [Pinned FlashInfer loader](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/flashinfer/jit/core.py)

The follow-up derivative retains every original cache file and registers the baked SM89 sampling module through FlashInfer's native provider API. It adds the pinned official shim, package metadata and one provider entry point. `FLASHINFER_DISABLE_JIT=1` rejects any FlashInfer compilation fallback; `VLLM_FORCE_AOT_LOAD=1` rejects a Torch AOT miss. vLLM explicitly excludes the latter variable from its compiler fingerprint. Diagnostic wrappers record native call durations while preserving their arguments and results, and disable themselves after startup. The official shim is pinned by SHA-256, and architecture, runtime version and outer cache compatibility remain checked. [FlashInfer release](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.0.post1)

The fresh-host native-provider run completed all 222 calls and all nine literal checks with both strict flags enabled. FlashInfer's `build_and_load` took 0.002214 seconds and called `try_load`/`load` without `build`. Native v2 scheduler warmup took 0.294850 seconds, replacing the earlier 58.826-second gap. Engine core initialization fell to 6.67 seconds, while the full model-gate-to-verified interval remained 30.119 seconds. It would be incorrect to call the complete startup a six-second operation.

That interval included 7.055 seconds of preimport, repeated worker imports and setup, 3.189 seconds of model loading, 4.075 seconds in memory planning, and 2.533 seconds in final warmup and graph capture. Timings inside memory planning and warmup are nested and must not be added to their parent durations. The native graph captures took about one second each. The exported Torch and Triton files were reused; the rejected startup-plan baseline explained the remaining memory profiling.

A supported `VLLM_WORKER_MULTIPROC_METHOD=fork` experiment preserved the native safety guard. The API process had already initialized CUDA, so vLLM changed the method back to `spawn`. All 222 calls passed, but the 29.085-second post-gate interval did not establish an import reduction. A subsequent diagnostic identified the import-time query in `vllm/third_party/flash_linear_attention/ops/utils.py`: `_check_platform()` calls `get_available_device()`, which asks Triton's active driver for the current target and reaches `torch.cuda.current_device()`. No native guard was disabled and no installed engine source was patched. [Pinned FLA utility](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/third_party/flash_linear_attention/ops/utils.py)

### Scoped runtime assets

The SM89 profile removes only 42,704 cubin payload files from the pinned `flashinfer_cubin/cubins` tree. Their size is 6,546,167,652 bytes. Package code, Python bytecode, native libraries, compiler tools, headers, checksums and directory paths remain unchanged. The profile is bound to the exact tested Qwen3.5-0.8B BF16 engine configuration, including its ordinary FlashAttention backend and absence of speculative decoding. It is not a general FlashInfer runtime profile for other hardware or kernels.

The profile reduces compressed runtime transfer from 5,432,616,775 to 4,975,192,841 bytes: 457,423,934 bytes, or 8.42%. Disk savings therefore greatly exceed transfer savings. Original compiler files are unchanged, and the launcher preserves their original cache prefix using a checked alias in the fresh worker. It refuses to replace a directory owned by another tree. The new outer runtime/cache identities explicitly record their source artifacts and this derivation.

Offline checks passed imports, timestamp-independent bytecode, C++ compilation, Ada CUDA compilation with CURAND headers, and audio decoding with networking disabled. The fresh GPU test used the new agent pin, required Torch AOT, disabled FlashInfer JIT and downloads, restored checksum verification, and completed all 222 calls with nine correct literals. Native logs confirmed `FLASH_ATTN` and FlashAttention 2. An audit observed FlashInfer package bytecode reads but no cubin payload access. This verifies the selected closure for the checked workload and hardware.

The scoped launch's scheduled-arrival time to first content was 62.790739 seconds. Agent milestones were runtime publication at 26.612372 seconds, model publication and gate release at 30.782828, and verified inference at 56.879528. Runtime hydration's milestone span was 25.724050 seconds; model hydration's span was 29.396106. Post-gate startup was 26.096700 seconds. Its native stages included 3.046 seconds of model loading, 3.927 seconds of memory planning, 2.501 seconds of final warmup/capture, and a 0.001922-second FlashInfer module load.

Raw reception telemetry recorded 4,975,192,841 bytes received in 22.399301 seconds and extraction completing at 22.415217 seconds, only 16 milliseconds later. The buffer high-water mark was 749,007,400 bytes. The 25.046331-second consumer-wait total aggregates waits across concurrent consumers; it is not an additional wall-clock delay. These observations do not show a sustained extraction bottleneck in this run. The older controller drops the new reception fields in typed reports, so the evidence retains them from raw agent state.

Compared with the earlier native-provider run, runtime publication was 12.147 seconds earlier, model publication was 9.205 seconds later, the critical model gate was 7.977 seconds earlier, and post-gate startup was 4.022 seconds shorter. First content was 11.739 seconds earlier. The agent version also changed and these are different fresh hosts, so that total cannot be attributed solely to pruning. A matched control with the same agent/controller was stopped before allocation during a hydration-lock investigation; it cost zero. No matched-repeat speedup is claimed.

### Runtime provenance and reproduction

The vLLM runtime contains 185,601 entries and 5,432,616,775 compressed bytes. The SGLang runtime contains 206,329 entries and 4,503,626,009 compressed bytes. Both artifacts passed offline compiler/header/audio checks, then actual GPU sampling, native engine readiness, tokenizer parity and literal inference. Local RTX 5090 checks are functional evidence only: their different cache histories and timing origins do not support a matched local startup comparison.

The SGLang native dependency layer reused a read-only existing builder layer after BuildKit evicted the new APT cache. A pinned `xz` package completed that layer. The recorded inventory proves that all 327 native package versions match the checked source recipe; it does not claim that a second fresh APT transaction occurred. The complete source lock and recipe remain available for fresh builds.

Use `deploy/hydration/build.py` and the checked source locks to build the complete runtime; run its offline smoke check before `switchyard artifacts pack` and `publish`. `local-smoke.sh` provides the actual GPU acceptance gate. `matrix.py` performs model-specific bake and gauntlet operations sequentially through `run.py`, which requires fresh output directories and preserves failed receipts. Supply the provider environment file before the R2 environment file, since the latter can contain a different provider credential.

```sh
python scripts/serving-comparison/matrix.py \
  --provider-env /path/to/provider.env --r2-env /path/to/r2.env \
  --ledger results/comparison-budget.json \
  --vllm-bake results/cloud-vllm-0.8b-bake-il \
  --sglang-bake results/cloud-sglang-0.8b-bake-mo-v2 --suffix mo

python scripts/serving-comparison/strict-cache.py \
  --config configs/runpod/serving-comparison/vllm.json \
  --bake results/cloud-vllm-0.8b-bake-il \
  --sampling-provider-shim /path/to/pinned-flashinfer-shim.whl \
  --profile --out results/vllm-sampling-aot-profile-cache

python scripts/serving-comparison/summarize.py \
  --ledger results/comparison-budget.json --out results/comparison-summary.json
```

The derivative must be published with the artifact CLI before its generated configuration is used by the guarded gauntlet. The wrapper's reservation is cumulative, including failures, and an allocated pod is not considered gone until an independent GET returns 404. Tracked evidence is under `docs/serving-comparison-data`. `comparison-budget.json` includes failures and zero-cost preallocation stops; `comparison-summary.json` contains case counts, literal checks, arrival/service timing, prefix counters, native stage observations and raw reception fields. Artifact identity files contain full SHA-256 IDs, publication receipts and source provenance. All finalized owned pods were independently confirmed absent with GET 404. Finalized comparison rental spend is $0.6937564375247596, leaving $0.3062435624752404 of the authorized $1 reservation.

`runtime-profile.py --validate-only` checks source launch fingerprint, runtime labels, exact engine identity, model revision/archive and GPU binding before cloning. Negative checks reject changed launch environment, attention overrides (including equals/underscore forms), speculative settings, model revision and GPU binding. `runtime-profile.py` then verifies both complete source trees, derives the scoped runtime and cache artifacts, and emits a fresh configuration. Use `--agent-url` and `--agent-sha256` to reproduce the accepted new-agent pin. Publish both emitted artifacts before a guarded gauntlet.

The calibrated configurations retain the baseline experiment files, and all three calibrated/profile configurations passed read-only discovery with the new controller. The scoped profile configuration changes only scheduler priors from its accepted launch, leaving its engine/runtime fingerprints intact. Its prior is 203 serial output tokens/s, 761 aggregate output tokens/s, and 70 seconds to first content. They use 90% of measured serial-long and unique-prefix queue output rates, with startup rounded upward by 10%. Their `input_work_per_token=0.05` remains an unmeasured conservative assumption. `kv_tokens=131072` bounds 32 active 4,096-token contexts, rather than claiming maximum physical cache capacity. The calibration receipt identifies the exact source run for every value; its baseline startup priors include unpromoted cache behavior and should not be substituted for the optimized profile's observed startup.

### Serving recommendation

Retain vLLM as Switchyard's default with native precompiled sampling reuse for the verified 0.8B SM89 configuration. Its existing adapter and the measured larger-model unique-prefix batch results support that choice. Keep SGLang available through the engine selector for an explicit workload decision: it is the stronger throughput candidate for the measured 0.8B workload and for shared-prefix 4B batches. A production decision should compare the expected lifetime of a worker and its request mix, because a throughput gain on a warm server does not recover repeated cold-start compilation cost on a short-lived one.

The matched runs validate both engines against the existing serving boundary. They do not validate persistent KV/SSM state across pods, initialized process snapshots, untested model-specific cache artifacts, maximum capacity, or broader answer quality.
