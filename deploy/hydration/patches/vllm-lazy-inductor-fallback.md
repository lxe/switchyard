# Lazy Inductor fallback experiment

This patch is an experimental change to `vllm/env_override.py` in the pinned `vllm 0.31.0+cu129` runtime. It has not been selected for the canonical runtime. The [source descriptor](vllm-lazy-inductor-fallback.json) records the exact original and resulting SHA-256 hashes and replacement text; the [diff](vllm-lazy-inductor-fallback.patch) shows the change against that original source.

The native vLLM helper wraps Torch's `FALLBACK_ALLOW_LIST` so custom operations in the `vllm::` and `vllm_aiter::` namespaces can use Inductor fallbacks. Calling the helper during `import vllm` eagerly imports both Inductor lowering and graph modules. The candidate defers the same helper until the native lowering module finishes importing. It also updates the graph module's local set reference when that module is already loaded, rather than importing the graph module to update it.

The proxy's membership rules, native entries, mutation behavior and idempotence remain unchanged. The existing pattern-matcher backport remains in place. The [native import-order check](test_lazy_inductor.py) passed with completed vLLM-first, lowering-first and graph-first imports. These checks do not cover a partially initialized lowering module that has entered `sys.modules` before defining `FALLBACK_ALLOW_LIST`. The original helper returns in that condition, and this candidate retains that limitation.

## Startup measurements

The [comparison receipt](../../../experiments/vllm-lazy-inductor-startup.json) records six fresh CPU imports and four actual agent launches on an RTX 5090 using canonical Qwen 0.8B weights. The median standalone `import vllm` time fell from 1.713 to 0.730 seconds. The endpoint launches used the same agent binary, runtime objects, compiler-cache seed and explicit native spawn setting. Both sources were staged at `/opt/switchyard/startup-profile/env_override.py` with the same import hook; staging and source hashing finished before the request epoch.

| Measurement | Pristine source | Patched source |
| --- | ---: | ---: |
| Profiled request to first useful token | 30.188 s | 30.074 s |
| Unprofiled request to first useful token, reversed run order | 31.457 s | 30.719 s |
| Profiled API `vllm.env_override` import | 1.877 s | 0.957 s |
| Profiled API compilation-config import | 0.007 s | 0.569 s |
| Profiled API CLI benchmark import | 1.687 s | 2.792 s |
| Profiled worker engine-core import | 3.719 s | 3.948 s |

The import spans nest and cannot be added. Most of the shorter initial import moved into configuration loading through `torch._inductor.custom_graph_pass`, whose package initialization imports standalone compilation and Torch Dynamo. Native post-load initialization also varied: the unprofiled pristine run took 3.06 seconds there, while the patched run took 1.29 seconds. The small endpoint differences across these two pairs do not establish a stable reduction in first useful token, so the patch remains experimental.

The native startup plan applied in all four launches under its existing free-memory checks. Every launch passed agent inference admission, tokenization and a streamed `Paris` response. Each then completed 24 requests with at most eight in flight, for 96 successful concurrency checks. Those requests used identical short prompts with prefix caching; their timings are not a throughput benchmark.

The source overlays are private diagnostics against the original runtime identity. They do not identify a published patched runtime or a compatible exported cache. Selecting any library patch for deployment requires a new runtime artifact identity and its own cache provenance.
