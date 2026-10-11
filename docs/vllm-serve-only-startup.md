# vLLM serve-only startup experiment

The [serve-only adapter](../crates/switchyard-weights/benchmarks/serve_only.py) is an experimental runner for the pinned vLLM 0.31.0 runtime. It registers the native `ServeSubcommand` without registering the benchmark, snapshot, batch and other general CLI commands. It has not been selected for the canonical extent runner because the matched endpoint trial found no demonstrated startup benefit.

## Native parser and dispatch

The pinned `vllm.entrypoints.openai.api_server` module forwards to `vllm.entrypoints.launchers.api_server.entry.main`. That function uses the same `make_arg_parser`, environment setup, logging configuration and argument validation as `vllm serve`, then calls the single API server directly. It does not run `ServeSubcommand.cmd`, which applies positional-model precedence and selects headless, parallel, Rust-frontend and gRPC dispatch. A CPU check confirmed that the direct entrypoint accepted `metadata-model --language-model-only --headless`, retained the default `model=Qwen/Qwen3-0.6B` alongside `model_tag=metadata-model`, and called the single-server function.

The candidate therefore retains the native `FlexibleArgumentParser`, leading `serve` token, `ServeSubcommand.subparser_init`, validation and command dispatch. The leading token preserves native `--model` normalization and YAML configuration precedence. Commands outside its serve scope, including `--omni`, execute the installed general CLI source. Native engine creation, loader-plugin registration, memory and cache guards, server construction and API behavior remain in the same serving functions.

The [parser comparison](../crates/switchyard-weights/benchmarks/test_serve_only.py) passed 11 native cases. It compared complete argument namespaces, validation failures and dispatch after the native command ran, including extent-loader settings, language-model-only mode, positional and YAML model precedence, deprecated `--model`, underscore normalization, headless serving, multiple API servers and data-parallel defaults. Only final server-launch functions were intercepted for those CPU checks; actual GPU serving was verified separately.

## Endpoint measurements

The [numeric receipt](../experiments/vllm-serve-only-startup.json) records four fresh private containers using canonical Qwen 0.8B weights on an RTX 5090. Every run used the same fixed agent, original runtime objects, mature compiler-cache seed and explicit spawn setting. Pristine and candidate CLI sources used the same source overlay at `/opt/switchyard/startup-profile/cli_main.py`. Staging and source hashing finished before the request epoch. The Rust runner, loader identity, runtime identity and cache labels were unchanged for the diagnostic.

| Measurement | General CLI | Serve-only candidate |
| --- | ---: | ---: |
| Profiled request to first useful token | 30.446 s | 29.833 s |
| Unprofiled request to first useful token, reversed run order | 30.027 s | 30.084 s |
| Profiled API `import vllm` to engine configuration | 4.361 s | 4.343 s |

The general CLI's benchmark import took 1.737 seconds, followed by launch and preload imports of 0.343 and 0.132 seconds. Those imports load dependencies that serving also needs: the candidate's serve import took 2.201 seconds, including 2.166 seconds in the API-server entry module. The complete API import-to-configuration interval differed by only 18 milliseconds. The candidate also began Python imports about 0.77 seconds earlier on the request clock, before its CLI adapter executed. The profiled wall-time difference therefore does not establish that skipping command registration saves startup time, and the reversed unprofiled pair showed no benefit.

The native startup plan applied in both profiled runs. Its existing free-memory guard rejected the plan in both unprofiled runs: available memory was 26.57 GiB for the candidate and 26.55 GiB for the general CLI, below the recorded 26.58 GiB baseline. Native full memory profiling ran in those two cases. Each pair therefore followed the same native plan path, while the two pairs used different paths; none of the guards were relaxed.

All four launches passed agent inference admission, tokenization and the first streamed `Paris` response. Each then completed 24 requests with at most eight in flight, for 96 successful concurrency checks. The bursts used identical short prompts and prefix caching; their timings are not a throughput benchmark. Import spans nest and reuse dependencies, so their durations cannot be added or treated as wholly avoidable work.

The candidate and its parser evidence remain available for review. Promoting a runner change would require a new adapter identity and corresponding cache provenance; these diagnostic overlays do not represent a published adapter or cache.
