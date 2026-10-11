# RunPod experiment configuration

`qwen-small.json` is the fleet definition for the live experiment. It pins three Qwen3.5 snapshots, a small Ubuntu image by digest, a separately hydrated vLLM runtime, engine arguments, allowed GPU types and regions, R2 archive prefixes, bootstrap agent checksum, and rental limits. The R2 account address and bucket identify the experiment's existing artifacts; access still requires credentials. Replace these fields to use your own storage.

`gauntlet.json` defines cold and hot single calls, a 1,000-call burst, comparisons with the reference policies, staggered arrivals, longer inputs, underestimated outputs, and dependent requests. Each case records its output estimate, hard output bound, arrival pattern, throughput target, and completion budget. Model output lengths remain unknown until generation finishes.

`smoke.json` is the first GPU boundary check: one cold call, one hot call, and a 32-call burst. Use it with `--model qwen3.5-0.8b` before the full experiment to verify the hydration recipe and streamed usage on one endpoint. It uses the fleet's rental and cleanup limits.

`prices-2026-10-10.json` preserves the initial catalog quote. Availability is a snapshot. Every execution queries the catalog again with the configured minimum CUDA version and records the eligible offers in its receipt directory. The configured disk rate is $0.10 per GB per month, converted using 730 hours per month. Rental receipts are estimates until reconciled with provider billing.

Capacity and startup values under `prior` are assumptions for planning, shared across the allowed GPUs. They are not benchmarks. The live reports measure end-to-end output throughput, streamed first-content latency, and actual token usage. Cold startup includes provider allocation, image preparation, hydration, engine initialization, and client transport.

Credentials are named by environment variables. Credential files, presigned storage URLs, agent tokens, ownership journals, downloaded tokenizers, and raw run artifacts stay outside Git. The source configuration and workload definitions are checked in; every run saves the configuration it actually used.

See [the live execution guide](../../docs/RUNPOD.md) for commands, recovery, and measurement limits.
