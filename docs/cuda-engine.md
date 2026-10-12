# CUDA engine

The Qwen3.5 engine on hand-written CUDA kernels moved to its own repository, [lxe/pocket-engine](https://github.com/lxe/pocket-engine). It started here as `crates/switchyard-cuda` (commit b4b3d7c). On one H100 it serves Qwen3.5-2B at 7,502 output tokens per second with 64 requests in flight, against 5,467 for vLLM 0.31.0 on the same machine.

Switchyard runs it like any other engine: as a separate process behind its OpenAI-compatible API. Early launch uses the agent's ready file (`SWITCHYARD_READY_FILE`, see [hydration](HYDRATION.md)); `pocket-engine serve --wait-for "$SWITCHYARD_READY_FILE"` sets up the GPU and then waits for it.
