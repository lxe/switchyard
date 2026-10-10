# switchyard-sim

Deterministic GPU instance lifecycle, serving physics, and synthetic provider/OpenAI-compatible HTTP interfaces. Profiles configure capacity, memory, startup, caches, network delivery, failures, and billing.

`CloudSimulator` owns an incremental fleet. `simulate_with_scheduler` replays an offline trace with a supplied controller. The convenience replay and routing helpers compose the reference policies from `switchyard-scheduler`; the physical engine validates every returned action.

Run the bundled service from the workspace root:

```sh
cargo run -p switchyard --release -- serve --clock-rate 0
```

The [provider API](../../docs/API.md) describes lifecycle, probes, pinned endpoints, and manual clock advancement. The [architecture](../../docs/ARCHITECTURE.md) documents physical approximations and the hidden-state boundary.
