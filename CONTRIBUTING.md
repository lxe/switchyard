# Contributing to Switchyard

`Switchyard` is a Rust tool for testing inference-fleet controllers against configurable serving and startup models. Contributions should preserve the separation between physical outcomes, controller observations, and calibration evidence. Start with [the architecture](docs/ARCHITECTURE.md) and [the calibration guide](docs/CALIBRATION.md).

## Local development

Use the Rust version declared by `rust-version` in `Cargo.toml` or a compatible newer toolchain. The project uses Rust edition 2024. Normal development and validation do not need a GPU, cloud credentials, a model download, or an external service.

Run the relevant checks from the project root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets
```

Export examples and run a repeatable comparison with:

```sh
cargo run -- init --out data
cargo run -- suite --seed 7 --seeds 3 --out results
cargo run -- fuzz --cases 1000 --seed 42 --out results/fuzz
cargo run --release -- bench --sizes 1,100,1000,10000 --repetitions 3
```

Generated profiles are editable examples. Results are simulated outcomes, and their provenance must remain visible. Keep generated runs and local experiment data separate from source changes.

## Changes to physical behavior

Explain the physical quantity affected by the change, its units, and the timing boundary. For example, an input-work coefficient changes GPU service, while upload bandwidth changes when a request becomes eligible for service. A startup prior changes readiness; it should not also silently alter steady-state capacity.

Add a small analytical fixture for a material change to admission, service sharing, completion timing, retry behavior, or billing. Useful checks include work conservation, bounded active sequences and KV reservation, client delivery ordering, minimum billing, and deterministic replay. Tests should establish the expected physical behavior independently of the implementation being tested.

Keep the event engine generic. Application roles belong in configurable service classes and dependency graphs. GPU names and provider prices belong in profiles. A controller's desired outcome must not change the hardware's actual output length, service curve, or sampled host conditions.

## Scheduler changes

Implement or extend the `Scheduler` contract and return explicit actions. Make decisions from `FleetView`: known inputs, announced plans, profile priors, and runtime observations. Do not access actual future output lengths or private startup/cache samples. An alternative controller should be runnable against the same physical engine and trace without changing either.

Test queues, announced bursts, output underestimation, memory pressure, dependency chains, slow starts, and failures where relevant. Compare cost, completed work, client TTFT, full response latency, and unfinished requests. Record the configuration and seeds, and explain a useful improvement without hiding the service class or workload that pays for it.

## Calibration contributions

Attach an accessible, sanitized source for every measured claim. Include the configuration, GPU and replica count, workload shape, timing window, token totals, successful and failed requests, and observation date. Mark missing fields rather than reconstructing them from rounded medians or advertised hardware performance.

Keep measured, documented, boundary-adjusted, and assumed values distinct. Do not turn a fleet observation into a per-GPU rate without a known replica count. Do not infer normal build probability from a deliberately selected startup experiment. The host-byte perturbation is additional to the baseline startup; document why it does not count the same hydration twice.

Do not add credentials, private prompts, customer documents, user identities, private absolute paths, or machine-specific source references to distributable fixtures and documentation. Historical local evidence may need a separate sanitized artifact before publication. Provider access and paid experiments require their own authorization; the simulator's tests should remain offline.

## Submitting a change

Describe the problem, resulting behavior, assumptions, and relevant validation. Include a small replayable scenario when it makes the change easier to assess. Update documentation when fields, units, observable state, or result semantics change. Report unresolved calibration limits alongside any performance comparison.

Contributions are provided under the project's MIT license. Preserve third-party notices and do not include model weights, datasets, or benchmark artifacts unless their redistribution terms permit it.
