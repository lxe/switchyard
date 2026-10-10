# Switchyard repository design

Switchyard is a Cargo workspace containing scheduling contracts, a scheduler/router, a fleet simulator, a test gauntlet, and a root CLI. The boundaries let a controller depend on observable inputs without depending on the simulator's private physical state. The CLI composes the packages into runnable experiments.

## Projects reviewed

The following primary repositories were reviewed on October 10, 2026. Their organization informed this workspace and README; Switchyard does not import their implementation or claim compatible interfaces.

| Project | Observed pattern | Application in Switchyard |
| --- | --- | --- |
| [Oumi](https://github.com/oumi-ai/oumi) | Source, tests, documentation, and runnable configuration recipes have distinct locations; the README connects workflows to commands and guides. | Checked-in `configs/quickstart` recipes and a command table connected to detailed documentation. |
| [SkyPilot](https://github.com/skypilot-org/skypilot) | The README gives a concrete definition, installation guidance, and a complete task example before a larger examples catalog. | A runnable workload comparison and complete HTTP workflow near the top of the README. |
| [Axum](https://github.com/tokio-rs/axum) | Related Rust packages use a workspace with shared dependencies, compiler requirements, and lint settings. | Four focused libraries and a root package share requirements through the workspace manifest. |
| [Turmoil](https://github.com/tokio-rs/turmoil) | The README identifies simulation crates and their responsibilities; deterministic tests and runnable examples are visible entry points. | A package responsibility table, replayable generated cases, and separate library/HTTP examples. |
| [madsim](https://github.com/madsim-rs/madsim) | A workspace separates simulation infrastructure from ecosystem integration packages. | Shared contracts and scheduling policies have a boundary where provider integrations can be added. |

[DSLab](https://github.com/osukhoroslov/dslab) is another related Rust distributed-systems simulation project. Its modular simulation scope is useful context, although its short README provides less guidance for Switchyard's introductory workflows.

## Package dependency rules

`switchyard-core` owns serializable inputs and observation contracts. `switchyard-scheduler` implements decisions using those contracts. `switchyard-sim` owns physical outcomes and exposes a provider API; its convenience entry points use the reference controller. `switchyard-gauntlet` owns experiment construction and tests across these packages. The root `switchyard` package supplies the CLI and a library facade.

Tests that establish physical behavior remain with the simulator. Tests that compare policies or exercise HTTP and scheduler boundaries live in the gauntlet. Configuration recipes are tracked source files; generated measurements, local evidence, and execution results stay outside Git.

A future live-provider adapter can translate real observations into the same scheduling contract. Provider credentials, billing rules, model deployment, and runtime measurements belong at that integration boundary. Deterministic simulator tests should continue to run without cloud access.
