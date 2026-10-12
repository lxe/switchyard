# switchyard-gauntlet

Workload generation, cross-component regressions, scheduler comparisons, and software performance measurements for Switchyard. The crate contains portable example profiles and scenarios, seeded generative fuzzing, accounting invariants, and report generation.

Run all component regressions from the workspace root:

```sh
cargo test -p switchyard-gauntlet --all-targets
cargo run -p switchyard --release -- fuzz --cases 1000 --seed 42
cargo run -p switchyard --release -- suite --seed 7 --seeds 3
cargo run -p switchyard --release -- bench --sizes 1,100,1000,10000
```

`fuzz::case(seed)` constructs a replayable workload, and `fuzz::check` checks conservation and admission invariants. `performance::measure` records simulator wall time separately from virtual GPU throughput. The integration tests cover physical service, scheduler decisions, observation privacy, provider HTTP, and streaming.

See [validation and results](../../docs/RESULTS.md) for evidence and [calibration](../../docs/CALIBRATION.md) for measurement boundaries.

`empirical::replay` compares a configured simulation to reference cloud metrics, while `empirical::fit` estimates service parameters using only declared training cases. [Cloud observation replay](../../docs/EMPIRICAL_SIMULATION.md) describes the checked-in dataset and CLI tolerance gate.
