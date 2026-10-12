# switchyard-scheduler

Inference placement and routing policies built on Switchyard's observable-state contract. `BuiltinScheduler` implements `Scheduler`; `RoutingRuntime` owns a replaceable controller for the stable client endpoint.

The reference policies compare warm capacity, reactive scaling, cheap elastic capacity, announced workload planning, and adaptive host probing. Planning uses input sizes and output estimates; admission separately respects hard output limits.

```rust
use switchyard_core::Policy;
use switchyard_scheduler::BuiltinScheduler;

let controller = BuiltinScheduler::new(Policy::Adaptive);
```

This crate depends on `switchyard-core` and has no dependency on the simulator's physical engine. New policies implement `Scheduler::decide(&FleetView)` and return `Action` values. See the [architecture](../../docs/ARCHITECTURE.md) and [gauntlet](../switchyard-gauntlet/README.md) for validation.

See [Inference scheduling](../../docs/SCHEDULING.md) for measured profiles, adaptive feedback, contribution-based scaling, and bounded live provisioning.
