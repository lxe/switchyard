# switchyard-core

Shared request, hardware, configuration, and scheduler observation types for Switchyard. `FleetView` and `Scheduler` define the controller boundary: known inputs, estimates, and observations enter the controller, and provider actions leave it.

The crate also defines configuration validation and nearest-rank statistics. `PlannedRequest` excludes actual output length. Full offline traces retain that field as fixture truth for the physical engine.

See the [configuration reference](../../docs/CONFIGURATION.md) for units and validity rules, and the [architecture](../../docs/ARCHITECTURE.md) for visibility rules.
