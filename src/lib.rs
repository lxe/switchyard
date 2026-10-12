//! Switchyard: inference simulation, scheduling, routing, and validation.
//!
//! This facade composes the workspace libraries. Use their individual crates
//! when extending one component without depending on the complete toolkit.
pub use switchyard_core::config;
pub use switchyard_core::config::*;
pub use switchyard_core::{Action, FleetView, Scheduler};
pub use switchyard_gauntlet::{fuzz, performance, report, scenarios};
pub use switchyard_scheduler as scheduler;
pub use switchyard_scheduler::runtime;
pub use switchyard_sim::{
    engine, http, observations, simulate, simulate_with_observations, simulate_with_scheduler,
};
