//! Deterministic GPU fleet physics and synthetic provider/serving HTTP APIs.
pub use switchyard_core::config;
pub use switchyard_scheduler as scheduler;
pub use switchyard_scheduler::runtime;
pub mod engine;
pub mod http;
pub use config::*;
pub use engine::{simulate, simulate_with_scheduler};
