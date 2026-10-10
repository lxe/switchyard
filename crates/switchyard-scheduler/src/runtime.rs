//! Composition of a request router and the cloud simulation.
//!
//! The controller consumes `FleetView` and returns provider actions. It has no
//! access to sampled host reality or future output. HTTP may bundle this router
//! with the provider, while external controllers can use provider routes alone.
use crate::BuiltinScheduler;
use switchyard_core::{Policy, Scheduler};

pub struct RoutingRuntime {
    pub(crate) controller: Box<dyn Scheduler + Send>,
}

impl RoutingRuntime {
    pub fn new(policy: Policy) -> Self {
        Self::with_scheduler(BuiltinScheduler::new(policy))
    }

    pub fn with_scheduler(controller: impl Scheduler + Send + 'static) -> Self {
        Self {
            controller: Box::new(controller),
        }
    }

    /// Borrow the controller that drives this routing runtime's provider actions.
    pub fn controller_mut(&mut self) -> &mut (dyn Scheduler + Send) {
        self.controller.as_mut()
    }
}
