use serde::Serialize;
use std::time::Instant;

#[derive(Clone)]
pub struct Telemetry {
    epoch: Instant,
}
impl Default for Telemetry {
    fn default() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }
}
impl Telemetry {
    pub fn event(&self, phase: &str, details: impl Serialize) {
        eprintln!(
            "{}",
            serde_json::json!({"event":phase,"elapsed_s":self.epoch.elapsed().as_secs_f64(),"details":details})
        );
    }
}
