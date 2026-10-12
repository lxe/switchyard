//! Versioned serving measurements usable by both live profile preparation and simulations.
use crate::{Profile, Scenario, validate};
use serde::{Deserialize, Serialize};

/// Binding is an opaque deployment fingerprint supplied by the serving adapter.
/// It covers weights/revision, engine options, precision, runtime and cache compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingCalibration {
    pub schema_version: u32,
    pub binding_sha256: String,
    pub profile: Profile,
}
impl ServingCalibration {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || self.binding_sha256.len() != 64
            || !self.binding_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid serving calibration version or binding".into());
        }
        let scenario: Scenario = serde_json::from_value(serde_json::json!({
            "name":"calibration-validation", "notes":"", "tick_s":0.1, "launch_fail_probability":0.0, "jobs":[], "requests":[], "warm":[],
            "horizon_s":1.0, "max_nodes_per_model":1, "idle_timeout_s":1.0,
            "reactive_delay_s":0.0, "capacity_scale":1.0, "startup_scale":1.0,
            "build_probability":0.0, "faults":[]
        }))
        .map_err(|e| e.to_string())?;
        validate(std::slice::from_ref(&self.profile), &scenario)
    }
    /// Preserve the current provider quote and physical admission limits.
    pub fn apply(&self, target: &mut Profile, expected_binding: &str) -> Result<(), String> {
        self.validate()?;
        if self.binding_sha256 != expected_binding
            || self.profile.id != target.id
            || self.profile.model != target.model
            || self.profile.gpu != target.gpu
            || self.profile.workload != target.workload
            || self.profile.max_in_flight > target.max_in_flight
            || self.profile.kv_tokens > target.kv_tokens
        {
            return Err(
                "serving calibration differs from deployment identity or memory limits".into(),
            );
        }
        let rate = target.usd_per_hour;
        let mut measured = self.profile.clone();
        measured.usd_per_hour = rate;
        *target = measured;
        Ok(())
    }
}
