//! Optional empirical environment. Samples are private physics inputs, not scheduler hints.
use crate::{Profile, Scenario};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Observations {
    pub profiles: BTreeMap<String, ObservedEnvironment>,
    /// Intervals during which the controller cannot dispatch; GPU work continues.
    pub controller_pauses: Vec<Interval>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObservedEnvironment {
    /// Ordered capacity observations; between observations the last limit applies.
    pub capacity: Vec<CapacityLimit>,
    /// Paired boot/hydration/readiness/first-use observations retain correlations.
    pub startups: Vec<StartupSample>,
    pub sampling: Sampling,
    pub rejection_delay_s: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampling {
    #[default]
    Sequential,
    Bootstrap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityLimit {
    pub at_s: f64,
    pub instances: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartupSample {
    pub boot_s: f64,
    pub hydration_s: f64,
    pub hydration_bytes: u64,
    pub ready_s: f64,
    pub first_use_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Interval {
    pub start_s: f64,
    pub end_s: f64,
}

impl Observations {
    pub fn validate(&self, profiles: &[Profile], _: &Scenario) -> Result<(), String> {
        let valid = |x: f64| x.is_finite() && x >= 0.0;
        for (id, reality) in &self.profiles {
            if !profiles.iter().any(|p| &p.id == id) || !valid(reality.rejection_delay_s) {
                return Err(
                    "observation references unknown profile or invalid rejection delay".into(),
                );
            }
            let mut previous = None;
            for limit in &reality.capacity {
                if !valid(limit.at_s) || previous.is_some_and(|t| t >= limit.at_s) {
                    return Err(
                        "capacity observations must have strictly increasing nonnegative times"
                            .into(),
                    );
                }
                previous = Some(limit.at_s);
            }
            for s in &reality.startups {
                if [s.boot_s, s.hydration_s, s.ready_s, s.first_use_s]
                    .into_iter()
                    .any(|x| !valid(x))
                    || s.ready_s < s.boot_s + s.hydration_s
                    || (s.hydration_bytes > 0 && s.hydration_s <= 0.0)
                {
                    return Err("invalid paired startup observation".into());
                }
            }
        }
        let mut end = 0.0;
        for interval in &self.controller_pauses {
            if !valid(interval.start_s)
                || !valid(interval.end_s)
                || interval.end_s <= interval.start_s
                || interval.start_s < end
            {
                return Err(
                    "controller pauses must be ordered, disjoint positive intervals".into(),
                );
            }
            end = interval.end_s;
        }
        Ok(())
    }
}
