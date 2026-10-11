//! Validated request plans shared by live evaluation and cache preparation.
use crate::{Error, Result, WorkCall, planner::Message};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use switchyard_core::Policy;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Experiment {
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub name: String,
    pub policy: Policy,
    pub calls: usize,
    pub prompt: String,
    pub planned_output: u64,
    pub max_output: u64,
    pub arrival_interval_s: f64,
    pub desired_output_tps: f64,
    pub completion_budget_s: f64,
    #[serde(default)]
    pub serial_dependencies: bool,
    /// An identical system message can exercise ordinary engine prefix reuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Put the case and request identity before the system content as a control.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub unique_prefix: bool,
    /// Cycle output bounds to exercise uncertainty without changing scheduling classes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_bounds: Vec<u64>,
}

impl Experiment {
    /// Validate the entire plan before any paid allocation is created.
    pub fn validate(&self) -> Result<()> {
        if self.cases.is_empty() {
            return Err(Error::Config("workload plan is empty".into()));
        }
        let mut names = BTreeSet::new();
        for case in &self.cases {
            if !names.insert(&case.name)
                || !switchyard_provider::config::valid_component(&case.name)
                || case.calls == 0
                || case.calls > 100_000
                || case.planned_output == 0
                || case.planned_output > case.max_output
                || case
                    .output_bounds
                    .iter()
                    .any(|n| *n == 0 || *n > case.max_output)
                || !case.arrival_interval_s.is_finite()
                || case.arrival_interval_s < 0.0
                || !case.desired_output_tps.is_finite()
                || case.desired_output_tps <= 0.0
                || !case.completion_budget_s.is_finite()
                || case.completion_budget_s <= 0.0
            {
                return Err(Error::Config(
                    "invalid workload name, count, token bounds, or timing".into(),
                ));
            }
        }
        Ok(())
    }
}

impl Case {
    pub fn request(&self, model: &str, id: usize) -> WorkCall {
        let mut messages = vec![];
        if let Some(system) = &self.system_prompt {
            messages.push(Message {
                role: "system".into(),
                content: if self.unique_prefix {
                    format!("Case {} request {id}.\n{system}", self.name)
                } else {
                    system.clone()
                },
            });
        }
        messages.push(Message {
            role: "user".into(),
            content: format!("{}\nRequest identifier: {id}.", self.prompt),
        });
        let max_output = if self.output_bounds.is_empty() {
            self.max_output
        } else {
            self.output_bounds[id % self.output_bounds.len()]
        };
        WorkCall {
            model: model.into(),
            messages,
            planned_output: self.planned_output.min(max_output),
            max_output,
            arrival_delay_s: id as f64 * self.arrival_interval_s,
            class: "standard".into(),
            depends_on: if self.serial_dependencies && id > 0 {
                vec![id - 1]
            } else {
                vec![]
            },
        }
    }
    pub fn requests(&self, model: &str) -> Vec<WorkCall> {
        (0..self.calls).map(|id| self.request(model, id)).collect()
    }
}
