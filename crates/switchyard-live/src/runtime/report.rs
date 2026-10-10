//! Summaries use observed completion times and actual token usage.
use super::WorkloadReport;
use crate::serving::Completion;
use std::collections::BTreeMap;
use switchyard_core::{scheduler::PlannedRequest, stats::percentile};

pub(super) struct RunMetrics {
    pub(super) name: String,
    pub(super) policy: String,
    pub(super) target_tps: f64,
    pub(super) budget_exceeded: bool,
    pub(super) begin: f64,
    pub(super) finish: f64,
    pub(super) rental_usd: f64,
    pub(super) invalid_actions: usize,
    pub(super) launches: usize,
    pub(super) probes: usize,
    pub(super) releases: usize,
    pub(super) controller: serde_json::Value,
}

pub(super) fn summarize(
    run: RunMetrics,
    requests: &BTreeMap<usize, PlannedRequest>,
    completions: Vec<Completion>,
    failures: BTreeMap<usize, String>,
) -> WorkloadReport {
    let elapsed = run.finish - run.begin;
    let output: u64 = completions.iter().map(|c| c.output_tokens).sum();
    let ttft = completions
        .iter()
        .map(|c| c.first_content_s - requests[&c.request].arrival_s)
        .collect::<Vec<_>>();
    let response = completions
        .iter()
        .map(|c| c.completed_s - requests[&c.request].arrival_s)
        .collect();
    WorkloadReport {
        name: run.name,
        policy: run.policy,
        requests: requests.len(),
        completed: completions.len(),
        failed: failures.len(),
        unfinished: requests.len() - completions.len() - failures.len(),
        target_output_tokens_per_s: run.target_tps,
        budget_exceeded: run.budget_exceeded,
        output_tokens: output,
        output_tokens_per_s: output as f64 / elapsed,
        ttft_p50_s: percentile(ttft.clone(), 0.5),
        ttft_p95_s: percentile(ttft, 0.95),
        response_p95_s: percentile(response, 0.95),
        elapsed_s: elapsed,
        estimated_rental_usd: run.rental_usd,
        invalid_actions: run.invalid_actions,
        launches: run.launches,
        probes: run.probes,
        releases: run.releases,
        controller: run.controller,
        completions,
        failures,
    }
}
