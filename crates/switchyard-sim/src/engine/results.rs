//! Result aggregation is separate from runtime observability.
use super::*;

impl Simulation {
    pub(super) fn finish(self, seed: u64, controller_diagnostics: serde_json::Value) -> Run {
        let nodes: Vec<_> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(id, n)| {
                let p = &self.profiles[n.profile];
                let end = n.ended_s.unwrap_or(self.scenario.horizon_s);
                NodeResult {
                    host: n.placement.as_ref().map_or(id, |p| p.host),
                    endpoint: n.placement.as_ref().map(|p| p.endpoint.clone()),
                    id,
                    profile: p.id.clone(),
                    warm: n.warm,
                    allocated_s: n.allocated_s,
                    ready_s: n.ready_s,
                    ended_s: end,
                    busy_s: n.busy_s,
                    emitted_tokens: n.emitted,
                    usd: if n.placement.as_ref().is_none_or(|p| p.host == id) {
                        (end - n.allocated_s).max(p.minimum_bill_s) * p.usd_per_hour / 3600.0
                    } else {
                        0.0
                    },
                    failed: n.failed,
                }
            })
            .collect();
        let hosts: Vec<_> = nodes
            .iter()
            .filter(|n| n.host == n.id)
            .map(|owner| HostResult {
                id: owner.id,
                plan: self.nodes[owner.id]
                    .placement
                    .as_ref()
                    .map(|p| p.plan.clone()),
                endpoints: nodes
                    .iter()
                    .filter(|n| n.host == owner.id)
                    .map(|n| n.id)
                    .collect(),
                allocated_s: owner.allocated_s,
                ended_s: owner.ended_s,
                usd: owner.usd,
            })
            .collect();
        let results: Vec<_> = self.states.iter().map(|s| s.result.clone()).collect();
        let completed: Vec<_> = results
            .iter()
            .filter(|r| r.completed_s.is_some_and(|t| t <= self.scenario.horizon_s))
            .collect();
        let latency = |r: &RequestResult| {
            r.completed_s.expect("completed") - self.scenario.requests[r.id].arrival_s
        };
        let classes = self
            .scenario
            .scheduling
            .classes
            .keys()
            .map(|class| {
                let arrived = self
                    .scenario
                    .requests
                    .iter()
                    .filter(|r| r.class == *class && r.arrival_s <= self.scenario.horizon_s)
                    .count();
                let done: Vec<_> = completed.iter().filter(|r| r.class == *class).collect();
                ClassResult {
                    name: class.clone(),
                    arrived,
                    completed: done.len(),
                    latency_p95_s: percentile(done.iter().map(|r| latency(r)).collect(), 0.95),
                    slo_fraction: if arrived == 0 {
                        0.0
                    } else {
                        done.iter()
                            .filter(|r| latency(r) <= self.scenario.requests[r.id].latency_target_s)
                            .count() as f64
                            / arrived as f64
                    },
                }
            })
            .filter(|c| c.arrived > 0)
            .collect();
        let is_batch =
            |id: usize| self.scenario.jobs[self.scenario.requests[id].job].desired_output_tps > 0.0;
        let batch: Vec<_> = completed.iter().filter(|r| is_batch(r.id)).collect();
        let batch_start = self
            .scenario
            .requests
            .iter()
            .filter(|r| is_batch(r.id))
            .map(|r| r.arrival_s)
            .min_by(f64::total_cmp);
        let batch_end = batch
            .iter()
            .filter_map(|r| r.completed_s)
            .max_by(f64::total_cmp);
        let batch_elapsed = batch_start.zip(batch_end).map(|(a, b)| b - a);
        let batch_output: u64 = batch
            .iter()
            .map(|r| self.scenario.requests[r.id].actual_output)
            .sum();
        let jobs: Vec<_> = self
            .scenario
            .jobs
            .iter()
            .map(|j| {
                let requests: Vec<_> = self
                    .scenario
                    .requests
                    .iter()
                    .filter(|r| r.job == j.id)
                    .collect();
                let done: Vec<_> = completed.iter().filter(|r| r.job == j.id).collect();
                let end = if done.len() == requests.len() && !requests.is_empty() {
                    done.iter()
                        .filter_map(|r| r.completed_s)
                        .max_by(f64::total_cmp)
                } else {
                    None
                };
                let elapsed = end.map(|t| t - j.arrival_s);
                let output: u64 = done
                    .iter()
                    .map(|r| self.scenario.requests[r.id].actual_output)
                    .sum();
                let tps = elapsed
                    .filter(|t| *t > 0.0)
                    .map_or(0.0, |t| output as f64 / t);
                JobResult {
                    name: j.name.clone(),
                    requests: requests.len(),
                    completed: done.len(),
                    completion_s: end,
                    elapsed_s: elapsed,
                    output_tps: tps,
                    desired_output_tps: j.desired_output_tps,
                    target_met: elapsed.is_some_and(|t| t <= j.completion_budget_s)
                        && (j.desired_output_tps == 0.0 || tps >= j.desired_output_tps),
                }
            })
            .collect();
        let out: u64 = completed
            .iter()
            .map(|r| self.scenario.requests[r.id].actual_output)
            .sum();
        let rejected = results.iter().filter(|r| r.rejected).count();
        let cost: f64 = nodes.iter().map(|n| n.usd).sum();
        let idle_cost = nodes
            .iter()
            .map(|n| {
                let duration = (n.ended_s - n.allocated_s)
                    .max(self.profiles[self.nodes[n.id].profile].minimum_bill_s);
                n.usd
                    * (1.0
                        - nodes
                            .iter()
                            .filter(|other| other.host == n.host)
                            .map(|other| {
                                other.busy_s
                                    * self.nodes[other.id]
                                        .placement
                                        .as_ref()
                                        .map_or(1.0, |p| p.capacity_fraction)
                            })
                            .sum::<f64>()
                            / duration.max(1e-8))
                    .max(0.0)
            })
            .sum();
        let mut profiles_used: Vec<_> = nodes.iter().map(|n| n.profile.clone()).collect();
        profiles_used.sort();
        profiles_used.dedup();
        let summary = Summary {
            scenario: self.scenario.name.clone(),
            policy: self.policy,
            seed,
            completed: completed.len(),
            rejected,
            unfinished: results.len() - completed.len() - rejected,
            cost_usd: cost,
            idle_cost_usd: idle_cost,
            classes,
            queue_p95_s: percentile(
                results
                    .iter()
                    .filter_map(|r| {
                        r.started_s
                            .map(|t| t - self.scenario.requests[r.id].arrival_s)
                    })
                    .collect(),
                0.95,
            ),
            first_token_p95_s: percentile(
                results
                    .iter()
                    .filter_map(|r| {
                        r.first_token_s
                            .filter(|t| *t <= self.scenario.horizon_s)
                            .map(|t| t - self.scenario.requests[r.id].arrival_s)
                    })
                    .collect(),
                0.95,
            ),
            batch_elapsed_s: batch_elapsed,
            batch_output_tps: batch_elapsed
                .filter(|t| *t > 0.0)
                .map_or(0.0, |t| batch_output as f64 / t),
            launched: self.launches,
            peak_nodes: self.peak_nodes,
            failed_launches: self.failed_launches,
            capacity_rejections: self.capacity_rejections,
            attempts: results.iter().map(|r| r.attempts).sum(),
            output_tokens: out,
            wasted_output_tokens: self.wasted_output,
            incomplete_output_tokens: (nodes.iter().map(|n| n.emitted_tokens).sum::<f64>()
                - out as f64
                - self.wasted_output)
                .max(0.0),
            target_jobs_met: jobs
                .iter()
                .filter(|j| j.desired_output_tps > 0.0 && j.target_met)
                .count(),
            target_jobs_total: jobs.iter().filter(|j| j.desired_output_tps > 0.0).count(),
            max_active_per_node: self.max_active,
            max_reserved_kv_tokens: self.max_kv,
            profiles_used,
            jobs,
            invalid_decisions: self.invalid_decisions,
        };
        Run {
            hosts,
            summary,
            requests: results,
            nodes,
            timeline: self.timeline,
            controller_diagnostics,
        }
    }
}
