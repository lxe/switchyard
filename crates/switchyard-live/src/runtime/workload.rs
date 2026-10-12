//! Planned request admission, scheduler actions, and real streaming completions.
use super::{LiveRuntime, WorkCall, WorkloadReport};
use crate::{Error, Result, serving::Event};
use std::{collections::BTreeMap, time::Duration};
use switchyard_core::{
    Action, FleetView, Job, Workload,
    scheduler::{ObservedRequest, PlannedRequest, RequestStatus},
};
use switchyard_provider::Provider;
use tokio::sync::mpsc;

impl<P: Provider> LiveRuntime<P> {
    pub async fn run_workload(
        &mut self,
        name: &str,
        calls: &[WorkCall],
        desired_tps: f64,
        completion_budget_s: f64,
    ) -> Result<WorkloadReport> {
        let outcome = self
            .execute_workload(name, calls, desired_tps, completion_budget_s)
            .await;
        if let Err(error) = &outcome {
            self.abort_requests().await;
            self.save("last-workload-error.json",&serde_json::json!({"workload":name,"error":error.to_string(),"owned_pods":self.pods(),"agent_observations":self.observations}))?;
        }
        outcome
    }
    async fn execute_workload(
        &mut self,
        name: &str,
        calls: &[WorkCall],
        desired_tps: f64,
        completion_budget_s: f64,
    ) -> Result<WorkloadReport> {
        if calls.is_empty()
            || calls.len() > 100_000
            || !desired_tps.is_finite()
            || desired_tps <= 0.0
            || !completion_budget_s.is_finite()
            || completion_budget_s <= 0.0
            || name.is_empty()
            || name.contains(['/', '\\'])
        {
            return Err(Error::Config(
                "invalid workload size, name, or target".into(),
            ));
        }
        std::fs::create_dir_all(self.out.join("workloads"))?;
        let begin = self.now();
        let initial_cost = self.rental_usd();
        let counters = (self.launches, self.probes, self.releases, self.invalid);
        let mut requests = BTreeMap::new();
        let mut states = BTreeMap::new();
        for (id, call) in calls.iter().enumerate() {
            let model = self
                .config
                .models
                .iter()
                .find(|m| m.id == call.model)
                .ok_or_else(|| Error::Config("unknown workload model".into()))?;
            let count = self.planners[&call.model].input_tokens(&call.messages)?;
            if call.planned_output == 0
                || call.planned_output > call.max_output
                || call.max_output == 0
                || count.saturating_add(call.max_output) > model.engine.max_model_len
                || !call.arrival_delay_s.is_finite()
                || call.arrival_delay_s < 0.0
                || !self.scheduling.classes.contains_key(&call.class)
                || call.depends_on.iter().any(|d| *d >= id)
            {
                return Err(Error::Config(
                    "invalid workload inputs or dependencies".into(),
                ));
            }
            requests.insert(
                id,
                PlannedRequest {
                    routing_allowed: true,
                    id,
                    job: 0,
                    model: call.model.clone(),
                    workload: Workload::Text,
                    class: call.class.clone(),
                    arrival_s: begin + call.arrival_delay_s,
                    input_tokens: count,
                    planned_output: call.planned_output,
                    max_output: call.max_output,
                    image_megapixels: 0.0,
                    upload_bytes: serde_json::to_vec(&call.messages)?.len() as u64,
                    cached_fraction: 0.0,
                    adapter: None,
                    latency_target_s: completion_budget_s,
                    depends_on: call.depends_on.clone(),
                },
            );
            states.insert(
                id,
                ObservedRequest {
                    state: RequestStatus::Future,
                    completed_s: None,
                },
            );
        }
        let jobs = BTreeMap::from([(
            0,
            Job {
                id: 0,
                name: name.into(),
                announced_s: begin,
                arrival_s: begin,
                completion_budget_s,
                desired_output_tps: desired_tps,
            },
        )]);
        let (tx, mut rx) = mpsc::channel::<Event>(4096);
        let mut completions = vec![];
        let mut failures = BTreeMap::new();
        let mut attempts = vec![0usize; calls.len()];
        let mut last_scale = f64::NEG_INFINITY;
        let mut next_observation = begin;
        let mut budget_exceeded = false;
        let outcome: Result<()> = async {
            loop {
                self.limits()?;
                self.finish_observations()?;
                let now = self.now();
                for (id, request) in &requests {
                    if states[id].state == RequestStatus::Future && request.arrival_s <= now {
                        states.get_mut(id).expect("known request").state = RequestStatus::Queued;
                    }
                }
                while let Ok(event) = rx.try_recv() {
                    match event {
                        Event::Progress { request, generated } => {
                            for node in &mut self.nodes {
                                if let Some(a) =
                                    node.observed.active.iter_mut().find(|a| a.id == request)
                                {
                                    a.generated = generated as f64;
                                    a.prefill = 0.0;
                                }
                            }
                        }
                        Event::Completed { request, result } => {
                            for node in &mut self.nodes {
                                if node.observed.active.iter().any(|a| a.id == request) {
                                    if result.is_ok() { node.successful_calls = node.successful_calls.saturating_add(1); }
                                    node.observed.active.retain(|a| a.id != request);
                                    node.observed.last_used_s = now;
                                }
                            }
                            let observed = states.get_mut(&request).expect("known request");
                            match result {
                                Ok(receipt) => {
                                    observed.state = RequestStatus::Done;
                                    observed.completed_s = Some(receipt.completed_s);
                                    completions.push(receipt);
                                }
                                Err(error) if attempts[request] < 2 => {
                                    observed.state = RequestStatus::Queued;
                                    failures.insert(request, error.to_string());
                                }
                                Err(error) => {
                                    observed.state = RequestStatus::Rejected;
                                    failures.insert(request, error.to_string());
                                }
                            }
                        }
                    }
                }
                if states
                    .values()
                    .all(|s| matches!(s.state, RequestStatus::Done | RequestStatus::Rejected))
                {
                    break;
                }
                if now >= next_observation {
                    self.observe().await?;
                    next_observation = self.now() + 1.0;
                }
                let now = self.now();
                let allow_scale = now - last_scale >= self.scheduling.plan_interval_s;
                if allow_scale {
                    last_scale = now;
                }
                let nodes = self.nodes.iter().map(|n| n.observed.clone()).collect();
                let fleet = FleetView {
                    now,
                    allow_scale,
                    profiles: &self.profiles,
                    provisioning_cooldowns: self.provisioning_cooldowns.clone(),
                    requests: requests.clone(),
                    states: states.clone(),
                    nodes,
                    jobs: jobs.clone(),
                    scheduling: &self.scheduling,
                    max_nodes_per_model: self.config.limits.max_instances,
                    idle_timeout_s: 30.0,
                    reactive_delay_s: 0.0,
                    build_probability: 0.0,
                };
                while self.tasks.try_join_next().is_some() {}
                let actions = self.controller.decide(&fleet);
                if !actions.is_empty() {
                    let nodes = fleet.nodes.iter().enumerate().map(|(id, node)| serde_json::json!({
                        "id": id, "profile": node.profile, "allocated_s": node.allocated_s,
                        "ready_s": node.ready_s, "ended_s": node.ended_s, "is_ready": node.is_ready,
                        "active_requests": node.active.iter().map(|request| request.id).collect::<Vec<_>>(),
                        "host": node.host,
                    })).collect::<Vec<_>>();
                    super::telemetry::append(
                        &self.out,
                        "scheduler-decisions.ndjson",
                        &serde_json::json!({"at_s": fleet.now, "workload": name,
                            "scheduler": self.controller.name(), "actions": actions,
                            "nodes": nodes, "scheduling": self.scheduling, "diagnostics": self.controller.diagnostics()}),
                    )?;
                }
                for action in actions {
                    match action {
                        Action::Provision { profile } => {
                            if !self.provision(profile).await? {
                                // Later actions can refer to the planned new node. Replan
                                // from observed hosts if that allocation did not happen.
                                break;
                            }
                        }
                        Action::Release { node } => {
                            if self
                                .nodes
                                .get(node)
                                .is_some_and(|n| n.observed.active.is_empty())
                            {
                                self.release(node).await?;
                            } else {
                                self.invalid += 1;
                            }
                        }
                        Action::ProbeHost { node } => self.probe(node).await?,
                        Action::Route { request, node, .. } => {
                            self.route_request(
                                request,
                                node,
                                super::routing::RouteContext {
                                    now,
                                    calls,
                                    requests: &requests,
                                    states: &mut states,
                                    attempts: &mut attempts,
                                    events: &tx,
                                },
                            )
                            .await?;
                        }
                    }
                }
                if self.now() - begin > completion_budget_s {
                    budget_exceeded = true;
                    self.abort_requests().await;
                    break;
                }
                tokio::time::sleep(Duration::from_millis(self.config.limits.poll_ms)).await;
            }
            Ok(())
        }
        .await;
        if let Err(error) = outcome {
            // Retain completions already observed before any provider, host or
            // routing error. Cancellation and owned-pod cleanup still happen in
            // the caller; an interrupted workload must not look fully completed.
            self.save(
                &format!("workloads/{name}-partial.json"),
                &serde_json::json!({
                    "error":error.to_string(), "requests":requests.len(),
                    "completed":completions.len(), "completions":completions,
                    "failures":failures, "attempts":attempts,
                    "states":states.iter().map(|(id,s)|(*id,format!("{:?}",s.state))).collect::<BTreeMap<_,_>>(),
                    "observed_at_s":self.now(), "elapsed_s":self.now()-begin,
                    "estimated_rental_usd":self.rental_usd()-initial_cost,
                }),
            )?;
            return Err(error);
        }
        failures.retain(|id, _| states[id].state == RequestStatus::Rejected);
        let report = super::report::summarize(
            super::report::RunMetrics {
                name: name.into(),
                policy: self.controller.name().into(),
                target_tps: desired_tps,
                budget_exceeded,
                begin,
                finish: self.now(),
                rental_usd: self.rental_usd() - initial_cost,
                invalid_actions: self.invalid - counters.3,
                launches: self.launches - counters.0,
                probes: self.probes - counters.1,
                releases: self.releases - counters.2,
                controller: self.controller.diagnostics(),
            },
            &requests,
            completions,
            failures,
        );
        std::fs::create_dir_all(self.out.join("workloads"))?;
        self.save(&format!("workloads/{name}.json"), &report)?;
        Ok(report)
    }
}
