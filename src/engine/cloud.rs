//! Provider facade and request admission over the physical event engine.
use super::*;
use serde::Deserialize;

/// Known inference inputs accepted by a provisioned instance. Output length is
/// sampled privately; `max_output` remains the hard generation and memory limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceSpec {
    pub model: String,
    pub input_tokens: u64,
    pub planned_output: u64,
    pub max_output: u64,
    pub upload_bytes: u64,
    pub workload: Workload,
    pub image_megapixels: f64,
    pub adapter: Option<String>,
    pub class: String,
    pub cached_fraction: f64,
}

/// Incremental cloud-provider simulator. All mutations advance the same event
/// engine used by trace replay; polling never regenerates or reruns history.
pub struct CloudSimulator {
    simulation: Simulation,
    runtime: Option<crate::runtime::RoutingRuntime>,
    config: CloudConfig,
}

impl CloudSimulator {
    pub fn new(profiles: Vec<Profile>, seed: u64) -> Result<Self, String> {
        Self::with_config(profiles, CloudConfig::default(), seed)
    }

    pub fn with_config(
        profiles: Vec<Profile>,
        config: CloudConfig,
        seed: u64,
    ) -> Result<Self, String> {
        if config.instance_queue_limit == 0
            || config.router_queue_limit == 0
            || !config.request_completion_budget_s.is_finite()
            || config.request_completion_budget_s <= 0.0
        {
            return Err("invalid cloud queue limits or request completion budget".into());
        }
        let scenario = Scenario {
            name: "cloud".into(),
            notes: "Externally controlled virtual fleet".into(),
            horizon_s: 3600.0,
            tick_s: config.tick_s,
            idle_timeout_s: config.idle_timeout_s,
            max_nodes_per_model: config.max_instances_per_model,
            launch_fail_probability: config.launch_fail_probability,
            reactive_delay_s: 0.0,
            capacity_scale: config.capacity_scale,
            startup_scale: config.startup_scale,
            build_probability: config.build_probability,
            scheduling: config.scheduling.clone(),
            host_reality: config.host_reality.clone(),
            warm: config.warm.clone(),
            jobs: vec![],
            requests: vec![],
            faults: config.faults.clone(),
        };
        validate(&profiles, &scenario)?;
        let mut simulation = Simulation::new(profiles, scenario, seed, "external".into());
        simulation.record_delivery = true;
        Ok(Self {
            simulation,
            runtime: None,
            config,
        })
    }

    pub fn clock(&self) -> f64 {
        self.simulation.now
    }

    /// Advance simulated seconds. The per-call limit bounds work in HTTP clock
    /// controls; long experiments may issue repeated advances.
    pub fn advance(&mut self, seconds: f64) -> Result<(), String> {
        if !seconds.is_finite()
            || !(0.0..=3600.0).contains(&seconds)
            || self.clock() + seconds > 1e9
        {
            return Err("invalid advance: use finite seconds between 0 and 3600".into());
        }
        let until = self.clock() + seconds;
        if let Some(runtime) = &mut self.runtime {
            self.simulation
                .run_until(until, Some(runtime.controller.as_mut()));
        } else {
            self.simulation.run_until(until, None);
        }
        Ok(())
    }

    pub fn offerings(&self) -> serde_json::Value {
        serde_json::json!({"offerings": self.simulation.profiles})
    }

    /// Observable provider state. Private readiness, sampled caches, bandwidth,
    /// and future output are deliberately absent until measured or delivered.
    pub fn snapshot(&self) -> serde_json::Value {
        let sim = &self.simulation;
        let instances: Vec<_> = sim.nodes.iter().enumerate().map(|(id, n)| {
            let host = sim.host_observation(n);
            let ready = n.ended_s.is_none() && n.initialized;
            serde_json::json!({"id": id, "profile_id": sim.profiles[n.profile].id,
                "endpoint": format!("/v1/instances/{id}/openai/v1"),
                "status": if n.failed { "failed" } else if n.ended_s.is_some() { "released" } else if ready { "ready" } else { "starting" },
                "allocated_at_s": n.allocated_s, "ready_at_s": if ready { Some(n.ready_s) } else { None },
                "released_at_s": n.ended_s, "active_requests": n.active.iter().map(|a| a.id).collect::<Vec<_>>(),
                "host": host})
        }).collect();
        let requests: Vec<_> = (0..sim.states.len())
            .map(|id| self.request(id).expect("existing request"))
            .collect();
        serde_json::json!({"time_s": sim.now, "instances": instances, "requests": requests})
    }

    pub fn provision(&mut self, profile_id: &str) -> Result<usize, String> {
        let sim = &mut self.simulation;
        let profile = sim
            .profiles
            .iter()
            .position(|p| p.id == profile_id)
            .ok_or("unknown profile")?;
        let live = sim
            .nodes
            .iter()
            .filter(|n| {
                n.ended_s.is_none() && sim.profiles[n.profile].model == sim.profiles[profile].model
            })
            .count();
        if live >= sim.scenario.max_nodes_per_model {
            return Err("capacity: instance quota reached".into());
        }
        let id = sim.nodes.len();
        sim.launch(profile, false);
        Ok(id)
    }

    pub fn release(&mut self, id: usize) -> Result<(), String> {
        let sim = &mut self.simulation;
        if sim
            .states
            .iter()
            .any(|s| s.node == Some(id) && s.result.completed_s.is_some_and(|t| t > sim.now))
        {
            return Err("busy: instance has undelivered responses".into());
        }
        let node = sim.nodes.get_mut(id).ok_or("unknown instance")?;
        if !node.active.is_empty() {
            return Err("busy: instance has active requests".into());
        }
        node.ended_s.get_or_insert(sim.now);
        sim.dispatch_pinned();
        Ok(())
    }

    pub fn probe(&mut self, id: usize) -> Result<(), String> {
        let sim = &mut self.simulation;
        let node = sim.nodes.get_mut(id).ok_or("unknown instance")?;
        if node.ended_s.is_some() || sim.now < node.host.setup_until_s {
            return Err("unavailable: host probe is not ready".into());
        }
        if node.host.probe_pending {
            return Err("busy: probe pending".into());
        }
        node.host.probe_pending = true;
        let done = sim.now + node.host.probe_duration_s;
        sim.push(done, EventKind::ProbeDone(id));
        Ok(())
    }

    pub fn submit(&mut self, node: usize, spec: InferenceSpec) -> Result<usize, String> {
        self.accept(Some(node), spec, false)
    }

    /// Submit through a serving endpoint. Requests wait for model readiness and
    /// admission slots; queued input does not consume GPU work or KV capacity.
    pub fn enqueue(&mut self, node: usize, spec: InferenceSpec) -> Result<usize, String> {
        self.accept(Some(node), spec, true)
    }

    /// Attach an independent controller to the convenience routing front door.
    pub fn set_router(&mut self, runtime: crate::runtime::RoutingRuntime) {
        self.runtime = Some(runtime);
    }

    /// Submit a model request to the router's global queue. The controller picks
    /// and provisions hosts through the same observable contract as replay.
    pub fn route(&mut self, spec: InferenceSpec) -> Result<usize, String> {
        let id = self.accept(None, spec, true)?;
        let runtime = self
            .runtime
            .get_or_insert_with(|| crate::runtime::RoutingRuntime::new(self.config.router_policy));
        let allow_scale = self.simulation.now >= self.simulation.next_plan;
        if allow_scale {
            self.simulation.next_plan =
                self.simulation.now + self.simulation.scenario.scheduling.plan_interval_s;
        }
        self.simulation
            .control(runtime.controller.as_mut(), allow_scale);
        Ok(id)
    }

    fn accept(
        &mut self,
        node: Option<usize>,
        spec: InferenceSpec,
        queue: bool,
    ) -> Result<usize, String> {
        let sim = &mut self.simulation;
        let instance = node
            .map(|node| sim.nodes.get(node).ok_or("unknown instance"))
            .transpose()?;
        if instance
            .is_some_and(|instance| instance.ended_s.is_some() || (!queue && !instance.initialized))
        {
            return Err("unavailable: instance not ready".into());
        }
        if queue
            && node.is_some_and(|node| {
                sim.pinned.iter().filter(|(_, n)| *n == node).count()
                    >= self.config.instance_queue_limit
            })
        {
            return Err("capacity: instance queue limit reached".into());
        }
        if node.is_none()
            && sim
                .states
                .iter()
                .filter(|s| !s.pinned && s.state == State::Queued)
                .count()
                >= self.config.router_queue_limit
        {
            return Err("capacity: router queue limit reached".into());
        }
        let id = sim.states.len();
        let job = sim.scenario.jobs.len();
        let mut request = Request {
            id: 0,
            job: 0,
            model: spec.model,
            workload: spec.workload,
            class: spec.class,
            arrival_s: sim.now,
            input_tokens: spec.input_tokens,
            planned_output: spec.planned_output,
            max_output: spec.max_output,
            actual_output: 1,
            image_megapixels: spec.image_megapixels,
            upload_bytes: spec.upload_bytes,
            cached_fraction: spec.cached_fraction,
            adapter: spec.adapter,
            latency_target_s: self.config.request_completion_budget_s,
            depends_on: vec![],
        };
        let mut check = sim.scenario.clone();
        let intent = Job {
            id: 0,
            name: format!("request-{id}"),
            announced_s: sim.now,
            arrival_s: sim.now,
            completion_budget_s: self.config.request_completion_budget_s,
            desired_output_tps: 0.0,
        };
        check.requests = vec![request.clone()];
        check.jobs = vec![intent.clone()];
        validate(&sim.profiles, &check)?;
        let compatible = instance.map_or_else(
            || sim.profiles.iter().any(|p| p.compatible(&request)),
            |instance| sim.profiles[instance.profile].compatible(&request),
        );
        if !compatible {
            return Err("invalid request: model or memory incompatible with instance".into());
        }
        // A bounded 0.5..1.5 multiplier exercises planning uncertainty without
        // revealing a fixture's actual output through this public interface.
        let output_rng_before = sim.output_random.0;
        request.actual_output = ((request.planned_output as f64 * (0.5 + sim.output_random.unit()))
            .round() as u64)
            .clamp(1, request.max_output);
        request.id = id;
        request.job = job;
        sim.plans.push(PlannedRequest::from(&request));
        sim.scenario.requests.push(request);
        sim.scenario.jobs.push(Job { id: job, ..intent });
        sim.states.push(RequestState {
            state: State::Queued,
            result: RequestResult {
                id,
                job,
                class: sim.scenario.requests[id].class.clone(),
                started_s: None,
                first_token_s: None,
                completed_s: None,
                attempts: 0,
                rejected: false,
            },
            delivery: vec![],
            egress_until_s: 0.0,
            node: None,
            pinned: node.is_some(),
        });
        if queue {
            if let Some(node) = node {
                sim.pinned.push_back((id, node));
                sim.dispatch_pinned();
            }
        } else if !sim.admit(id, node.expect("direct admission requires instance"), 1.0) {
            sim.output_random.0 = output_rng_before;
            sim.plans.pop();
            sim.scenario.requests.pop();
            sim.scenario.jobs.pop();
            sim.states.pop();
            return Err("capacity: instance admission limit reached".into());
        }
        Ok(id)
    }

    pub fn request(&self, id: usize) -> Result<serde_json::Value, String> {
        let sim = &self.simulation;
        let state = sim.states.get(id).ok_or("unknown request")?;
        let result = &state.result;
        let completed = result.completed_s.filter(|t| *t <= sim.now);
        let first = result.first_token_s.filter(|t| *t <= sim.now);
        let generated = state
            .delivery
            .iter()
            .take_while(|segment| segment.start_s <= sim.now)
            .last()
            .map_or(0, |segment| {
                let fraction = ((sim.now - segment.start_s)
                    / (segment.end_s - segment.start_s).max(1e-12))
                .clamp(0.0, 1.0);
                (segment.before + (segment.after - segment.before) * fraction + 1e-8).floor() as u64
            });
        Ok(
            serde_json::json!({"id": id, "submitted_at_s": sim.scenario.requests[id].arrival_s, "status": if completed.is_some() { "completed" } else if state.state == State::Rejected { "failed" } else if state.state == State::Queued { "queued" } else { "active" },
            "started_at_s": result.started_s, "first_token_at_s": first, "completed_at_s": completed,
            "generated_tokens": generated, "usage": completed.map(|_| serde_json::json!({"prompt_tokens": sim.scenario.requests[id].input_tokens,
                "completion_tokens": sim.scenario.requests[id].actual_output}))}),
        )
    }
}
