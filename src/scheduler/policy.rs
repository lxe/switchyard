//! Scaling and routing only observe input, plans, and generated token counts.
//! Actual future output remains confined to the engine's service model.
use super::*;
use std::collections::BTreeMap;

impl SchedulerWorld<'_> {
    fn estimated_remaining(&self, id: usize, profile: &Profile) -> f64 {
        let request = &self.scenario.requests[&(id)];
        self.nodes
            .iter()
            .flat_map(|n| &n.active)
            .find(|a| a.id == id)
            .map_or_else(
                || profile.prefill_work(request) + request.planned_output as f64,
                |a| {
                    a.prefill
                        + (request.planned_output as f64 - a.generated)
                            .max(self.scenario.scheduling.replan_output_floor)
                },
            )
    }

    pub(super) fn predicted_start(&self, profile: &Profile) -> f64 {
        let quantile = if self.policy.plans_ahead() { 0.9 } else { 0.5 };
        let restore = percentile(profile.startup.restore_samples_s.clone(), quantile).unwrap_or(
            profile.startup.total()
                * if self.policy.plans_ahead() {
                    1.0 + profile.startup.jitter
                } else {
                    1.0
                },
        );
        let build = percentile(profile.startup.build_samples_s.clone(), quantile)
            .unwrap_or(profile.startup.total() * 8.0);
        restore * (1.0 - self.scenario.build_probability) + build * self.scenario.build_probability
    }

    pub(super) fn plan(&mut self) {
        if self.policy == Policy::WarmOnly {
            return;
        }
        let mut pools: BTreeMap<(String, Workload), Vec<usize>> = BTreeMap::new();
        for request in self.scenario.requests.values() {
            if !request.routing_allowed {
                continue;
            }
            let state = self.states[&request.id].state;
            if matches!(state, State::Done | State::Rejected) {
                continue;
            }
            let job = &self.scenario.jobs[&(request.job)];
            let class = &self.scenario.scheduling.classes[&request.class];
            let known = if self.policy.plans_ahead() && class.plan_ahead {
                job.announced_s <= self.now
            } else {
                request.arrival_s
                    + if self.policy.plans_ahead() {
                        0.0
                    } else {
                        self.scenario.reactive_delay_s
                    }
                    <= self.now
            };
            if known {
                pools
                    .entry((request.model.clone(), request.workload))
                    .or_default()
                    .push(request.id);
            }
        }
        // Shared demand is planned once per compatible pool, including all jobs.
        for ((model, workload), ids) in pools {
            let candidates: Vec<_> = self
                .profiles
                .iter()
                .enumerate()
                .filter(|(i, p)| {
                    !self.avoided_profiles.contains(i)
                        && ids
                            .iter()
                            .all(|id| p.compatible(&self.scenario.requests[id]))
                })
                .map(|(i, _)| i)
                .collect();
            if candidates.is_empty() {
                continue;
            }
            let first_arrival = ids
                .iter()
                .map(|id| self.scenario.requests[id].arrival_s)
                .min_by(f64::total_cmp)
                .expect("nonempty pool")
                .max(self.now);
            let deadline = ids
                .iter()
                .map(|id| {
                    let job = &self.scenario.jobs[&self.scenario.requests[id].job];
                    let output: f64 = self
                        .scenario
                        .requests
                        .values()
                        .filter(|r| r.job == job.id)
                        .map(|r| r.planned_output as f64)
                        .sum();
                    job.arrival_s
                        + job
                            .completion_budget_s
                            .min(if job.desired_output_tps > 0.0 {
                                output / job.desired_output_tps
                            } else {
                                job.completion_budget_s
                            })
                })
                .min_by(f64::total_cmp)
                .expect("nonempty pool");
            let window = if self.policy == Policy::Reactive {
                self.scenario.scheduling.reactive_drain_s
            } else {
                (deadline - first_arrival).max(self.scenario.scheduling.minimum_plan_window_s)
            };
            let end = first_arrival + window;
            let reference = &self.profiles[candidates[0]];
            let work: f64 = ids
                .iter()
                .map(|id| self.estimated_remaining(*id, reference))
                .sum();
            let work_for = |profile: &Profile| {
                ids.iter()
                    .map(|id| self.estimated_remaining(*id, profile))
                    .sum::<f64>()
                    .max(1e-8)
            };
            let capacity_for = |profile: &Profile| {
                let has_adapters = ids
                    .iter()
                    .filter_map(|id| self.scenario.requests[id].adapter.as_deref())
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    > 1;
                profile.capacity(ids.len().min(profile.max_in_flight)) * 1.0 * work
                    / work_for(profile)
                    * if has_adapters {
                        profile.multi_adapter_factor
                    } else {
                        1.0
                    }
            };
            let live: Vec<_> = self
                .nodes
                .iter()
                .filter(|n| {
                    n.ended_s.is_none()
                        && self.profiles[n.profile].model == model
                        && self.profiles[n.profile].workload == workload
                })
                .collect();
            let mut capacity: f64 = live
                .iter()
                .map(|n| {
                    let p = &self.profiles[n.profile];
                    capacity_for(p)
                        * (end - first_arrival.max(n.ready_s)).max(0.0)
                        * if n.warm && self.policy.plans_ahead() {
                            self.scenario.scheduling.warm_batch_capacity_fraction
                        } else {
                            1.0
                        }
                })
                .sum();
            let total_model_nodes = self
                .nodes
                .iter()
                .filter(|n| n.ended_s.is_none() && self.profiles[n.profile].model == model)
                .count();
            let available = self
                .scenario
                .max_nodes_per_model
                .saturating_sub(total_model_nodes);
            let mut chosen = vec![];
            for _ in 0..available {
                if capacity >= work {
                    break;
                }
                let best = candidates
                    .iter()
                    .copied()
                    .min_by(|a, b| {
                        let score = |i: usize| {
                            let p = &self.profiles[i];
                            let startup = self.predicted_start(p);
                            let useful = (end - first_arrival.max(self.now + startup)).max(0.0)
                                * capacity_for(p);
                            let cost = p.usd_per_hour
                                * (window + startup + self.scenario.idle_timeout_s)
                                    .max(p.minimum_bill_s)
                                / 3600.0;
                            if self.policy == Policy::CheapestElastic {
                                p.usd_per_hour / capacity_for(p).max(1e-8)
                            } else if useful > 0.0 {
                                cost / useful
                            } else {
                                1e6 + startup / capacity_for(p).max(1e-8)
                            }
                        };
                        score(*a).total_cmp(&score(*b)).then(a.cmp(b))
                    })
                    .expect("nonempty candidates");
                let p = &self.profiles[best];
                let startup = self.predicted_start(p);
                // Announced jobs buy capacity close to arrival, avoiding idle churn.
                if self.now + startup + self.scenario.scheduling.plan_interval_s < first_arrival {
                    break;
                }
                capacity +=
                    capacity_for(p) * (end - first_arrival.max(self.now + startup)).max(0.0);
                chosen.push(best);
            }
            for profile in chosen {
                self.launch(profile, false);
            }
        }
    }

    fn node_for(&self, id: usize) -> Option<usize> {
        let request = &self.scenario.requests[&(id)];
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                let p = &self.profiles[node.profile];
                let used = node.active.iter().try_fold(0_u64, |total, a| {
                    let r = &self.scenario.requests[&(a.id)];
                    total.checked_add(r.input_tokens + r.max_output)
                });
                let reserve = if self.policy.plans_ahead()
                    && node.warm
                    && !self.scenario.scheduling.classes[&request.class].can_use_reserved_warm_slots
                {
                    self.scenario
                        .scheduling
                        .reserve_warm_slots
                        .min(p.max_in_flight.saturating_sub(1))
                } else {
                    0
                };
                let adapter_fits = request.adapter.as_ref().is_none_or(|adapter| {
                    node.adapters.iter().any(|(a, _)| a == adapter)
                        || node.adapters.len() < p.adapter_cache_slots
                        || node.adapters.iter().any(|(a, _)| {
                            !node.active.iter().any(|active| {
                                self.scenario.requests[&(active.id)].adapter.as_ref() == Some(a)
                            })
                        })
                });
                node.ended_s.is_none()
                    && node.is_ready
                    && p.compatible(request)
                    && node.active.len() < p.max_in_flight - reserve
                    && adapter_fits
                    && used
                        .and_then(|used| {
                            used.checked_add(request.input_tokens + request.max_output)
                        })
                        .is_some_and(|used| used <= p.kv_tokens)
            })
            .min_by(|(ia, a), (ib, b)| {
                let score = |node: &ObservedNode| {
                    let p = &self.profiles[node.profile];
                    let work: f64 = node
                        .active
                        .iter()
                        .map(|a| self.estimated_remaining(a.id, p))
                        .sum();
                    (work + p.prefill_work(request) + request.planned_output as f64)
                        / (p.capacity(node.active.len() + 1) * 1.0)
                        + p.network_s * 2.0
                        + request.upload_bytes as f64 / p.network_bytes_per_s
                };
                score(a).total_cmp(&score(b)).then(ia.cmp(ib))
            })
            .map(|(i, _)| i)
    }

    pub(super) fn dispatch(&mut self) {
        let mut active_per_job: HashMap<usize, usize> = HashMap::new();
        for node in &self.nodes {
            for a in &node.active {
                *active_per_job
                    .entry(self.scenario.requests[&(a.id)].job)
                    .or_default() += 1;
            }
        }
        let mut groups: BTreeMap<(u32, usize), VecDeque<usize>> = BTreeMap::new();
        for (&id, state) in &self.states {
            if state.state != State::Queued {
                continue;
            }
            let r = &self.scenario.requests[&id];
            if !r.routing_allowed {
                continue;
            }
            if !r.depends_on.iter().all(|dependency| {
                self.states[dependency]
                    .completed_s
                    .is_some_and(|t| t <= self.now)
            }) {
                continue;
            }
            let priority = if self.policy.plans_ahead()
                && self.now - r.arrival_s < self.scenario.scheduling.aging_s
            {
                self.scenario.scheduling.classes[&r.class].priority
            } else {
                0
            };
            groups.entry((priority, r.job)).or_default().push_back(id);
        }
        loop {
            let selection = groups
                .iter()
                .filter_map(|(key, queue)| {
                    let id = *queue.front()?;
                    let node = self.node_for(id)?;
                    Some((*key, id, node))
                })
                .min_by(|(a, ia, _), (b, ib, _)| {
                    let fair = |job| {
                        if self.policy.plans_ahead() {
                            *active_per_job.get(&job).unwrap_or(&0)
                        } else {
                            0
                        }
                    };
                    a.0.cmp(&b.0)
                        .then(fair(a.1).cmp(&fair(b.1)))
                        .then_with(|| {
                            self.scenario.requests[ia]
                                .arrival_s
                                .total_cmp(&self.scenario.requests[ib].arrival_s)
                        })
                        .then(ia.cmp(ib))
                });
            let Some((key, id, node_index)) = selection else {
                break;
            };
            groups.get_mut(&key).expect("selected group").pop_front();
            let request = &self.scenario.requests[&(id)];
            let node = &mut self.nodes[node_index];
            let p = &self.profiles[node.profile];
            let upload_s = request.upload_bytes as f64 / p.network_bytes_per_s;
            let adapter_ready = if let Some(adapter) = &request.adapter {
                if let Some(pos) = node.adapters.iter().position(|(a, _)| a == adapter) {
                    let entry = node.adapters.remove(pos).expect("existing adapter");
                    let ready = entry.1;
                    node.adapters.push_back(entry);
                    ready
                } else {
                    if node.adapters.len() == p.adapter_cache_slots {
                        let pos = node
                            .adapters
                            .iter()
                            .position(|(a, _)| {
                                !node.active.iter().any(|active| {
                                    self.scenario.requests[&(active.id)].adapter.as_ref() == Some(a)
                                })
                            })
                            .expect("admission checked eviction");
                        node.adapters.remove(pos);
                    }
                    let ready = self.now + p.network_s + upload_s + p.adapter_load_s;
                    node.adapters.push_back((adapter.clone(), ready));
                    ready
                }
            } else {
                self.now
            };
            node.active.push(Active {
                id,
                prefill: p.prefill_work(request),
                generated: 0.0,
                available_s: (self.now + p.network_s + upload_s).max(adapter_ready),
            });
            node.last_used_s = self.now;
            self.states.get_mut(&id).expect("known request").state = State::Active;
            self.actions.push(Action::Route {
                request: id,
                node: node_index,
                service_weight: if self.policy.plans_ahead() && node.warm {
                    self.scenario.scheduling.classes[&request.class].weight
                } else {
                    1.0
                },
            });
            *active_per_job.entry(request.job).or_default() += 1;
        }
    }
}
