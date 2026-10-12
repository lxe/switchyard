//! Scaling and routing only observe input, plans, and generated token counts.
//! Actual future output remains confined to the engine's service model.
use super::*;
use std::collections::BTreeMap;

impl SchedulerWorld<'_> {
    fn output_estimate(&self, id: usize) -> u64 {
        self.output_estimates
            .get(&id)
            .copied()
            .unwrap_or(self.scenario.requests[&id].planned_output)
    }
    fn estimated_remaining(&self, id: usize, profile: &Profile) -> f64 {
        let request = &self.scenario.requests[&(id)];
        self.progress.get(&id).map_or_else(
            || profile.prefill_work(request) + self.output_estimate(id) as f64,
            |(remaining_fraction, generated)| {
                profile.prefill_work(request) * remaining_fraction
                    + (self.output_estimate(id) as f64 - generated)
                        .max(self.scenario.scheduling.replan_output_floor)
            },
        )
    }

    pub(super) fn predicted_start(&self, profile: &Profile) -> f64 {
        if self.policy == Policy::Adaptive
            && let Some(index) = self.profiles.iter().position(|p| p.id == profile.id)
            && let Some(start) = self.startup_observed.get(&index)
        {
            return *start;
        }
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
        let measured = self
            .profiles
            .iter()
            .position(|p| p.id == profile.id)
            .and_then(|i| self.startup_extra.get(&i))
            .copied();
        let mean_bandwidth = profile
            .host
            .volume_bandwidth_bytes_per_s
            .iter()
            .sum::<f64>()
            / profile.host.volume_bandwidth_bytes_per_s.len() as f64;
        let expected_bytes = profile.host.container_image_bytes as f64
            * (1.0 - profile.host.image_cache_probability)
            + profile.host.model_bytes as f64 * (1.0 - profile.host.model_cache_probability);
        restore * (1.0 - self.scenario.build_probability)
            + build * self.scenario.build_probability
            + measured.unwrap_or(expected_bytes / mean_bandwidth)
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
            if matches!(state, State::Done | State::Rejected | State::Delivering) {
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
                        && self.scenario.topology.plan_for(&p.id).is_none_or(|plan| {
                            let mut additions = BTreeMap::<&str, usize>::new();
                            for endpoint in &plan.endpoints {
                                let ep = self
                                    .profiles
                                    .iter()
                                    .find(|p| p.id == endpoint.profile)
                                    .expect("validated topology");
                                *additions.entry(&ep.model).or_default() += 1;
                            }
                            additions.iter().all(|(model, count)| {
                                self.nodes
                                    .iter()
                                    .filter(|n| {
                                        n.ended_s.is_none()
                                            && self.profiles[n.profile].model == **model
                                    })
                                    .count()
                                    + count
                                    <= self.scenario.max_nodes_per_model
                            })
                        })
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
                    let output = self.job_outputs[&job.id];
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
            let calculate_capacity = |profile: &Profile| {
                let mean_reservation = ids
                    .iter()
                    .map(|id| {
                        let r = &self.scenario.requests[id];
                        (r.input_tokens + r.max_output) as f64
                    })
                    .sum::<f64>()
                    / ids.len() as f64;
                let memory_slots = (profile.kv_tokens as f64 / mean_reservation.max(1.0))
                    .floor()
                    .max(1.0) as usize;
                let has_adapters = ids
                    .iter()
                    .filter_map(|id| self.scenario.requests[id].adapter.as_deref())
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    > 1;
                let profile_index = self
                    .profiles
                    .iter()
                    .position(|p| p.id == profile.id)
                    .expect("profile index");
                let factor = if self.policy == Policy::Adaptive {
                    ids.iter()
                        .map(|id| {
                            self.learning.capacity_factor(
                                profile_index,
                                &self.scenario.requests[id],
                                &self.scenario.scheduling.heuristics,
                            )
                        })
                        .fold(f64::INFINITY, f64::min)
                } else {
                    1.0
                };
                factor
                    * self.placement_fraction(profile)
                    * profile.capacity(ids.len().min(profile.max_in_flight).min(memory_slots))
                    * work
                    / work_for(profile)
                    * if has_adapters {
                        profile.multi_adapter_factor
                    } else {
                        1.0
                    }
            };
            // Pool statistics and bucket quantiles are invariant during this decision.
            // Compute once rather than recomputing per drain-search iteration.
            let capacities: HashMap<_, _> = self
                .profiles
                .iter()
                .filter(|p| p.model == model && p.workload == workload)
                .map(|p| (p.id.as_str(), calculate_capacity(p)))
                .collect();
            let capacity_for = |profile: &Profile| capacities[profile.id.as_str()];
            let node_capacity = |node: &ObservedNode| {
                let profile = &self.profiles[node.profile];
                capacity_for(profile)
                    * node.placement.as_ref().map_or(1.0, |p| {
                        p.capacity_fraction / self.placement_fraction(profile)
                    })
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
                    node_capacity(n)
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
                .min(ids.len())
                .saturating_sub(total_model_nodes);
            // Forecast when existing compatible capacity drains this known work.
            let existing_drain = if live.is_empty() {
                f64::INFINITY
            } else {
                let mut low = first_arrival;
                let mut high = first_arrival
                    + work / live.iter().map(|n| node_capacity(n)).sum::<f64>().max(1e-8)
                    + live
                        .iter()
                        .map(|n| (n.ready_s - first_arrival).max(0.0))
                        .fold(0.0, f64::max);
                for _ in 0..40 {
                    let mid = (low + high) * 0.5;
                    let served: f64 = live
                        .iter()
                        .map(|n| node_capacity(n) * (mid - first_arrival.max(n.ready_s)).max(0.0))
                        .sum();
                    if served >= work {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }
                high
            };
            let mut chosen = vec![];
            for _ in 0..available {
                if capacity >= work {
                    break;
                }
                let Some(best) = candidates
                    .iter()
                    .copied()
                    .filter(|i| {
                        let mut additions = BTreeMap::<&str, usize>::new();
                        for index in chosen.iter().copied().chain(std::iter::once(*i)) {
                            let profile = &self.profiles[index];
                            if let Some(plan) = self.scenario.topology.plan_for(&profile.id) {
                                for endpoint in &plan.endpoints {
                                    let ep = self
                                        .profiles
                                        .iter()
                                        .find(|p| p.id == endpoint.profile)
                                        .expect("validated topology");
                                    *additions.entry(&ep.model).or_default() += 1;
                                }
                            } else {
                                *additions.entry(&profile.model).or_default() += 1;
                            }
                        }
                        if additions.iter().any(|(model, count)| {
                            self.nodes
                                .iter()
                                .filter(|n| {
                                    n.ended_s.is_none() && self.profiles[n.profile].model == **model
                                })
                                .count()
                                + count
                                > self.scenario.max_nodes_per_model
                        }) {
                            return false;
                        }
                        let ready =
                            first_arrival.max(self.now + self.predicted_start(&self.profiles[*i]));
                        if live.is_empty() && chosen.is_empty() {
                            return true;
                        }
                        let horizon = if existing_drain.is_finite() {
                            existing_drain
                        } else {
                            end
                        };
                        let existing_at_ready: f64 = live
                            .iter()
                            .map(|n| {
                                node_capacity(n) * (ready - first_arrival.max(n.ready_s)).max(0.0)
                            })
                            .sum();
                        let remaining = (work - existing_at_ready).max(0.0);
                        let total_rate: f64 = live.iter().map(|n| node_capacity(n)).sum();
                        let gain = if total_rate > 0.0 {
                            remaining / total_rate
                                - remaining / (total_rate + capacity_for(&self.profiles[*i]))
                        } else {
                            horizon - ready
                        };
                        ready < horizon
                            && gain >= self.scenario.scheduling.heuristics.minimum_scale_gain_s
                    })
                    .min_by(|a, b| {
                        let score = |i: usize| {
                            let p = &self.profiles[i];
                            let startup = self.predicted_start(p);
                            let contribution_end = if existing_drain.is_finite() {
                                end.max(existing_drain)
                            } else {
                                end
                            };
                            let useful = (contribution_end - first_arrival.max(self.now + startup))
                                .max(0.0)
                                * capacity_for(p);
                            let cost = p.usd_per_hour
                                * (window + startup + self.scenario.idle_timeout_s)
                                    .max(p.minimum_bill_s)
                                / 3600.0;
                            let value = if self.policy == Policy::CheapestElastic {
                                p.usd_per_hour / capacity_for(p).max(1e-8)
                            } else if useful > 0.0 {
                                cost / useful
                            } else {
                                1e6 + startup / capacity_for(p).max(1e-8)
                            };
                            value * self.placement_penalties.get(&i).copied().unwrap_or(1.0)
                        };
                        score(*a)
                            .total_cmp(&score(*b))
                            .then_with(|| self.profiles[*a].id.cmp(&self.profiles[*b].id))
                    })
                else {
                    break;
                };
                let p = &self.profiles[best];
                let startup = self.predicted_start(p);
                // Announced jobs buy capacity close to arrival, avoiding idle churn.
                if self.now + startup + self.scenario.scheduling.plan_interval_s < first_arrival {
                    break;
                }
                let contribution_end = if self.now + startup >= end && existing_drain.is_finite() {
                    existing_drain
                } else {
                    end
                };
                capacity += capacity_for(p)
                    * (contribution_end - first_arrival.max(self.now + startup)).max(0.0);
                chosen.push(best);
                if capacity <= 1e-8 {
                    break;
                }
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
                    (work + p.prefill_work(request) + self.output_estimate(id) as f64)
                        / (p.capacity(node.active.len() + 1)
                            * node.placement.as_ref().map_or(1.0, |p| p.capacity_fraction)
                            * if self.policy == Policy::Adaptive {
                                self.learning.capacity_factor(
                                    node.profile,
                                    request,
                                    &self.scenario.scheduling.heuristics,
                                )
                            } else {
                                1.0
                            })
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
        let mut groups: BTreeMap<(u32, usize, String, String, Workload), VecDeque<usize>> =
            BTreeMap::new();
        for (&id, state) in &self.states {
            if state.state != State::Queued {
                continue;
            }
            let r = &self.scenario.requests[&id];
            if !r.routing_allowed {
                continue;
            }
            if !r.depends_on.iter().all(|dependency| {
                self.states
                    .get(dependency)
                    .and_then(|state| state.completed_s)
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
            groups
                .entry((
                    priority,
                    r.job,
                    r.class.clone(),
                    r.model.clone(),
                    r.workload,
                ))
                .or_default()
                .push_back(id);
        }
        loop {
            let mut available_pools = std::collections::BTreeSet::new();
            for node in &self.nodes {
                if !node.is_ready || node.ended_s.is_some() {
                    continue;
                }
                let profile = &self.profiles[node.profile];
                let used: u64 = node
                    .active
                    .iter()
                    .map(|a| {
                        let r = &self.scenario.requests[&a.id];
                        r.input_tokens + r.max_output
                    })
                    .sum();
                if used >= profile.kv_tokens {
                    continue;
                }
                for (name, class) in &self.scenario.scheduling.classes {
                    let reserve = if self.policy.plans_ahead()
                        && node.warm
                        && !class.can_use_reserved_warm_slots
                    {
                        self.scenario
                            .scheduling
                            .reserve_warm_slots
                            .min(profile.max_in_flight.saturating_sub(1))
                    } else {
                        0
                    };
                    if node.active.len() < profile.max_in_flight - reserve {
                        available_pools.insert((
                            profile.model.clone(),
                            profile.workload,
                            name.clone(),
                        ));
                    }
                }
            }
            if available_pools.is_empty() {
                break;
            }
            let selection = groups
                .iter()
                .filter_map(|(key, queue)| {
                    if !available_pools.contains(&(key.3.clone(), key.4, key.2.clone())) {
                        return None;
                    }
                    let candidates = if self.scenario.scheduling.allow_queue_overtaking {
                        queue.len()
                    } else {
                        1
                    };
                    let (id, node) = queue
                        .iter()
                        .take(candidates)
                        .find_map(|id| self.node_for(*id).map(|node| (*id, node)))?;
                    Some((key.clone(), id, node))
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
            let queue = groups.get_mut(&key).expect("selected group");
            let position = queue
                .iter()
                .position(|entry| *entry == id)
                .expect("selected request");
            queue.remove(position);
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
            self.progress.insert(
                id,
                (
                    if p.prefill_work(request) > 0.0 {
                        1.0
                    } else {
                        0.0
                    },
                    0.0,
                ),
            );
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
