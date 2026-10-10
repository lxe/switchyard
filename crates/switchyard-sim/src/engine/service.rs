//! Shared GPU work and client token delivery between events.
use super::*;

impl Simulation {
    pub(super) fn advance(&mut self, until: f64) {
        for n in &mut self.nodes {
            if n.ended_s.is_some() || !n.initialized || n.active.is_empty() {
                continue;
            }
            let p = &self.profiles[n.profile];
            let mut cursor = self.now;
            while cursor < until - 1e-10 && !n.active.is_empty() {
                let ready: Vec<_> = n
                    .active
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.available_s <= cursor + 1e-10)
                    .map(|(i, _)| i)
                    .collect();
                let next_available = n
                    .active
                    .iter()
                    .filter(|a| a.available_s > cursor + 1e-10)
                    .map(|a| a.available_s)
                    .min_by(f64::total_cmp)
                    .unwrap_or(until)
                    .min(until);
                if ready.is_empty() {
                    cursor = next_available;
                    continue;
                }
                let adapters: std::collections::HashSet<_> = ready
                    .iter()
                    .filter_map(|i| self.scenario.requests[n.active[*i].id].adapter.as_deref())
                    .collect();
                let capacity = p.capacity(ready.len())
                    * self.scenario.capacity_scale
                    * if adapters.len() > 1 {
                        p.multi_adapter_factor
                    } else {
                        1.0
                    };
                let sum_weight: f64 = ready.iter().map(|i| n.active[*i].weight).sum();
                let finish_dt = ready
                    .iter()
                    .map(|i| {
                        let a = &n.active[*i];
                        (a.prefill + a.output) / (capacity * a.weight / sum_weight)
                    })
                    .min_by(f64::total_cmp)
                    .expect("ready work");
                let dt = finish_dt
                    .min(next_available - cursor)
                    .min(until - cursor)
                    .max(1e-12);
                n.busy_s += dt;
                for i in ready {
                    let a = &mut n.active[i];
                    let rate = capacity * a.weight / sum_weight;
                    let before_input = a.prefill;
                    let input_used = (rate * dt).min(a.prefill);
                    a.prefill -= input_used;
                    let output_used = (rate * dt - input_used).min(a.output);
                    a.output -= output_used;
                    a.generated += output_used;
                    n.emitted += output_used;
                    let state = &mut self.states[a.id];
                    if output_used > 0.0 {
                        let start_s =
                            (cursor + before_input / rate + p.network_s).max(state.egress_until_s);
                        let duration =
                            (output_used / rate).max(output_used * 4.0 / p.network_bytes_per_s);
                        let end_s = start_s + duration;
                        state.egress_until_s = end_s;
                        if a.generated >= 1.0 && state.result.first_token_s.is_none() {
                            state.result.first_token_s = Some(
                                start_s
                                    + (1.0 - (a.generated - output_used)) * duration / output_used,
                            );
                        }
                        if self.record_delivery {
                            state.delivery.push(OutputSegment {
                                start_s,
                                end_s,
                                before: a.generated - output_used,
                                after: a.generated,
                            });
                        }
                    }
                    if a.prefill <= 1e-8 && a.output <= 1e-8 {
                        state.result.completed_s =
                            Some((cursor + dt + p.network_s).max(state.egress_until_s));
                        state.state = State::Done;
                    }
                }
                cursor += dt;
                n.last_used_s = cursor;
                n.active.retain(|a| self.states[a.id].state != State::Done);
            }
        }
    }
}
