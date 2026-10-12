//! Bounded learning from delivered requests and public allocation outcomes.
use super::*;
use serde::Serialize;

const MAX_KEYS: usize = 1024;
type OutputKey = (String, Workload, String, Option<String>, u32, u32, bool);
fn key(r: &PlannedRequest) -> OutputKey {
    (
        r.model.clone(),
        r.workload,
        r.class.clone(),
        r.adapter.clone(),
        r.input_tokens.max(1).ilog2(),
        r.max_output.max(1).ilog2(),
        r.cached_fraction > 0.0,
    )
}
#[derive(Default, Serialize)]
pub(super) struct Learning {
    #[serde(skip)]
    outputs: BTreeMap<OutputKey, VecDeque<u64>>,
    #[serde(skip)]
    output_quantiles: BTreeMap<OutputKey, f64>,
    allocations: BTreeMap<usize, VecDeque<(bool, f64)>>,
    #[serde(skip)]
    capacity: BTreeMap<(usize, OutputKey), VecDeque<f64>>,
    #[serde(skip)]
    capacity_quantiles: BTreeMap<(usize, OutputKey), f64>,
    startups: BTreeMap<usize, VecDeque<f64>>,
    #[serde(skip)]
    ready_nodes: std::collections::BTreeSet<usize>,
    #[serde(skip)]
    windows: BTreeMap<usize, Window>,
    pub completed_samples: usize,
    pub service_windows: usize,
}
#[derive(Default)]
struct Window {
    profile: usize,
    scope: Option<OutputKey>,
    start: f64,
    end: f64,
    service_seconds: f64,
    work: f64,
    samples: usize,
    invalid: bool,
}
fn bounded_push<T>(values: &mut VecDeque<T>, value: T, limit: usize) {
    values.push_back(value);
    while values.len() > limit {
        values.pop_front();
    }
}
impl Learning {
    pub fn observe(&mut self, event: SchedulerFeedback, limit: usize) {
        match event {
            SchedulerFeedback::Allocation {
                profile,
                accepted,
                elapsed_s,
            } => {
                if elapsed_s.is_finite() && elapsed_s >= 0.0 {
                    bounded_push(
                        self.allocations.entry(profile).or_default(),
                        (accepted, elapsed_s),
                        limit,
                    );
                }
            }
            SchedulerFeedback::Completion {
                node,
                profile,
                request,
                output_tokens,
                input_work,
                started_s,
                completed_s,
            } => {
                if !started_s.is_finite()
                    || !completed_s.is_finite()
                    || completed_s <= started_s
                    || output_tokens > request.max_output
                    || !input_work.is_finite()
                    || input_work < 0.0
                {
                    return;
                }
                if self.outputs.len() >= MAX_KEYS
                    && !self.outputs.contains_key(&key(&request))
                    && let Some((old, _)) = self.outputs.pop_first()
                {
                    self.output_quantiles.remove(&old);
                }
                bounded_push(
                    self.outputs.entry(key(&request)).or_default(),
                    output_tokens,
                    limit,
                );
                let scope = key(&request);
                let values = &self.outputs[&scope];
                self.output_quantiles.insert(
                    scope,
                    percentile(values.iter().map(|n| *n as f64).collect(), 0.9).unwrap_or(0.0),
                );
                // Profile-specific work coefficients are applied when flushing the window.
                let window = self.windows.entry(node).or_insert_with(|| Window {
                    profile,
                    scope: Some(key(&request)),
                    start: started_s,
                    end: completed_s,
                    ..Window::default()
                });
                window.start = window.start.min(started_s);
                window.end = window.end.max(completed_s);
                window.service_seconds += completed_s - started_s;
                window.invalid |= window.scope.as_ref() != Some(&key(&request));
                window.work += output_tokens as f64 + input_work;
                window.samples += 1;
                self.completed_samples += 1;
            }
            SchedulerFeedback::FailedRequest { node } => {
                self.windows.entry(node).or_default().invalid = true;
            }
        }
    }
    pub fn update(&mut self, fleet: &FleetView<'_>) {
        let h = &fleet.scheduling.heuristics;
        for (id, node) in fleet.nodes.iter().enumerate() {
            if node.ended_s.is_some() {
                self.ready_nodes.remove(&id);
            }
            if node.ended_s.is_none() && node.is_ready && !node.warm && self.ready_nodes.insert(id)
            {
                bounded_push(
                    self.startups.entry(node.profile).or_default(),
                    (node.ready_s - node.allocated_s).max(0.0),
                    h.history_limit,
                );
            }
            if node.active.is_empty()
                && let Some(w) = self.windows.remove(&id)
                && !w.invalid
                && w.samples >= h.minimum_samples
                && w.end > w.start
            {
                let elapsed = w.end - w.start;
                let p = &fleet.profiles[w.profile];
                let concurrency = (w.service_seconds / elapsed).round().max(1.0) as usize;
                // Client elapsed time includes queue/network delays, giving a conservative service estimate.
                let factor = (w.work / elapsed / p.capacity(concurrency.min(p.max_in_flight)))
                    .clamp(0.1, h.maximum_capacity_multiplier);
                let scope = (w.profile, w.scope.expect("qualified homogeneous window"));
                if self.capacity.len() >= MAX_KEYS
                    && !self.capacity.contains_key(&scope)
                    && let Some((old, _)) = self.capacity.pop_first()
                {
                    self.capacity_quantiles.remove(&old);
                }
                bounded_push(
                    self.capacity.entry(scope.clone()).or_default(),
                    factor,
                    h.history_limit,
                );
                self.capacity_quantiles.insert(
                    scope.clone(),
                    percentile(self.capacity[&scope].iter().copied().collect(), 0.25)
                        .unwrap_or(1.0),
                );
                self.service_windows += 1;
            }
        }
    }
    pub fn estimate(&self, r: &PlannedRequest, h: &Heuristics) -> u64 {
        self.outputs
            .get(&key(r))
            .filter(|s| s.len() >= h.minimum_samples)
            .and_then(|_| self.output_quantiles.get(&key(r)).copied())
            .map_or(r.planned_output, |n| {
                (n.ceil() as u64).max(1).min(r.max_output)
            })
    }
    pub fn capacity_factor(&self, profile: usize, request: &PlannedRequest, h: &Heuristics) -> f64 {
        self.capacity_quantiles
            .get(&(profile, key(request)))
            .copied()
            .map_or(1.0, |n| {
                (n * h.capacity_safety_fraction).clamp(0.1, h.maximum_capacity_multiplier)
            })
    }
    pub fn startup(&self, profile: usize) -> Option<f64> {
        self.startups
            .get(&profile)
            .and_then(|s| percentile(s.iter().copied().collect(), 0.9))
    }
    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({"completed_samples":self.completed_samples,"service_windows":self.service_windows,
            "output_buckets":self.outputs.iter().map(|(key,samples)| serde_json::json!({"scope":key,"samples":samples})).collect::<Vec<_>>(),
            "capacity_buckets":self.capacity.iter().map(|(key,samples)| serde_json::json!({"scope":key,"samples":samples})).collect::<Vec<_>>(),
            "allocations":self.allocations,"startups":self.startups})
    }
    pub fn placement_penalty(&self, profile: usize) -> f64 {
        self.allocations.get(&profile).map_or(1.0, |samples| {
            let accepted = samples.iter().filter(|(ok, _)| *ok).count() as f64;
            let probability = (accepted + 1.0) / (samples.len() as f64 + 2.0);
            let delay = samples.iter().map(|(_, s)| s).sum::<f64>() / samples.len() as f64;
            (0.5 / probability).max(1.0) * (1.0 + delay / 30.0)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> PlannedRequest {
        PlannedRequest {
            routing_allowed: true,
            id: 0,
            job: 0,
            model: "model".into(),
            workload: Workload::Text,
            class: "standard".into(),
            arrival_s: 0.0,
            input_tokens: 32,
            planned_output: 96,
            max_output: 128,
            image_megapixels: 0.0,
            upload_bytes: 0,
            cached_fraction: 0.0,
            adapter: None,
            latency_target_s: 10.0,
            depends_on: vec![],
        }
    }
    #[test]
    fn output_estimates_require_samples_and_never_cross_workload_buckets() {
        let mut learning = Learning::default();
        let h = Heuristics::default();
        let r = request();
        for i in 0..h.minimum_samples {
            assert_eq!(learning.estimate(&r, &h), 96);
            learning.observe(
                SchedulerFeedback::Completion {
                    node: 0,
                    profile: 0,
                    request: Box::new(r.clone()),
                    output_tokens: 8,
                    input_work: 0.0,
                    started_s: i as f64,
                    completed_s: i as f64 + 1.0,
                },
                h.history_limit,
            );
        }
        assert_eq!(learning.estimate(&r, &h), 8);
        let mut other = r.clone();
        other.input_tokens = 2048;
        assert_eq!(learning.estimate(&other, &h), 96);
        other = r.clone();
        other.adapter = Some("different-adapter".into());
        assert_eq!(learning.estimate(&other, &h), 96);
        other = r.clone();
        other.cached_fraction = 1.0;
        assert_eq!(learning.estimate(&other, &h), 96);
        for i in 0..2000 {
            other.model = format!("model-{i}");
            learning.observe(
                SchedulerFeedback::Completion {
                    node: 0,
                    profile: 0,
                    request: Box::new(other.clone()),
                    output_tokens: 8,
                    input_work: 0.0,
                    started_s: 0.0,
                    completed_s: 1.0,
                },
                h.history_limit,
            );
        }
        assert_eq!(learning.outputs.len(), MAX_KEYS);
        assert!(
            learning
                .outputs
                .values()
                .all(|v| v.len() <= h.history_limit)
        );
    }
}
