//! Bounded passive state reads run independently of request dispatch.
use super::{
    LiveRuntime,
    lifecycle::{received_counters, update_observation},
    telemetry,
};
use crate::{Error, Result};
use switchyard_provider::Provider;

pub(super) struct StateObservation {
    start_s: f64,
    finish_s: f64,
    response: Result<switchyard_agent::State>,
}

impl<P: Provider + 'static> LiveRuntime<P> {
    /// Keep one outstanding read per live node. A slow HTTP response cannot
    /// prevent completion processing or dispatch to already-ready endpoints.
    pub(super) async fn observe(&mut self) -> Result<()> {
        self.finish_observations()?;
        for id in 0..self.nodes.len() {
            self.schedule_observation(id);
        }
        Ok(())
    }

    pub(super) fn schedule_observation(&mut self, id: usize) {
        let Some(node) = self.nodes.get(id) else {
            return;
        };
        if node.owned.ended_s.is_some() || self.state_reads.values().any(|(n, _)| *n == id) {
            return;
        }
        let endpoint = node.endpoint.clone();
        let epoch = self.epoch;
        let queued_s = epoch.elapsed().as_secs_f64();
        let task = self.state_tasks.spawn(async move {
            let start_s = epoch.elapsed().as_secs_f64();
            let response = endpoint.state().await;
            StateObservation {
                start_s,
                finish_s: epoch.elapsed().as_secs_f64(),
                response,
            }
        });
        self.state_reads.insert(task.id(), (id, queued_s));
    }

    pub(super) fn finish_observations(&mut self) -> Result<()> {
        while let Some(result) = self.state_tasks.try_join_next_with_id() {
            let (task, observation) = match result {
                Ok((task, observation)) => (task, observation),
                Err(error) => {
                    let Some((_, start_s)) = self.state_reads.get(&error.id()) else {
                        continue;
                    };
                    (
                        error.id(),
                        StateObservation {
                            start_s: *start_s,
                            finish_s: self.now(),
                            response: Err(Error::Request("agent observation task failed".into())),
                        },
                    )
                }
            };
            let Some((id, _)) = self.state_reads.remove(&task) else {
                continue;
            };
            let node = &mut self.nodes[id];
            if node.owned.ended_s.is_some() {
                continue;
            }
            let StateObservation {
                start_s,
                finish_s,
                response,
            } = observation;
            let explicit_probe = node.observed.host.probe_pending;
            node.observed.host.probe_pending = false;
            match response {
                Ok(state) => {
                    if node.bandwidth.record(
                        finish_s,
                        received_counters(&state),
                        &self.scheduling.probe,
                    ) {
                        node.observed.host.probe = None;
                    }
                    if let Some(sample) = node.bandwidth.qualify(&self.scheduling.probe) {
                        node.observed.host.probe = Some(switchyard_core::scheduler::HostProbe {
                            at_s: sample.at_s,
                            started_s: sample.started_s,
                            received_bytes: sample.received_bytes,
                            volume_bytes_per_s: sample.bytes_per_s,
                            image_cached: state.image_cached,
                            model_cached: state.model_cached,
                            hydration_bytes: state
                                .bytes_total
                                .saturating_add(state.artifact_bytes_total),
                        });
                        if explicit_probe {
                            self.probes += 1;
                        }
                    }
                    if explicit_probe {
                        telemetry::append(
                            &self.out,
                            "host-probes.ndjson",
                            &serde_json::json!({
                                "at_s":finish_s, "start_s":start_s, "node":id, "qualified_window":node.observed.host.probe
                            }),
                        )?;
                    }
                    let observation = serde_json::json!({"at_s":finish_s,"start_s":start_s,"finish_s":finish_s,"pod":node.owned.pod.id,"state":state});
                    telemetry::append(&self.out, "agent-observations.ndjson", &observation)?;
                    self.observations.push(observation);
                    if update_observation(&mut node.observed, &state, finish_s)? {
                        println!(
                            "{} ready after {:.1}s",
                            self.profiles[node.observed.profile].model,
                            finish_s - node.owned.allocated_s
                        );
                    }
                }
                Err(_) => {
                    telemetry::append(
                        &self.out,
                        "agent-observations.ndjson",
                        &serde_json::json!({
                            "at_s":finish_s,"start_s":start_s,"finish_s":finish_s,"pod":node.owned.pod.id,"status":"fetch_failed",
                        }),
                    )?;
                    if !node.observed.is_ready
                        && finish_s - node.owned.allocated_s
                            >= self.config.limits.startup_timeout_s as f64
                    {
                        return Err(Error::Limit("agent failed to become reachable".into()));
                    }
                }
            }
            if !node.observed.is_ready {
                let p = &self.profiles[node.observed.profile];
                let remaining_s = node
                    .observed
                    .host
                    .probe
                    .as_ref()
                    .filter(|s| {
                        s.fresh(finish_s, &self.scheduling.probe)
                            && node.observed.host.remaining_bytes_known
                    })
                    .map_or(0.0, |s| {
                        node.observed
                            .host
                            .bytes_total
                            .saturating_sub(node.observed.host.bytes_downloaded)
                            as f64
                            / s.volume_bytes_per_s
                    });
                node.observed.ready_s = finish_s + remaining_s + p.startup.engine_s;
            }
        }
        Ok(())
    }

    pub(super) async fn cancel_observations(&mut self) {
        self.state_tasks.abort_all();
        while self.state_tasks.join_next().await.is_some() {}
        self.state_reads.clear();
    }
}
