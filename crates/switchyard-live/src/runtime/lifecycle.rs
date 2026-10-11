//! Ownership-safe provisioning, observations, and resource cleanup.
use super::{AllocationRejection, LiveRuntime, Node, OwnedPod, PendingLaunch, nonce, rental_rate};
use crate::{Error, Result, serving::Endpoint};
use std::{collections::VecDeque, time::Duration};
use switchyard_agent::{Phase, Ticket};
use switchyard_core::{
    probe::{BandwidthWindow, TransferCounter},
    scheduler::{HostObservation, HostProbe, ObservedNode, StartupPhase},
};
use switchyard_provider::{Launch, Provider};

impl<P: Provider> LiveRuntime<P> {
    pub(super) async fn provision(&mut self, profile: usize) -> Result<bool> {
        if self
            .provisioning_cooldowns
            .get(&profile)
            .is_some_and(|until| *until > self.now())
        {
            return Ok(false);
        }
        if self
            .nodes
            .iter()
            .filter(|n| n.owned.ended_s.is_none())
            .count()
            >= self.config.limits.max_instances
        {
            return Ok(false);
        }
        let Some(p) = self.profiles.get(profile) else {
            self.invalid += 1;
            return Ok(false);
        };
        let rate: f64 = self
            .nodes
            .iter()
            .filter(|n| n.owned.ended_s.is_none())
            .map(|n| rental_rate(&n.owned))
            .sum();
        if rate + p.usd_per_hour > self.config.limits.max_usd_per_hour {
            return Ok(false);
        }
        self.limits()?;
        let model = self
            .config
            .models
            .iter()
            .find(|m| m.id == p.model)
            .ok_or_else(|| Error::Config("profile model missing".into()))?;
        let offering = self.profile_offerings[profile].clone();
        let name = format!("switchyard-{}-{}", self.run_id, self.pending.len());
        let ticket_key = format!("{}/{name}.json", self.config.bootstrap.ticket_prefix);
        self.pending.push(PendingLaunch {
            name: name.clone(),
            ticket_key: ticket_key.clone(),
            resolved: false,
        });
        self.save("pending-launches.json", &self.pending)?;
        let manifest = self.manifests[&model.id].clone();
        let source = model.artifact.ticket(&self.storage, manifest)?;
        let ticket = Ticket {
            source,
            model: model.id.clone(),
            revision: model.revision.clone(),
            engine_args: model.engine.args(
                model,
                &self
                    .config
                    .bootstrap
                    .startup
                    .model_directory()
                    .to_string_lossy(),
            ),
            concurrency: self.config.limits.hydration_concurrency,
            stall_s: self.config.limits.hydration_stall_s,
            lifetime_s: self.config.limits.run_timeout_s,
            startup: self.config.bootstrap.startup.clone(),
            runtime: if let Some(runtime) = &self.config.bootstrap.runtime {
                let mut caches = vec![];
                for cache in model.caches.iter().filter(|c| c.gpu == offering.gpu) {
                    caches.push(switchyard_agent::runtime::CacheTicket {
                        name: cache.name.clone(),
                        artifact: cache.artifact.ticket(
                            &self.storage,
                            self.cache_manifests[&cache.artifact.id].clone(),
                        )?,
                        compatibility: cache.requirements(runtime, model)?,
                    });
                }
                Some(switchyard_agent::runtime::RuntimeTicket {
                    hydration: runtime.hydration,
                    artifact: runtime.artifact.ticket(
                        &self.storage,
                        self.runtime_manifest.clone().expect("prepared runtime"),
                    )?,
                    compatibility: runtime.compatibility.clone(),
                    program: runtime.program.clone(),
                    args: runtime.args.clone(),
                    environment: runtime.environment.clone(),
                    caches,
                    cache_export: self.cache_export.clone(),
                })
            } else {
                None
            },
        };
        self.storage.put_json(&ticket_key, &ticket).await?;
        let token = nonce()?;
        let allocated = self.now();
        let launched = self
            .provider
            .create(Launch {
                name: name.clone(),
                offering: offering.clone(),
                container: self.config.bootstrap.container(
                    self.config.disk_gb,
                    self.storage.presign(&ticket_key)?,
                    token.clone(),
                ),
            })
            .await;
        let pod = match launched {
            Ok(pod) => pod,
            Err(error) => {
                if !matches!(error, switchyard_provider::Error::UnknownLaunch { .. }) {
                    self.pending.last_mut().expect("journaled launch").resolved = true;
                    self.save("pending-launches.json", &self.pending)?;
                }
                if let switchyard_provider::Error::AllocationRejected { status } = error {
                    let at_s = self.now();
                    let retry_at_s = at_s + self.scheduling.retry_cooldown_s;
                    self.provisioning_cooldowns.insert(profile, retry_at_s);
                    self.allocation_rejections.push(AllocationRejection {
                        at_s,
                        profile,
                        name,
                        status,
                        retry_at_s,
                    });
                    self.save("allocation-rejections.json", &self.allocation_rejections)?;
                    return Ok(false);
                }
                return Err(error.into());
            }
        };
        // Persist provider ownership before any fallible endpoint setup.
        let owned = OwnedPod {
            pod: pod.clone(),
            allocated_s: allocated,
            ended_s: None,
            profile,
            quoted: offering,
        };
        let mut journal = self.pods();
        journal.push(owned.clone());
        self.save("owned-pods.json", &journal)?;
        let endpoint = Endpoint::new(
            self.provider.endpoint_url(&pod, 8080),
            &token,
            self.config.limits.request_timeout_s,
        )?;
        let observed = ObservedNode {
            profile,
            warm: false,
            allocated_s: allocated,
            ready_s: allocated + p.startup.total(),
            ended_s: None,
            active: vec![],
            adapters: VecDeque::new(),
            last_used_s: allocated,
            is_ready: false,
            host: HostObservation {
                phase: StartupPhase::Booting,
                bytes_downloaded: 0,
                bytes_total: p.host.model_bytes,
                probe_pending: false,
                probe_attempted_s: None,
                probe: None,
            },
        };
        self.nodes.push(Node {
            owned,
            endpoint,
            observed,
            ticket_key,
            checked_tokenizer: false,
            bandwidth: BandwidthWindow::default(),
        });
        self.pending.last_mut().expect("journaled launch").resolved = true;
        self.save("pending-launches.json", &self.pending)?;
        if let Err(error) = self.journal() {
            let _ = self.release(self.nodes.len() - 1).await;
            return Err(error);
        }
        self.launches += 1;
        let actual_rate: f64 = self
            .nodes
            .iter()
            .filter(|n| n.owned.ended_s.is_none())
            .map(|n| rental_rate(&n.owned))
            .sum();
        if actual_rate > self.config.limits.max_usd_per_hour {
            return Err(Error::Limit(
                "actual hourly pod quotes exceed the fleet budget".into(),
            ));
        }
        self.limits()?;
        Ok(true)
    }
    pub(super) async fn observe(&mut self) -> Result<()> {
        let now = self.now();
        for node in &mut self.nodes {
            if node.owned.ended_s.is_some() {
                continue;
            }
            let start_s = self.epoch.elapsed().as_secs_f64();
            let response = node.endpoint.state().await;
            let finish_s = self.epoch.elapsed().as_secs_f64();
            if response.is_err() {
                super::telemetry::append(
                    &self.out,
                    "agent-observations.ndjson",
                    &serde_json::json!({
                        "at_s":finish_s,"start_s":start_s,"finish_s":finish_s,"pod":node.owned.pod.id,"status":"fetch_failed",
                    }),
                )?;
            }
            match response {
                Ok(state) => {
                    let at_s = finish_s;
                    if node.bandwidth.record(
                        at_s,
                        received_counters(&state),
                        &self.scheduling.probe,
                    ) {
                        node.observed.host.probe = None;
                    }
                    let observation = serde_json::json!({"at_s":at_s,"start_s":start_s,"finish_s":finish_s,"pod":node.owned.pod.id,"state":state});
                    super::telemetry::append(&self.out, "agent-observations.ndjson", &observation)?;
                    self.observations.push(observation);
                    if update_observation(&mut node.observed, &state, at_s)? {
                        println!(
                            "{} ready after {:.1}s",
                            self.profiles[node.observed.profile].model,
                            at_s - node.owned.allocated_s
                        );
                    }
                }
                Err(_)
                    if now - node.owned.allocated_s
                        < self.config.limits.startup_timeout_s as f64 => {}
                Err(_) => return Err(Error::Limit("agent failed to become reachable".into())),
            }
            if !node.observed.is_ready {
                node.observed.ready_s = now + self.profiles[node.observed.profile].startup.engine_s;
            }
        }
        Ok(())
    }
    pub(super) async fn probe(&mut self, id: usize) -> Result<()> {
        let Some(node) = self.nodes.get_mut(id) else {
            self.invalid += 1;
            return Ok(());
        };
        let attempted_s = self.epoch.elapsed().as_secs_f64();
        if node.owned.ended_s.is_some()
            || !self
                .scheduling
                .probe
                .due(attempted_s, node.observed.host.probe_attempted_s)
        {
            return Ok(());
        }
        node.observed.host.probe_attempted_s = Some(attempted_s);
        let response = node.endpoint.state().await;
        let finish_s = self.epoch.elapsed().as_secs_f64();
        if let Ok(state) = response {
            let at_s = finish_s;
            let counters = received_counters(&state);
            let reset = node
                .bandwidth
                .record(at_s, counters.clone(), &self.scheduling.probe);
            if reset {
                node.observed.host.probe = None;
            }
            let sample = node.bandwidth.qualify(&self.scheduling.probe);
            let reported_rate =
                if state.hydration_mode == switchyard_artifacts::HydrationMode::TrustedStreaming {
                    state
                        .artifact_network_bytes_per_s
                        .or(state.volume_bytes_per_s)
                } else {
                    state
                        .volume_bytes_per_s
                        .or(state.artifact_volume_bytes_per_s)
                };
            super::telemetry::append(
                &self.out,
                "host-probes.ndjson",
                &serde_json::json!({"at_s": at_s, "start_s":attempted_s,"finish_s":finish_s,"node": id,
                    "state": state, "phase": state.phase, "rate_bytes_per_s": sample.as_ref().map(|s| s.bytes_per_s),
                    "reported_cumulative_rate_bytes_per_s": reported_rate,
                    "counters": counters, "qualified_window": sample, "counter_reset":reset,
                    "model_bytes_received": state.bytes_downloaded,
                    "artifact_bytes_received": state.artifact_network_bytes_downloaded,
                    "model_bytes_total": state.bytes_total, "artifact_bytes_total": state.artifact_bytes_total}),
            )?;
            if update_observation(&mut node.observed, &state, at_s)? {
                println!(
                    "{} ready after {:.1}s",
                    self.profiles[node.observed.profile].model,
                    at_s - node.owned.allocated_s
                );
            }
            let Some(sample) = sample else {
                return Ok(());
            };
            node.observed.host.probe = Some(HostProbe {
                at_s: sample.at_s,
                started_s: sample.started_s,
                received_bytes: sample.received_bytes,
                volume_bytes_per_s: sample.bytes_per_s,
                image_cached: state.image_cached,
                model_cached: state.model_cached,
                hydration_bytes: state.bytes_total.saturating_add(state.artifact_bytes_total),
            });
            self.probes += 1;
        } else {
            super::telemetry::append(
                &self.out,
                "host-probes.ndjson",
                &serde_json::json!({"at_s":finish_s,"start_s":attempted_s,"finish_s":finish_s,"node":id,"status":"fetch_failed"}),
            )?;
        }
        Ok(())
    }
    /// Confirm cloud termination before touching storage tickets or local receipts.
    async fn terminate_node(&mut self, id: usize) -> Result<()> {
        let Some(node) = self.nodes.get(id) else {
            self.invalid += 1;
            return Ok(());
        };
        if node.owned.ended_s.is_some() {
            return Ok(());
        }
        if let Some(pod) = self.provider.get(&node.owned.pod.id).await? {
            if pod.name != node.owned.pod.name {
                return Err(Error::Config(
                    "ownership changed; refusing pod termination".into(),
                ));
            }
            self.provider.terminate(&pod.id).await?;
            if self
                .provider
                .get(&pod.id)
                .await?
                .is_some_and(|p| !p.status.is_terminated())
            {
                return Err(Error::Request(
                    "pod termination is not yet confirmed".into(),
                ));
            }
        }
        let now = self.now();
        self.nodes[id].owned.ended_s = Some(now);
        self.nodes[id].observed.ended_s = Some(now);
        self.nodes[id].observed.host.phase = StartupPhase::Ended;
        self.releases += 1;
        Ok(())
    }

    pub(super) async fn release(&mut self, id: usize) -> Result<()> {
        if id >= self.nodes.len() {
            self.invalid += 1;
            return Ok(());
        }
        self.terminate_node(id).await?;
        self.storage.delete(&self.nodes[id].ticket_key).await?;
        self.journal()
    }

    /// Terminate every owned allocation even when a ticket or receipt cannot be written.
    pub async fn cleanup(&mut self) -> Result<()> {
        self.abort_requests().await;

        let mut failures = vec![];
        for id in 0..self.nodes.len() {
            for attempt in 0..3 {
                match self.terminate_node(id).await {
                    Ok(()) => break,
                    Err(error) if attempt == 2 => failures.push(error.to_string()),
                    Err(_) => tokio::time::sleep(Duration::from_secs(2)).await,
                }
            }
        }
        let confirmed = self
            .nodes
            .iter()
            .filter(|n| n.owned.ended_s.is_some())
            .map(|n| n.owned.pod.name.clone())
            .collect();
        if let Err(error) = crate::ownership::reconcile(
            &self.provider,
            &self.storage,
            &mut self.pending,
            &confirmed,
        )
        .await
        {
            failures.push(error.to_string());
        }
        // Reconciliation also deletes pending tickets. Keep deletion idempotent for nodes
        // restored from an ownership journal that predates a pending-launch record.
        for node in self.nodes.iter().filter(|n| n.owned.ended_s.is_some()) {
            if let Err(error) = self.storage.delete(&node.ticket_key).await {
                failures.push(error.to_string());
            }
        }
        // All termination attempts precede receipt I/O; one failed write cannot skip
        // another receipt or prevent a remaining paid allocation from being terminated.
        for result in [
            self.journal(),
            self.save("pending-launches.json", &self.pending),
            self.save("agent-observations.json", &self.observations),
        ] {
            if let Err(error) = result {
                failures.push(error.to_string());
            }
        }
        let receipt = serde_json::json!({
            "estimated_rental_usd": self.rental_usd(),
            "pods": self.pods(),
            "cleanup_failures": failures,
        });
        if let Err(error) = self.save("rental-receipt.json", &receipt) {
            failures.push(error.to_string());
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(Error::Request(format!(
                "cleanup operations incomplete: {}",
                failures.join("; ")
            )))
        }
    }
}

fn received_counters(state: &switchyard_agent::State) -> Vec<TransferCounter> {
    let mut counters = vec![TransferCounter {
        source: format!(
            "model:{:?}:{}:{}",
            state.hydration_mode, state.model, state.revision
        ),
        received_bytes: state.bytes_downloaded,
        total_bytes: state.bytes_total,
    }];
    counters.extend(state.artifacts.iter().map(|artifact| TransferCounter {
        source: format!("artifact:{}:{}", artifact.name, artifact.id),
        received_bytes: artifact.network_bytes_downloaded,
        total_bytes: artifact.total_bytes,
    }));
    // Older agents can expose aggregate artifact wire counters without entries.
    if state.artifacts.is_empty() && state.artifact_bytes_total > 0 {
        counters.push(TransferCounter {
            source: "artifacts:aggregate".into(),
            received_bytes: state.artifact_network_bytes_downloaded,
            total_bytes: state.artifact_bytes_total,
        });
    }
    counters
}

fn update_observation(
    node: &mut ObservedNode,
    state: &switchyard_agent::State,
    at_s: f64,
) -> Result<bool> {
    let model_received = if state.model_cached {
        state.bytes_total
    } else {
        state.bytes_downloaded
    };
    node.host.bytes_downloaded = model_received.saturating_add(
        if state.hydration_mode == switchyard_artifacts::HydrationMode::TrustedStreaming {
            state
                .artifact_network_bytes_downloaded
                .min(state.artifact_bytes_total)
                .max(
                    state
                        .artifact_bytes_downloaded
                        .min(state.artifact_bytes_total),
                )
        } else {
            state.artifact_bytes_downloaded
        },
    );
    node.host.bytes_total = state.bytes_total.saturating_add(state.artifact_bytes_total);
    node.host.phase = match state.phase {
        Phase::Booting => StartupPhase::Booting,
        Phase::Hydrating => StartupPhase::Hydrating,
        Phase::Initializing | Phase::Exporting => StartupPhase::Initializing,
        Phase::Ready => StartupPhase::Ready,
        Phase::Failed => {
            return Err(Error::Request(
                state.error.clone().unwrap_or_else(|| "agent failed".into()),
            ));
        }
    };
    if matches!(state.phase, Phase::Exporting) {
        node.is_ready = false;
    }
    let became_ready = matches!(state.phase, Phase::Ready) && !node.is_ready;
    if became_ready {
        node.is_ready = true;
        node.ready_s = at_s;
        node.last_used_s = at_s;
    }
    Ok(became_ready)
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_core::ProbePolicy;
    fn state() -> switchyard_agent::State {
        serde_json::from_value(serde_json::json!({"model":"fixture","revision":"r1","phase":"hydrating", "bytes_downloaded":10,"bytes_total":1000,"hydration_seconds":100.0,"volume_bytes_per_s":1e12,"model_cached":false,"image_cached":null,"error":null,
            "artifact_bytes_total":10000,"artifact_bytes_downloaded":1000,"artifact_network_bytes_downloaded":100,
            "artifacts":[{"name":"runtime","id":"runtime1","phase":"downloading", "downloaded_bytes":1000,"network_bytes_downloaded":100,"total_bytes":10000,"unpacked_bytes":0,"total_unpacked_bytes":30000,"verified_chunks":0,"total_chunks":1,"cached":false,"chunk_cache_hits":0,"error":null}]})).unwrap()
    }
    #[test]
    fn mixed_domains_measure_received_deltas_not_verified_or_cumulative_rates() {
        let p = ProbePolicy {
            minimum_received_bytes: 100,
            ..ProbePolicy::default()
        };
        for mode in [
            switchyard_artifacts::HydrationMode::Verified,
            switchyard_artifacts::HydrationMode::TrustedStreaming,
        ] {
            let mut state = state();
            state.hydration_mode = mode;
            let mut w = BandwidthWindow::default();
            w.record(0.0, received_counters(&state), &p);
            state.bytes_downloaded = 60;
            state.artifact_bytes_downloaded = 9000;
            state.artifacts[0].downloaded_bytes = 9000;
            state.artifacts[0].network_bytes_downloaded = 300;
            // An aggregate is not added a second time when per-artifact counters exist.
            state.artifact_network_bytes_downloaded = 300;
            w.record(2.0, received_counters(&state), &p);
            let sample = w.qualify(&p).unwrap();
            assert_eq!(sample.received_bytes, 250);
            assert_eq!(sample.bytes_per_s, 125.0);
        }
    }
    #[test]
    fn completed_probe_snapshot_updates_remaining_transfer_before_decision() {
        let mut state = state();
        state.hydration_mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
        state.bytes_downloaded = state.bytes_total;
        state.artifact_network_bytes_downloaded = state.artifact_bytes_total;
        state.phase = Phase::Initializing;
        let mut node = ObservedNode {
            profile: 0,
            warm: false,
            allocated_s: 0.0,
            ready_s: 100.0,
            ended_s: None,
            active: vec![],
            adapters: VecDeque::new(),
            last_used_s: 0.0,
            is_ready: false,
            host: HostObservation {
                phase: StartupPhase::Hydrating,
                bytes_downloaded: 0,
                bytes_total: 11000,
                probe_pending: false,
                probe_attempted_s: None,
                probe: None,
            },
        };
        update_observation(&mut node, &state, 2.0).unwrap();
        assert_eq!(node.host.bytes_downloaded, node.host.bytes_total);
        assert_eq!(node.host.phase, StartupPhase::Initializing);
        assert!(!node.is_ready);

        // Cached files require no network traffic, while startup can still be
        // initializing. Their remaining transfer must also be zero.
        state.model_cached = true;
        state.bytes_downloaded = 0;
        state.artifact_network_bytes_downloaded = 0;
        state.artifact_bytes_downloaded = state.artifact_bytes_total;
        update_observation(&mut node, &state, 3.0).unwrap();
        assert_eq!(node.host.bytes_downloaded, node.host.bytes_total);
        assert!(!node.is_ready);
    }
}
