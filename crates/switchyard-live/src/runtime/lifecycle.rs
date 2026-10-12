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
        let p = p.clone();
        let model = self
            .config
            .models
            .iter()
            .find(|m| m.id == p.model)
            .ok_or_else(|| Error::Config("profile model missing".into()))?
            .clone();
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
        let (native_cache_binding, learned_cache) =
            self.lookup_native_cache(&model, &offering.gpu).await;
        let learned_export = self
            .config
            .bootstrap
            .native_cache_registry
            .as_ref()
            .zip(native_cache_binding.as_ref())
            .map(
                |(registry, binding)| switchyard_agent::export::CacheExportConfig {
                    name: registry.name.clone(),
                    compatibility: binding.compatibility.clone(),
                    drain_timeout_s: 1,
                    shutdown_timeout_s: 5,
                },
            );
        let ticket = Ticket {
            source,
            model: model.id.clone(),
            revision: model.revision.clone(),
            engine_args: model.engine.args(
                &model,
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
                        compatibility: cache.requirements(runtime, &model)?,
                    });
                }
                if let Some(cache) = learned_cache {
                    caches.push(cache);
                }
                Some(switchyard_agent::runtime::RuntimeTicket {
                    components: runtime
                        .components
                        .iter()
                        .map(|part| {
                            Ok(switchyard_agent::runtime::RuntimeComponent {
                                name: part.name.clone(),
                                artifact: part.artifact.ticket(
                                    &self.storage,
                                    self.component_manifests[&part.name].clone(),
                                )?,
                                compatibility: part.compatibility.clone(),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
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
                    cache_export: self.cache_export.clone().or(learned_export),
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
                remaining_bytes_known: false,
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
            native_cache_binding,
            successful_calls: 0,
            cache_collection_attempted: false,
            cache_quiescence_known: true,
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
        if let Some(cache) = self.collect_native_cache(id).await {
            self.collected_native_caches.push((id, cache));
        }
        self.terminate_node(id).await?;
        self.storage.delete(&self.nodes[id].ticket_key).await?;
        self.journal()
    }

    /// Terminate every owned allocation even when a ticket or receipt cannot be written.
    pub async fn cleanup(&mut self) -> Result<()> {
        // Remember active requests before cancellation; an aborted client is not
        // proof that the remote engine has finished it. Such nodes skip export.
        let collectable = self
            .nodes
            .iter()
            .map(|node| node.observed.active.is_empty())
            .collect::<Vec<_>>();
        self.cancel_observations().await;
        self.abort_requests().await;
        for (id, allowed) in collectable.into_iter().enumerate() {
            if allowed && let Some(cache) = self.collect_native_cache(id).await {
                self.collected_native_caches.push((id, cache));
            }
        }

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
        // Do not inject object publication into the scheduler's routing loop.
        // Terminal cleanup publishes only after all known and ambiguous owned
        // allocations have been reconciled; no paid host waits for uploads.
        if self.nodes.iter().all(|node| node.owned.ended_s.is_some())
            && self.pending.iter().all(|launch| launch.resolved)
        {
            for (id, cache) in std::mem::take(&mut self.collected_native_caches) {
                self.publish_native_cache(id, &cache).await;
            }
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

pub(super) fn received_counters(state: &switchyard_agent::State) -> Vec<TransferCounter> {
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

fn artifact_progress(state: &switchyard_agent::State) -> (u64, bool) {
    if state.artifacts.is_empty() {
        // Aggregate wire volume can contain discarded retry attempts. Older
        // observations without retained counters provide only a progress floor.
        let completed = state
            .artifact_bytes_downloaded
            .min(state.artifact_bytes_total);
        return if matches!(state.phase, Phase::Ready) {
            (state.artifact_bytes_total, true)
        } else {
            (
                completed,
                state.hydration_mode == switchyard_artifacts::HydrationMode::Verified
                    || completed == state.artifact_bytes_total,
            )
        };
    }
    state
        .artifacts
        .iter()
        .fold((0u64, true), |(bytes, known), artifact| {
            let completed = artifact.downloaded_bytes.min(artifact.total_bytes);
            let (progress, observed) = if matches!(
                artifact.phase,
                switchyard_agent::runtime::ArtifactPhase::Ready
            ) {
                // Ready proves publication, including trusted cache reuse with zero
                // wire and verified-byte counters.
                (artifact.total_bytes, true)
            } else if state.hydration_mode == switchyard_artifacts::HydrationMode::Verified {
                (completed, true)
            } else if let Some(reception) = &artifact.reception {
                // Reserved reception rejects retries and writes each prefix once.
                // Its cumulative storage counter includes consumed/reclaimed data.
                (
                    completed
                        .max(reception.storage_written_bytes)
                        .min(artifact.total_bytes),
                    true,
                )
            } else {
                // Retrying streaming has no retained compressed-byte counter. Keep
                // proved progress without converting repeated wire bytes to progress.
                (completed, completed == artifact.total_bytes)
            };
            (bytes.saturating_add(progress), known && observed)
        })
}

pub(super) fn update_observation(
    node: &mut ObservedNode,
    state: &switchyard_agent::State,
    at_s: f64,
) -> Result<bool> {
    let model_received = if state.model_cached {
        state.bytes_total
    } else {
        state.bytes_downloaded
    };
    let (artifact_received, progress_known) = artifact_progress(state);
    node.host.bytes_downloaded = model_received.saturating_add(artifact_received);
    node.host.remaining_bytes_known = progress_known;
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
    fn node() -> ObservedNode {
        ObservedNode {
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
                remaining_bytes_known: false,
                probe_pending: false,
                probe_attempted_s: None,
                probe: None,
            },
        }
    }
    #[test]
    fn mixed_cached_and_receiving_artifacts_contribute_progress_individually() {
        let mut state = state();
        state.bytes_downloaded = state.bytes_total;
        state.phase = Phase::Initializing;
        state.artifacts[0].cached = true;
        state.artifacts[0].phase = switchyard_agent::runtime::ArtifactPhase::Ready;
        state.artifacts[0].total_bytes = 4000;
        // Trusted cache reuse reports Ready with both counters zero.
        state.artifacts[0].downloaded_bytes = 0;
        state.artifacts[0].network_bytes_downloaded = 0;
        state.artifacts[0].unpacked_bytes = state.artifacts[0].total_unpacked_bytes;
        let mut receiving = state.artifacts[0].clone();
        receiving.name = "cache".into();
        receiving.id = "cache1".into();
        receiving.cached = false;
        receiving.phase = switchyard_agent::runtime::ArtifactPhase::Downloading;
        receiving.total_bytes = 6000;
        receiving.downloaded_bytes = 0;
        receiving.unpacked_bytes = 0;
        receiving.network_bytes_downloaded = 2000;
        receiving.reception = Some(switchyard_artifacts::ReceptionProgress {
            storage_written_bytes: 2000,
            ..Default::default()
        });
        state.artifacts.push(receiving);
        state.artifact_bytes_downloaded = 0;
        let mut node = node();
        for (wire, stored, phase, expected) in [
            (
                2000,
                2000,
                switchyard_agent::runtime::ArtifactPhase::Downloading,
                7000,
            ),
            (
                6000,
                2000,
                switchyard_agent::runtime::ArtifactPhase::Extracting,
                7000,
            ),
            (
                6000,
                6000,
                switchyard_agent::runtime::ArtifactPhase::Extracting,
                11000,
            ),
        ] {
            state.artifact_network_bytes_downloaded = wire;
            state.artifacts[1].network_bytes_downloaded = wire;
            state.artifacts[1].phase = phase;
            let reception = state.artifacts[1].reception.as_mut().unwrap();
            reception.network_received_bytes = wire;
            reception.storage_written_bytes = stored;
            reception.buffered_bytes = stored;
            state.hydration_mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
            update_observation(&mut node, &state, 2.0).unwrap();
            assert_eq!(node.host.bytes_downloaded, expected);
            assert!(node.host.remaining_bytes_known);
            assert_eq!(node.host.bytes_total, 11000);
            assert_eq!(node.host.phase, StartupPhase::Initializing);
            assert!(!node.is_ready);
            state.hydration_mode = switchyard_artifacts::HydrationMode::Verified;
            update_observation(&mut node, &state, 2.0).unwrap();
            assert_eq!(node.host.bytes_downloaded, 5000);
        }
    }
    #[test]
    fn retry_wire_bytes_do_not_complete_an_incomplete_transfer() {
        let mut state = state();
        state.hydration_mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
        state.bytes_downloaded = state.bytes_total;
        state.artifacts[0].downloaded_bytes = 0;
        state.artifacts[0].network_bytes_downloaded = 20000;
        state.artifacts[0].phase = switchyard_agent::runtime::ArtifactPhase::Extracting;
        state.artifact_bytes_downloaded = 0;
        state.artifact_network_bytes_downloaded = 20000;
        let mut node = node();
        update_observation(&mut node, &state, 2.0).unwrap();
        assert_eq!(node.host.bytes_downloaded, 1000);
        assert!(node.host.bytes_downloaded < node.host.bytes_total);
        assert!(!node.host.remaining_bytes_known);
    }
    #[test]
    fn aggregate_only_progress_uses_verified_floor_without_trusting_retry_wire() {
        let mut state = state();
        state.artifacts.clear();
        state.bytes_downloaded = state.bytes_total;
        state.artifact_network_bytes_downloaded = 6000;
        state.artifact_bytes_downloaded = 4000;
        let mut node = node();
        for (mode, known) in [
            (switchyard_artifacts::HydrationMode::TrustedStreaming, false),
            (switchyard_artifacts::HydrationMode::Verified, true),
        ] {
            state.hydration_mode = mode;
            update_observation(&mut node, &state, 2.0).unwrap();
            assert_eq!(node.host.bytes_downloaded, 5000);
            assert_eq!(node.host.remaining_bytes_known, known);
        }
        state.hydration_mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
        state.phase = Phase::Ready;
        state.artifact_bytes_downloaded = 0;
        state.artifact_network_bytes_downloaded = 0;
        update_observation(&mut node, &state, 3.0).unwrap();
        assert_eq!(node.host.bytes_downloaded, node.host.bytes_total);
        assert!(node.host.remaining_bytes_known);
    }
    #[test]
    fn trusted_non_cached_ready_credits_publication_with_zero_verified_bytes() {
        let mut state = state();
        state.hydration_mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
        state.bytes_downloaded = state.bytes_total;
        state.phase = Phase::Initializing;
        state.artifacts[0].phase = switchyard_agent::runtime::ArtifactPhase::Ready;
        state.artifacts[0].downloaded_bytes = 0;
        state.artifacts[0].network_bytes_downloaded = 20000;
        state.artifact_bytes_downloaded = 0;
        state.artifact_network_bytes_downloaded = 20000;
        let mut node = node();
        update_observation(&mut node, &state, 2.0).unwrap();
        assert_eq!(node.host.bytes_downloaded, node.host.bytes_total);
        assert!(node.host.remaining_bytes_known);
        assert!(!node.is_ready);
    }
    #[test]
    fn completed_probe_snapshot_updates_remaining_transfer_before_decision() {
        let mut state = state();
        state.hydration_mode = switchyard_artifacts::HydrationMode::TrustedStreaming;
        state.bytes_downloaded = state.bytes_total;
        state.artifact_network_bytes_downloaded = state.artifact_bytes_total;
        state.artifact_bytes_downloaded = state.artifact_bytes_total;
        // Legacy agents have aggregate counters without artifact entries.
        state.artifacts.clear();
        state.phase = Phase::Initializing;
        let mut node = node();
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
