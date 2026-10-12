//! Ownership-safe provisioning, observations, and resource cleanup.
use super::LiveRuntime;
use crate::{Error, Result};
use std::time::Duration;
use switchyard_agent::Phase;
use switchyard_core::{
    probe::TransferCounter,
    scheduler::{ObservedNode, StartupPhase},
};
use switchyard_provider::Provider;

impl<P: Provider + 'static> LiveRuntime<P> {
    pub(super) async fn probe(&mut self, id: usize) -> Result<()> {
        let now = self.now();
        let Some(node) = self.nodes.get_mut(id) else {
            self.invalid += 1;
            return Ok(());
        };
        if self.hosts[node.host].owned.ended_s.is_some()
            || !self
                .scheduling
                .probe
                .due(now, node.observed.host.probe_attempted_s)
        {
            return Ok(());
        }
        node.observed.host.probe_attempted_s = Some(now);
        node.observed.host.probe_pending = true;
        self.schedule_observation(id);
        Ok(())
    }
    /// Confirm cloud termination before touching storage tickets or local receipts.
    async fn terminate_host(&mut self, id: usize) -> Result<()> {
        let Some(host) = self.hosts.get(id) else {
            self.invalid += 1;
            return Ok(());
        };
        if host.owned.ended_s.is_some() {
            return Ok(());
        }
        if let Some(pod) = self.provider.get(&host.owned.pod.id).await? {
            if pod.name != host.owned.pod.name {
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
        let host = &mut self.hosts[id];
        host.owned.ended_s = Some(now);
        for endpoint in &host.endpoints {
            self.nodes[*endpoint].observed.ended_s = Some(now);
            self.nodes[*endpoint].observed.is_ready = false;
            self.nodes[*endpoint].observed.host.phase = StartupPhase::Ended;
        }
        self.releases += 1;
        Ok(())
    }

    pub(super) async fn release(&mut self, id: usize) -> Result<()> {
        let Some(node) = self.nodes.get(id) else {
            self.invalid += 1;
            return Ok(());
        };
        let host = node.host;
        if self.hosts[host].owned.ended_s.is_some() {
            return Ok(());
        }
        let endpoints = self.hosts[host].endpoints.clone();
        if endpoints
            .iter()
            .any(|id| !self.nodes[*id].observed.active.is_empty())
        {
            self.invalid += 1;
            return Ok(());
        }
        for endpoint in endpoints {
            if let Some(cache) = self.collect_native_cache(endpoint).await {
                self.collected_native_caches.push((endpoint, cache));
            }
        }
        self.terminate_host(host).await?;
        self.storage.delete(&self.hosts[host].ticket_key).await?;
        self.journal()
    }

    /// Terminate every owned allocation even when a ticket or receipt cannot be written.
    pub async fn cleanup(&mut self) -> Result<()> {
        let launch_failure = self.drain_launches().await.err();
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

        let mut failures = launch_failure
            .into_iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>();
        for id in 0..self.hosts.len() {
            for attempt in 0..3 {
                match self.terminate_host(id).await {
                    Ok(()) => break,
                    Err(error) if attempt == 2 => failures.push(error.to_string()),
                    Err(_) => tokio::time::sleep(Duration::from_secs(2)).await,
                }
            }
        }
        let confirmed = self
            .hosts
            .iter()
            .filter(|n| n.owned.ended_s.is_some())
            .map(|n| n.owned.pod.name.clone())
            .collect();
        if let Err(error) = crate::ownership::reconcile(
            self.provider.as_ref(),
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
        if self.hosts.iter().all(|host| host.owned.ended_s.is_some())
            && self.pending.iter().all(|launch| launch.resolved)
        {
            for (id, cache) in std::mem::take(&mut self.collected_native_caches) {
                self.publish_native_cache(id, &cache).await;
            }
        }
        // Reconciliation also deletes pending tickets. Keep deletion idempotent for nodes
        // restored from an ownership journal that predates a pending-launch record.
        for node in self.hosts.iter().filter(|n| n.owned.ended_s.is_some()) {
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
    use std::collections::VecDeque;
    use switchyard_core::ProbePolicy;
    use switchyard_core::{probe::BandwidthWindow, scheduler::HostObservation};
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
            placement: None,
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
