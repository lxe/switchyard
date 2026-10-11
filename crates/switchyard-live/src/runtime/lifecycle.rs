//! Ownership-safe provisioning, observations, and resource cleanup.
use super::{LiveRuntime, Node, OwnedPod, PendingLaunch, nonce, rental_rate};
use crate::{Error, Result, serving::Endpoint};
use std::{collections::VecDeque, time::Duration};
use switchyard_agent::{Phase, Ticket};
use switchyard_core::scheduler::{HostObservation, HostProbe, ObservedNode, StartupPhase};
use switchyard_provider::{Launch, Provider};

impl<P: Provider> LiveRuntime<P> {
    pub(super) async fn provision(&mut self, profile: usize) -> Result<bool> {
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
        let name = format!("switchyard-{}-{}", self.run_id, self.nodes.len());
        let ticket_key = format!("{}/{name}.json", self.config.bootstrap.ticket_prefix);
        self.pending.push(PendingLaunch {
            name: name.clone(),
            ticket_key: ticket_key.clone(),
            resolved: false,
        });
        self.save("pending-launches.json", &self.pending)?;
        let manifest = self.manifests[&model.id].clone();
        let urls = manifest
            .chunks
            .iter()
            .map(|c| {
                Ok((
                    c.name.clone(),
                    self.storage
                        .presign(&format!("{}/{}", model.archive_prefix, c.name))?,
                ))
            })
            .collect::<switchyard_provider::Result<_>>()?;
        let ticket = Ticket {
            model: model.id.clone(),
            revision: model.revision.clone(),
            manifest,
            urls,
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
                probe: None,
            },
        };
        self.nodes.push(Node {
            owned,
            endpoint,
            observed,
            ticket_key,
            checked_tokenizer: false,
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
            match node.endpoint.state().await {
                Ok(state) => {
                    self.observations.push(
                        serde_json::json!({"at_s":now,"pod":node.owned.pod.id,"state":state}),
                    );
                    node.observed.host.bytes_downloaded = state.bytes_downloaded.saturating_add(
                        if state.hydration_mode
                            == switchyard_artifacts::HydrationMode::TrustedStreaming
                        {
                            state
                                .artifact_network_bytes_downloaded
                                .min(state.artifact_bytes_total)
                        } else {
                            state.artifact_bytes_downloaded
                        },
                    );
                    node.observed.host.bytes_total =
                        state.bytes_total.saturating_add(state.artifact_bytes_total);
                    node.observed.host.phase = match state.phase {
                        Phase::Booting => StartupPhase::Booting,
                        Phase::Hydrating => StartupPhase::Hydrating,
                        Phase::Initializing => StartupPhase::Initializing,
                        Phase::Exporting => StartupPhase::Initializing,
                        Phase::Ready => StartupPhase::Ready,
                        Phase::Failed => {
                            return Err(Error::Request(
                                state.error.unwrap_or_else(|| "agent failed".into()),
                            ));
                        }
                    };
                    if matches!(state.phase, Phase::Exporting) {
                        node.observed.is_ready = false;
                    }
                    if matches!(state.phase, Phase::Ready) && !node.observed.is_ready {
                        node.observed.is_ready = true;
                        node.observed.ready_s = now;
                        node.observed.last_used_s = now;
                        println!(
                            "{} ready after {:.1}s",
                            self.profiles[node.observed.profile].model,
                            now - node.owned.allocated_s
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
        if let Ok(state) = node.endpoint.state().await {
            let rate =
                if state.hydration_mode == switchyard_artifacts::HydrationMode::TrustedStreaming {
                    state
                        .artifact_network_bytes_per_s
                        .or(state.volume_bytes_per_s)
                } else {
                    state
                        .volume_bytes_per_s
                        .or(state.artifact_volume_bytes_per_s)
                };
            let Some(rate) = rate.filter(|r| r.is_finite() && *r > 0.0) else {
                return Ok(());
            };
            node.observed.host.probe = Some(HostProbe {
                at_s: self.epoch.elapsed().as_secs_f64(),
                volume_bytes_per_s: rate,
                image_cached: state.image_cached,
                model_cached: state.model_cached,
                hydration_bytes: state.bytes_total.saturating_add(state.artifact_bytes_total),
            });
            self.probes += 1;
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
