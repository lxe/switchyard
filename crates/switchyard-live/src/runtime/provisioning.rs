//! Bounded launch preparation; no mutable routing state crosses cloud I/O.
use super::*;
use crate::{Result, serving::Endpoint};
use std::collections::VecDeque;
use switchyard_agent::Ticket;
use switchyard_core::scheduler::{PendingAllocation, SchedulerFeedback};
use switchyard_core::{
    probe::BandwidthWindow,
    scheduler::{HostObservation, ObservedNode, StartupPhase},
};
use switchyard_provider::{Launch, Provider};

pub(super) struct LaunchContext<P: Provider> {
    provider: Arc<P>,
    storage: Storage,
    config: ExecutionConfig,
    manifests: BTreeMap<String, ModelManifest>,
    runtime_manifest: Option<switchyard_artifacts::Manifest>,
    component_manifests: BTreeMap<String, switchyard_artifacts::Manifest>,
    cache_manifests: BTreeMap<String, switchyard_artifacts::Manifest>,
    cache_export: Option<switchyard_agent::export::CacheExportConfig>,
    epoch: Instant,
    out: PathBuf,
}
pub(super) struct LaunchOutcome {
    intent: usize,
    profile: usize,
    offering: Offering,
    ticket_key: String,
    token: String,
    allocated_s: f64,
    provider_elapsed_s: f64,
    endpoints: Vec<LaunchedEndpoint>,
    plan: Option<switchyard_core::topology::HostPlan>,
    result: switchyard_provider::Result<Pod>,
}
struct LaunchedEndpoint {
    profile: usize,
    id: Option<String>,
    binding: Option<crate::native_cache_registry::CacheBinding>,
}
struct LaunchModel {
    profile: usize,
    id: Option<String>,
    port: u16,
    model: crate::config::ModelConfig,
}
impl<P: Provider> LaunchContext<P> {
    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }
    async fn prepare_ticket(
        &self,
        model: &crate::config::ModelConfig,
        gpu: &str,
        endpoint_id: Option<&str>,
        port: u16,
    ) -> Result<(Ticket, Option<crate::native_cache_registry::CacheBinding>)> {
        let manifest = self.manifests[&model.id].clone();
        let source = model.artifact.ticket(&self.storage, manifest)?;
        let (native_cache_binding, learned_cache) = self.lookup_native_cache(model, gpu).await;
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
        let mut ticket = Ticket {
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
                for cache in model.caches.iter().filter(|c| c.gpu == gpu) {
                    caches.push(switchyard_agent::runtime::CacheTicket {
                        name: cache.name.clone(),
                        artifact: cache.artifact.ticket(
                            &self.storage,
                            self.cache_manifests[&cache.artifact.id].clone(),
                        )?,
                        compatibility: cache.requirements(runtime, model)?,
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
        if let Some(endpoint_id) = endpoint_id {
            let mut startup = self.config.bootstrap.startup.clone();
            let base = startup.model_root.as_deref().unwrap_or("/work/switchyard");
            startup.model_root = Some(format!(
                "{base}/models/{}",
                ticket.source.artifact_identity()
            ));
            ticket.startup = startup;
            ticket.engine_args = model
                .engine
                .args(model, &ticket.startup.model_directory().to_string_lossy());
            *ticket.engine_args.last_mut().expect("native bind") = format!("127.0.0.1:{port}");
            if let Some(native) = &model.engine.native
                && let Some(lora) = &native.lora
            {
                // Uploaded adapter files belong to the endpoint, not a sibling registry.
                let mut config: serde_json::Value = serde_json::from_str(&ticket.engine_args[1])?;
                config["lora"]["root"] = serde_json::json!(lora.root.join(endpoint_id));
                ticket.engine_args[1] = serde_json::to_string(&config)?;
            }
        }
        Ok((ticket, native_cache_binding))
    }
    async fn run(
        self,
        intent: usize,
        profile: usize,
        offering: Offering,
        models: Vec<LaunchModel>,
        ownership: (String, String),
        plan: Option<switchyard_core::topology::HostPlan>,
    ) -> Result<LaunchOutcome> {
        let (name, ticket_key) = ownership;
        let mut tickets = vec![];
        let mut endpoints = vec![];
        for model in models {
            let (ticket, binding) = self
                .prepare_ticket(&model.model, &offering.gpu, model.id.as_deref(), model.port)
                .await?;
            endpoints.push(LaunchedEndpoint {
                profile: model.profile,
                id: model.id.clone(),
                binding,
            });
            tickets.push((model.id, model.port, ticket));
        }
        if let Some(plan) = &plan {
            let ticket = switchyard_agent::host::HostTicket {
                schema_version: 1,
                storage_root: None,
                plan: plan.clone(),
                endpoints: tickets
                    .into_iter()
                    .map(
                        |(id, port, ticket)| switchyard_agent::host::HostedEndpoint {
                            id: id.expect("host endpoint ID"),
                            port,
                            ticket,
                        },
                    )
                    .collect(),
            };
            ticket
                .validate()
                .map_err(|e| Error::Config(e.to_string()))?;
            self.storage.put_json(&ticket_key, &ticket).await?;
        } else {
            self.storage
                .put_json(&ticket_key, &tickets.pop().expect("single endpoint").2)
                .await?;
        }

        let token = nonce()?;
        let allocated = self.now();
        let provider_started = self.now();
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
        Ok(LaunchOutcome {
            intent,
            profile,
            offering,
            ticket_key,
            token,
            allocated_s: allocated,
            provider_elapsed_s: self.now() - provider_started,
            endpoints,
            plan,
            result: launched,
        })
    }
    async fn lookup_native_cache(
        &self,
        model: &crate::config::ModelConfig,
        gpu: &str,
    ) -> (
        Option<crate::native_cache_registry::CacheBinding>,
        Option<switchyard_agent::runtime::CacheTicket>,
    ) {
        super::native_cache::lookup(
            &self.storage,
            &self.config,
            model,
            gpu,
            self.epoch,
            &self.out,
        )
        .await
    }
}

impl<P: Provider + 'static> LiveRuntime<P> {
    /// Reserve ownership and budget before dispatching any asynchronous launch work.
    pub(super) fn start_provision(&mut self, profile: usize) -> Result<bool> {
        if self
            .provisioning_cooldowns
            .get(&profile)
            .is_some_and(|t| *t > self.now())
        {
            return Ok(false);
        }
        let Some(p) = self.profiles.get(profile) else {
            self.invalid += 1;
            return Ok(false);
        };
        let h = &self.scheduling.heuristics;
        let count = self
            .hosts
            .iter()
            .filter(|n| n.owned.ended_s.is_none())
            .count()
            + self.inflight.len();
        let reserved_rate: f64 = self
            .inflight
            .values()
            .map(|a| self.profiles[a.profile].usd_per_hour)
            .sum();
        let owned_rate: f64 = self
            .hosts
            .iter()
            .filter(|n| n.owned.ended_s.is_none())
            .map(|n| rental_rate(&n.owned))
            .sum();
        let reserved_cost: f64 = self
            .inflight
            .values()
            .map(|a| {
                (self.now() - a.started_s).max(1.0) * self.profiles[a.profile].usd_per_hour / 3600.0
            })
            .sum();
        if self.inflight.len() >= h.maximum_pending_allocations
            || count >= self.config.limits.max_instances
            || owned_rate + reserved_rate + p.usd_per_hour > self.config.limits.max_usd_per_hour
            || self.rental_usd() + reserved_cost + p.usd_per_hour / 3600.0
                >= self.config.limits.max_run_usd
        {
            return Ok(false);
        }
        self.limits()?;
        let plan = self.config.topology.plan_for(&p.id).cloned();
        let deployments: Vec<_> = plan.as_ref().map_or_else(
            || vec![(profile, None)],
            |plan| {
                plan.endpoints
                    .iter()
                    .map(|e| {
                        (
                            self.profiles
                                .iter()
                                .position(|p| p.id == e.profile)
                                .expect("validated profile"),
                            Some(e.id.clone()),
                        )
                    })
                    .collect()
            },
        );
        let models: Vec<_> = deployments
            .into_iter()
            .enumerate()
            .map(|(i, (profile, id))| {
                Ok(LaunchModel {
                    profile,
                    id,
                    port: 8100 + i as u16,
                    model: self
                        .config
                        .models
                        .iter()
                        .find(|m| m.id == self.profiles[profile].model)
                        .ok_or_else(|| Error::Config("profile model missing".into()))?
                        .clone(),
                })
            })
            .collect::<Result<_>>()?;
        let mut additions = BTreeMap::<&str, usize>::new();
        for model in &models {
            *additions.entry(&model.model.id).or_default() += 1;
        }
        for pending in self.inflight.values() {
            let p = &self.profiles[pending.profile];
            if let Some(plan) = self.config.topology.plan_for(&p.id) {
                for endpoint in &plan.endpoints {
                    let p = self
                        .profiles
                        .iter()
                        .find(|p| p.id == endpoint.profile)
                        .expect("validated topology");
                    *additions.entry(&p.model).or_default() += 1;
                }
            } else {
                *additions.entry(&p.model).or_default() += 1;
            }
        }
        if additions.iter().any(|(model, count)| {
            self.nodes
                .iter()
                .filter(|n| {
                    n.observed.ended_s.is_none()
                        && self.profiles[n.observed.profile].model == **model
                })
                .count()
                + count
                > self.config.limits.max_endpoints_per_model
        }) {
            return Ok(false);
        }
        let offering = self.profile_offerings[profile].clone();
        let intent = self.pending.len();
        let name = format!("switchyard-{}-{intent}", self.run_id);
        let ticket_key = format!("{}/{name}.json", self.config.bootstrap.ticket_prefix);
        self.pending.push(PendingLaunch {
            name: name.clone(),
            ticket_key: ticket_key.clone(),
            resolved: false,
        });
        self.save("pending-launches.json", &self.pending)?;
        let started_s = self.now();
        self.inflight.insert(
            intent,
            PendingAllocation {
                profile,
                started_s,
                expected_ready_s: started_s
                    + models
                        .iter()
                        .map(|m| self.profiles[m.profile].startup.total())
                        .fold(0.0, f64::max),
            },
        );
        let context = LaunchContext {
            provider: self.provider.clone(),
            storage: self.storage.clone(),
            config: self.config.clone(),
            manifests: self.manifests.clone(),
            runtime_manifest: self.runtime_manifest.clone(),
            component_manifests: self.component_manifests.clone(),
            cache_manifests: self.cache_manifests.clone(),
            cache_export: self.cache_export.clone(),
            epoch: self.epoch,
            out: self.out.clone(),
        };
        self.launch_tasks.spawn(async move {
            (
                intent,
                context
                    .run(intent, profile, offering, models, (name, ticket_key), plan)
                    .await,
            )
        });
        Ok(true)
    }
    pub(super) fn finish_launches(&mut self) -> Result<()> {
        while let Some(result) = self.launch_tasks.try_join_next() {
            self.finish_launch(result)?;
        }
        Ok(())
    }
    fn finish_launch(
        &mut self,
        result: std::result::Result<(usize, Result<LaunchOutcome>), tokio::task::JoinError>,
    ) -> Result<bool> {
        let (intent, outcome) = result.map_err(|_| {
            Error::Request("launch task failed; reconcile pending ownership".into())
        })?;
        self.inflight.remove(&intent);
        let outcome = outcome?;
        let LaunchOutcome {
            intent,
            profile,
            offering,
            ticket_key,
            token,
            allocated_s: allocated,
            provider_elapsed_s,
            endpoints,
            plan,
            result,
        } = outcome;
        self.controller.observe(SchedulerFeedback::Allocation {
            profile,
            accepted: result.is_ok(),
            elapsed_s: provider_elapsed_s,
        });
        super::telemetry::append(
            &self.out,
            "allocation-outcomes.ndjson",
            &serde_json::json!({
            "at_s":self.now(), "intent":intent, "profile":profile, "accepted":result.is_ok(), "provider_elapsed_s":provider_elapsed_s }),
        )?;
        let pod = match result {
            Ok(pod) => pod,
            Err(error) => {
                if !matches!(error, switchyard_provider::Error::UnknownLaunch { .. }) {
                    self.pending[intent].resolved = true;
                    self.save("pending-launches.json", &self.pending)?;
                }
                if let switchyard_provider::Error::AllocationRejected { status } = error {
                    let at_s = self.now();
                    let retry_at_s = at_s + self.scheduling.retry_cooldown_s;
                    self.provisioning_cooldowns.insert(profile, retry_at_s);
                    if let Some(plan) = &plan {
                        for endpoint in &plan.endpoints {
                            if let Some(index) =
                                self.profiles.iter().position(|p| p.id == endpoint.profile)
                            {
                                self.provisioning_cooldowns.insert(index, retry_at_s);
                            }
                        }
                    }
                    self.allocation_rejections.push(AllocationRejection {
                        at_s,
                        profile,
                        name: self.pending[intent].name.clone(),
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
        let host_id = self.hosts.len();
        let first_endpoint = self.nodes.len();
        self.hosts.push(Host {
            owned,
            ticket_key,
            endpoints: vec![],
            plan: plan.as_ref().map(|p| p.id.clone()),
        });
        for launched in endpoints {
            let profile = launched.profile;
            let p = &self.profiles[profile];
            let base = self.provider.endpoint_url(&pod, 8080);
            let url = launched.id.as_ref().map_or(base.clone(), |id| {
                format!("{}/endpoints/{id}", base.trim_end_matches('/'))
            });
            let endpoint = Endpoint::new(url, &token, self.config.limits.request_timeout_s)?;
            let placement = plan.as_ref().zip(launched.id.as_ref()).map(|(plan, id)| {
                plan.placement(
                    first_endpoint,
                    plan.endpoints
                        .iter()
                        .find(|e| &e.id == id)
                        .expect("validated endpoint"),
                )
            });
            let observed = ObservedNode {
                placement,
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
                host: host_id,
                endpoint,
                observed,
                checked_tokenizer: false,
                bandwidth: BandwidthWindow::default(),
                native_cache_binding: launched.binding,
                successful_calls: 0,
                cache_collection_attempted: false,
                cache_quiescence_known: true,
            });
            self.hosts[host_id].endpoints.push(self.nodes.len() - 1);
        }
        self.pending[intent].resolved = true;
        self.save("pending-launches.json", &self.pending)?;
        self.journal()?;
        self.launches += 1;
        let actual_rate: f64 = self
            .hosts
            .iter()
            .filter(|n| n.owned.ended_s.is_none())
            .map(|n| rental_rate(&n.owned))
            .sum();
        let reserved_rate: f64 = self
            .inflight
            .values()
            .map(|a| self.profiles[a.profile].usd_per_hour)
            .sum();
        if actual_rate + reserved_rate > self.config.limits.max_usd_per_hour {
            return Err(Error::Limit(
                "actual hourly pod quotes exceed the fleet budget".into(),
            ));
        }
        self.limits()?;
        Ok(true)
    }
    /// Baking waits for its owned allocation; workload execution uses start_provision.
    pub(super) async fn provision(&mut self, profile: usize) -> Result<bool> {
        if !self.start_provision(profile)? {
            return Ok(false);
        }
        let result = self.launch_tasks.join_next().await.expect("started launch");
        self.finish_launch(result)
    }
    /// Never cancel a provider create with an ambiguous paid outcome.
    pub(super) async fn drain_launches(&mut self) -> Result<()> {
        let mut failure = None;
        while let Some(result) = self.launch_tasks.join_next().await {
            if let Err(error) = self.finish_launch(result) {
                failure = Some(error);
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(())
    }
}
