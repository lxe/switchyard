//! Owned lifecycle cache collection, separate from inference scheduling and baking.
use super::LiveRuntime;
use crate::{
    Error, Result,
    config::ModelConfig,
    native_cache_registry::{self, CacheBinding, CollectedCache},
    serving::Endpoint,
};
use std::{path::Path, time::Duration};
use switchyard_agent::{export::ExportPhase, runtime::CacheTicket};
use switchyard_provider::Provider;

impl<P: Provider + 'static> LiveRuntime<P> {
    fn cache_event(&self, mut event: serde_json::Value) {
        if let Some(fields) = event.as_object_mut() {
            fields.insert("at_s".into(), serde_json::json!(self.now()));
        }
        // Optional telemetry never prevents serving or paid-resource cleanup.
        if super::telemetry::append(&self.out, "native-cache-registry.ndjson", &event).is_err() {
            eprintln!("native cache registry telemetry write failed");
        }
    }

    pub(super) async fn collect_native_cache(&mut self, id: usize) -> Option<CollectedCache> {
        let config = self.config.bootstrap.native_cache_registry.as_ref()?;
        let node = self.nodes.get_mut(id)?;
        if self.hosts[node.host].owned.ended_s.is_some()
            || node.cache_collection_attempted
            || !node.cache_quiescence_known
            || node.successful_calls == 0
            || !node.observed.is_ready
            || !node.observed.active.is_empty()
            || self.cache_export.is_some()
        {
            return None;
        }
        let binding = node.native_cache_binding.clone()?;
        node.cache_collection_attempted = true;
        node.observed.is_ready = false;
        let endpoint = node.endpoint.clone();
        let pod = self.hosts[node.host].owned.pod.id.clone();
        let directory = self.out.join(format!("native-cache-{id}"));
        let began = self.now();
        let result = tokio::time::timeout(
            Duration::from_secs(config.collection_timeout_s),
            collect(&endpoint, config, binding.clone(), &directory),
        )
        .await;
        match result {
            Ok(Ok(cache)) => {
                self.cache_event(serde_json::json!({"phase":"collected","pod":pod,"binding":binding.identity,"artifact":cache.manifest.id,"elapsed_s":self.now()-began}));
                Some(cache)
            }
            Ok(Err(error)) => {
                self.cache_event(serde_json::json!({"phase":"collection_failed","pod":pod,"binding":binding.identity,"error":error.to_string(),"elapsed_s":self.now()-began}));
                None
            }
            Err(_) => {
                self.cache_event(serde_json::json!({"phase":"collection_timeout","pod":pod,"binding":binding.identity,"elapsed_s":self.now()-began}));
                None
            }
        }
    }

    pub(super) async fn publish_native_cache(&self, id: usize, cache: &CollectedCache) {
        let Some(config) = &self.config.bootstrap.native_cache_registry else {
            return;
        };
        if self.hosts[self.nodes[id].host].owned.ended_s.is_none() {
            return;
        }
        let began = self.now();
        let result = tokio::time::timeout(
            Duration::from_secs(config.publication_timeout_s),
            native_cache_registry::publish(&self.storage, config, cache),
        )
        .await;
        let (phase, detail) = match result {
            Ok(Ok(receipt)) => (
                "published",
                serde_json::to_value(receipt).unwrap_or_default(),
            ),
            Ok(Err(error)) => (
                "publication_failed",
                serde_json::json!({"error":error.to_string()}),
            ),
            Err(_) => ("publication_timeout", serde_json::Value::Null),
        };
        self.cache_event(serde_json::json!({"phase":phase,"pod":self.hosts[self.nodes[id].host].owned.pod.id,"binding":cache.binding.identity,"elapsed_s":self.now()-began,"detail":detail}));
    }
}

async fn collect(
    endpoint: &Endpoint,
    config: &native_cache_registry::NativeCacheRegistryConfig,
    binding: CacheBinding,
    directory: &Path,
) -> Result<CollectedCache> {
    let mut state = endpoint.start_cache_export().await?;
    let manifest = loop {
        match state.phase {
            ExportPhase::Ready => {
                break state.manifest.ok_or_else(|| {
                    Error::Request("ready native cache export lacks manifest".into())
                })?;
            }
            ExportPhase::Failed => return Err(Error::Request("native cache export failed".into())),
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
        state = endpoint.cache_export_status().await?;
    };
    native_cache_registry::validate_manifest(config, &binding, &manifest.id, &manifest)?;
    endpoint
        .download_cache_export(
            &manifest,
            &binding.compatibility,
            directory,
            config.max_artifact_bytes,
            config.transfer_concurrency,
        )
        .await?;
    Ok(CollectedCache {
        directory: directory.into(),
        manifest,
        binding,
    })
}

pub(super) async fn lookup(
    storage: &switchyard_provider::storage::Storage,
    execution: &crate::config::ExecutionConfig,
    model: &ModelConfig,
    gpu: &str,
    epoch: std::time::Instant,
    out: &Path,
) -> (Option<CacheBinding>, Option<CacheTicket>) {
    let event = |mut value: serde_json::Value| {
        if let Some(fields) = value.as_object_mut() {
            fields.insert(
                "at_s".into(),
                serde_json::json!(epoch.elapsed().as_secs_f64()),
            );
        }
        let _ = super::telemetry::append(out, "native-cache-registry.ndjson", &value);
    };
    let Some(config) = &execution.bootstrap.native_cache_registry else {
        return (None, None);
    };
    let binding = match config.binding(execution, model, gpu) {
        Ok(binding) => binding,
        Err(error) => {
            event(
                serde_json::json!({"phase":"binding_failed","model":model.id,"error":error.to_string()}),
            );
            return (None, None);
        }
    };
    let began = epoch.elapsed().as_secs_f64();
    let lookup = tokio::time::timeout(
        Duration::from_secs(config.lookup_timeout_s),
        native_cache_registry::lookup(storage, config, &binding),
    )
    .await;
    let ticket = match lookup {
        Ok(Ok(Some((artifact, manifest)))) => match artifact.ticket(storage, manifest) {
            Ok(artifact) => {
                event(
                    serde_json::json!({"phase":"hit","binding":binding.identity,"model":model.id,"gpu":gpu,"artifact":artifact.manifest.id,"elapsed_s":epoch.elapsed().as_secs_f64()-began}),
                );
                Some(CacheTicket {
                    name: config.name.clone(),
                    artifact,
                    compatibility: binding.compatibility.clone(),
                })
            }
            Err(error) => {
                event(
                    serde_json::json!({"phase":"lookup_failed","binding":binding.identity,"error":error.to_string()}),
                );
                None
            }
        },
        Ok(Ok(None)) => {
            event(
                serde_json::json!({"phase":"miss","binding":binding.identity,"model":model.id,"gpu":gpu,"elapsed_s":epoch.elapsed().as_secs_f64()-began}),
            );
            None
        }
        Ok(Err(error)) => {
            event(
                serde_json::json!({"phase":"lookup_failed","binding":binding.identity,"error":error.to_string(),"elapsed_s":epoch.elapsed().as_secs_f64()-began}),
            );
            None
        }
        Err(_) => {
            event(
                serde_json::json!({"phase":"lookup_timeout","binding":binding.identity,"elapsed_s":epoch.elapsed().as_secs_f64()-began}),
            );
            None
        }
    };
    (Some(binding), ticket)
}
