//! Opt-in shard and publication timings; counters never contain URLs or payloads.
use serde::Serialize;
use std::{collections::BTreeMap, sync::Arc, time::Instant};

#[derive(Debug, Clone, Serialize)]
pub struct HydrationEvent {
    pub artifact_id: String,
    pub event: String,
    pub object: Option<String>,
    pub elapsed_seconds: f64,
    pub unix_seconds: f64,
    pub metrics: BTreeMap<String, f64>,
}
#[derive(Clone)]
pub struct HydrationObserver(Arc<dyn Fn(HydrationEvent) + Send + Sync>);
impl std::fmt::Debug for HydrationObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HydrationObserver")
    }
}
impl HydrationObserver {
    pub fn new(callback: impl Fn(HydrationEvent) + Send + Sync + 'static) -> Self {
        Self(Arc::new(callback))
    }
}
#[derive(Clone)]
pub(crate) struct Profile {
    pub epoch: Instant,
    artifact_id: String,
    observer: Option<HydrationObserver>,
}
impl Profile {
    pub fn new(artifact_id: String, observer: Option<HydrationObserver>) -> Self {
        Self {
            epoch: Instant::now(),
            artifact_id,
            observer,
        }
    }
    pub fn enabled(&self) -> bool {
        self.observer.is_some()
    }
    pub fn emit(
        &self,
        event: &str,
        object: Option<&str>,
        metrics: impl IntoIterator<Item = (&'static str, f64)>,
    ) {
        if let Some(observer) = &self.observer {
            (observer.0)(HydrationEvent {
                artifact_id: self.artifact_id.clone(),
                event: event.into(),
                object: object.map(str::to_owned),
                elapsed_seconds: self.epoch.elapsed().as_secs_f64(),
                unix_seconds: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
                metrics: metrics.into_iter().map(|(k, v)| (k.into(), v)).collect(),
            });
        }
    }
}

#[derive(Default, Debug, Serialize)]
pub(crate) struct ExtractStats {
    pub source_read_seconds: f64,
    pub decoder_read_seconds: f64,
    pub file_create_seconds: f64,
    pub file_write_seconds: f64,
    pub progress_seconds: f64,
    pub file_bytes: u64,
    pub files: u64,
}
// Linux schedstat provides per-thread CPU and runnable-but-unscheduled time.
// Read twice per shard; no syscall is added to each decoded buffer.
pub(crate) fn thread_schedstat() -> [u64; 3] {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| {
            let v: Vec<_> = s
                .split_whitespace()
                .map(str::parse::<u64>)
                .collect::<std::result::Result<_, _>>()
                .ok()?;
            (v.len() == 3).then(|| [v[0], v[1], v[2]])
        })
        .unwrap_or_default()
}
