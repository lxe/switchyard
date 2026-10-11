//! Host diagnostics attach to provider-neutral transport events without legacy manifest emulation.
use crate::hydrate::diagnostics::HostSnapshot;
use serde::{Deserialize, Serialize, Serializer};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};
use switchyard_artifacts::transport::{TransferObservation, TransferObserver};
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransportSnapshot {
    pub transfers: Vec<TransferObservation>,
    pub omitted_transfers: usize,
    pub failure: Option<TransportFailureSnapshot>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportFailureSnapshot {
    pub transfer: TransferObservation,
    pub host: HostSnapshot,
    pub host_capture_complete: bool,
    pub host_capture_seconds: f64,
}
#[derive(Debug, Clone, Default)]
pub struct TransportDiagnostics(Arc<Mutex<Records>>);
#[derive(Debug, Default)]
struct Records {
    transfers: BTreeMap<(String, u64), TransferObservation>,
    omitted: usize,
    failure: Option<TransportFailureSnapshot>,
}
impl TransportDiagnostics {
    pub fn snapshot(&self) -> TransportSnapshot {
        let r = self.0.lock().unwrap();
        TransportSnapshot {
            transfers: r.transfers.values().cloned().collect(),
            omitted_transfers: r.omitted,
            failure: r.failure.clone(),
        }
    }
    pub(crate) fn observer(&self) -> TransferObserver {
        let diagnostics = self.clone();
        TransferObserver::new(move |observation| {
            let mut r = diagnostics.0.lock().unwrap();
            if observation.failure.is_some() && r.failure.is_none() {
                let began = Instant::now();
                let host = HostSnapshot::capture();
                r.failure = Some(TransportFailureSnapshot {
                    transfer: observation.clone(),
                    host,
                    host_capture_complete: true,
                    host_capture_seconds: began.elapsed().as_secs_f64(),
                });
            }
            let key = (observation.object.clone(), observation.offset);
            if r.transfers.contains_key(&key) || r.transfers.len() < 256 {
                r.transfers.insert(key, observation);
            } else if matches!(
                observation.boundary,
                switchyard_artifacts::transport::TransferBoundary::Queued
            ) {
                r.omitted += 1;
            }
        })
    }
}
impl Serialize for TransportDiagnostics {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.snapshot().serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for TransportDiagnostics {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let snapshot = TransportSnapshot::deserialize(deserializer)?;
        Ok(Self(Arc::new(Mutex::new(Records {
            transfers: snapshot
                .transfers
                .into_iter()
                .take(256)
                .map(|o| ((o.object.clone(), o.offset), o))
                .collect(),
            omitted: snapshot.omitted_transfers,
            failure: snapshot.failure,
        }))))
    }
}
