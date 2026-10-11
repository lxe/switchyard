//! Weight reception selection and bounded metadata extraction; no cloud APIs.
use crate::{BridgeProgress, ChunkBridge, WeightBridge};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    time::{Duration, Instant},
};
use switchyard_artifacts::{
    Entry, Error, Result,
    extents::{
        ExtentHydration, ExtentManifest, ExtentOptions, ExtentProgress, ExtentReport, ExtentTicket,
    },
    range_stream::{BufferLimits, RangeObserver, RangeStream},
    tensors::{ATTRIBUTE, TensorManifest},
};
pub enum WeightReception {
    Persistent(ExtentHydration),
    Bounded {
        receiver: RangeStream,
        ticket: ExtentTicket,
        destinations: TensorManifest,
        limits: BufferLimits,
        timeout: Duration,
    },
}
impl WeightReception {
    pub fn prepare(
        ticket: ExtentTicket,
        destination: &Path,
        options: ExtentOptions,
        limits: BufferLimits,
    ) -> Result<Self> {
        let Some(index) = ticket.manifest.metadata.attributes.get(ATTRIBUTE) else {
            if ticket
                .manifest
                .entries
                .iter()
                .any(|e| e.path() == "switchyard-final-state.json")
            {
                return Err(Error::Invalid(
                    "final serving weights require baked chunk destinations".into(),
                ));
            }
            return Ok(Self::Persistent(ExtentHydration::prepare(
                ticket,
                destination,
                options,
            )?));
        };
        let destinations: TensorManifest = serde_json::from_str(index)?;
        destinations.validate(&ticket.manifest.entries)?;
        limits.validate()?;
        let observer = RangeObserver::new(|event| {
            let clock = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
            println!(
                "SWITCHYARD_WEIGHT_STREAM {}",
                serde_json::json!({"event":"buffer_progress","monotonic_s":clock.tv_sec as f64+clock.tv_nsec as f64/1e9,"progress":event})
            );
        });
        let receiver = RangeStream::new(
            destination.parent().unwrap_or(Path::new(".")),
            limits.clone(),
            Some(observer),
        )?;
        Ok(Self::Bounded {
            receiver,
            ticket,
            destinations,
            limits,
            timeout: options.request_timeout,
        })
    }
    pub fn manifest(&self) -> &ExtentManifest {
        match self {
            Self::Persistent(h) => h.manifest(),
            Self::Bounded { ticket, .. } => &ticket.manifest,
        }
    }
    pub fn progress(&self) -> ExtentProgress {
        match self {
            Self::Persistent(h) => h.progress(),
            Self::Bounded { receiver, .. } => {
                let p = receiver.progress();
                ExtentProgress {
                    received_bytes: p.received_bytes,
                    completed_bytes: p.reclaimed_bytes,
                    completed_extents: 0,
                }
            }
        }
    }
    pub fn allocated_output_bytes(&self) -> Result<u64> {
        match self {
            Self::Persistent(h) => h.allocated_output_bytes(),
            Self::Bounded { receiver, .. } => Ok(receiver.progress().spool_bytes),
        }
    }
    pub async fn run(&self) -> Result<ExtentReport> {
        match self {
            Self::Persistent(h) => h.run().await,
            Self::Bounded {
                receiver,
                ticket,
                limits,
                timeout,
                ..
            } => {
                let metadata = ticket
                    .manifest
                    .entries
                    .iter()
                    .enumerate()
                    .filter_map(|(i, e)| (!e.path().ends_with(".safetensors")).then_some(i))
                    .collect::<Vec<_>>();
                receiver
                    .run_prioritized(ticket, *timeout, &metadata)
                    .await?;
                Ok(ExtentReport {
                    downloaded_bytes: receiver.progress().received_bytes,
                    completed_extents: ticket.manifest.extents.len(),
                    file_bytes: 0,
                    copy_buffer_bound: limits.download_concurrency * limits.download_buffer_bytes,
                })
            }
        }
    }
    pub async fn metadata(&self, root: &Path) -> Result<()> {
        let Self::Bounded {
            receiver, ticket, ..
        } = self
        else {
            return Err(Error::Invalid(
                "bounded metadata called on persistent reception".into(),
            ));
        };
        let mut files = BTreeMap::new();
        let mut expected = BTreeMap::new();
        for (i, e) in ticket.manifest.entries.iter().enumerate() {
            match e {
                Entry::Directory { path, .. } => fs::create_dir_all(root.join(path))?,
                Entry::File { path, size, .. } if !path.ends_with(".safetensors") => {
                    files.insert(
                        i,
                        fs::File::options()
                            .read(true)
                            .write(true)
                            .create_new(true)
                            .open(root.join(path))?,
                    );
                    expected.insert(i, *size);
                }
                _ => {}
            }
        }
        while let Some(chunk) = receiver
            .next_matching(|s| files.contains_key(&s.entry))
            .await?
        {
            let file = files[&chunk.spec.entry].try_clone()?;
            let entry = chunk.spec.entry;
            let count = chunk.spec.length;
            tokio::task::spawn_blocking(move || -> Result<()> {
                let mut bytes = vec![0; chunk.spec.length as usize];
                chunk.copying();
                chunk.read_into(&mut bytes)?;
                use std::os::unix::fs::FileExt;
                file.write_all_at(&bytes, chunk.spec.offset)?;
                chunk.consumed();
                Ok(())
            })
            .await
            .map_err(|_| Error::Worker)??;
            let remaining = expected.get_mut(&entry).unwrap();
            *remaining = remaining
                .checked_sub(count)
                .ok_or_else(|| Error::Invalid("metadata chunk overrun".into()))?;
            if expected.values().all(|n| *n == 0) {
                return Ok(());
            }
        }
        if expected.values().any(|n| *n != 0) {
            return Err(Error::Integrity("metadata chunks incomplete".into()));
        }
        Ok(())
    }
}
pub enum DeliveryBridge {
    Whole(WeightBridge),
    Chunks(ChunkBridge),
}
impl DeliveryBridge {
    pub fn bind(path: &Path, reception: &WeightReception) -> Result<Self> {
        Ok(match reception {
            WeightReception::Persistent(_) => Self::Whole(WeightBridge::bind(path)?),
            _ => Self::Chunks(ChunkBridge::bind(path)?),
        })
    }
    pub fn path(&self) -> &Path {
        match self {
            Self::Whole(b) => b.path(),
            Self::Chunks(b) => b.path(),
        }
    }
    pub async fn serve(
        &self,
        reception: &WeightReception,
        epoch: Instant,
        on_connected: impl FnOnce(),
        on_progress: impl Fn(BridgeProgress),
    ) -> Result<BridgeProgress> {
        match (self, reception) {
            (Self::Whole(b), WeightReception::Persistent(h)) => {
                b.serve(h, epoch, on_connected, on_progress).await
            }
            (
                Self::Chunks(b),
                WeightReception::Bounded {
                    receiver,
                    destinations,
                    limits,
                    ..
                },
            ) => {
                b.serve(receiver, destinations, limits, epoch, on_progress)
                    .await
            }
            _ => Err(Error::Invalid(
                "weight delivery/reception protocol mismatch".into(),
            )),
        }
    }
}
