//! Provider-neutral bounded transient range reception. No tensor interpretation.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::Notify;
mod buffer;
mod transport;
pub use buffer::ReadyRange;
const ALIGN: usize = 4096;

pub use crate::buffer_limits::BufferLimits;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Location {
    Ram,
    Spool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lifecycle {
    Reserved,
    Downloading,
    Ready,
    Copying,
    Consumed,
    Reclaimed,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BufferProgress {
    pub received_bytes: u64,
    pub ram_bytes: u64,
    pub spool_bytes: u64,
    pub peak_ram_bytes: u64,
    pub peak_spool_bytes: u64,
    pub reclaimed_bytes: u64,
    pub exhausted_buffer_seconds: f64,
    pub network_wait_seconds: f64,
    pub spool_write_seconds: f64,
    pub consumer_wait_seconds: f64,
    pub consumer_available: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeEvent {
    pub elapsed_s: f64,
    pub chunk_id: Option<usize>,
    pub lifecycle: Option<Lifecycle>,
    pub progress: BufferProgress,
}
#[derive(Clone)]
pub struct RangeObserver(Arc<dyn Fn(RangeEvent) + Send + Sync>);
impl RangeObserver {
    pub fn new(f: impl Fn(RangeEvent) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeSpec {
    pub id: usize,
    pub entry: usize,
    pub offset: u64,
    pub length: u64,
    pub padded_length: u64,
}
struct State {
    stats: BufferProgress,
    ready: VecDeque<ReadyRange>,
    finished: bool,
    failed: bool,
}
struct Shared {
    state: Mutex<State>,
    wake: Notify,
    cancelled: AtomicBool,
    started: AtomicBool,
    last_progress_ns: AtomicU64,
    directory: PathBuf,
    limits: BufferLimits,
    epoch: Instant,
    observer: Option<RangeObserver>,
}
impl Shared {
    fn emit(&self, id: Option<usize>, lifecycle: Option<Lifecycle>) {
        if lifecycle.is_none() {
            let now = self.epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let previous = self.last_progress_ns.load(Ordering::Relaxed);
            if now.saturating_sub(previous) < 100_000_000
                || self
                    .last_progress_ns
                    .compare_exchange(previous, now, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
            {
                return;
            }
        }
        if let Some(o) = &self.observer {
            let stats = self.state.lock().unwrap().stats.clone();
            (o.0)(RangeEvent {
                elapsed_s: self.epoch.elapsed().as_secs_f64(),
                chunk_id: id,
                lifecycle,
                progress: stats,
            });
        }
    }
    fn cancelled(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}
impl Drop for Shared {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
#[derive(Clone)]
pub struct RangeStream {
    shared: Arc<Shared>,
}
impl RangeStream {
    pub fn new(
        parent: &Path,
        limits: BufferLimits,
        observer: Option<RangeObserver>,
    ) -> Result<Self> {
        limits.validate()?;
        fs::create_dir_all(parent)?;
        let filesystem = nix::sys::statfs::statfs(parent)
            .map_err(std::io::Error::from)?
            .filesystem_type();
        if filesystem == nix::sys::statfs::TMPFS_MAGIC
            || filesystem == nix::sys::statfs::FsType(0x858458f6_u64 as _)
        {
            return Err(Error::Invalid(
                "transient overflow requires disk backing; tmpfs/ramfs are not NVMe spool".into(),
            ));
        }
        let mut nonce = [0; 16];
        getrandom::fill(&mut nonce).map_err(|_| Error::Worker)?;
        let name = nonce.iter().map(|n| format!("{n:02x}")).collect::<String>();
        let directory = parent.join(format!(".weight-buffer-{name}"));
        fs::create_dir(&directory)?;
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
        Ok(Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    stats: BufferProgress::default(),
                    ready: VecDeque::new(),
                    finished: false,
                    failed: false,
                }),
                wake: Notify::new(),
                cancelled: AtomicBool::new(false),
                started: AtomicBool::new(false),
                last_progress_ns: AtomicU64::new(0),
                directory,
                limits,
                epoch: Instant::now(),
                observer,
            }),
        })
    }
    pub fn progress(&self) -> BufferProgress {
        self.shared.state.lock().unwrap().stats.clone()
    }
    pub fn cancel(&self) {
        self.shared.cancelled.store(true, Ordering::Release);
        let ready = {
            let mut s = self.shared.state.lock().unwrap();
            s.failed = true;
            std::mem::take(&mut s.ready)
        };
        drop(ready);
        self.shared.wake.notify_waiters();
    }
    pub async fn next_ready(&self) -> Result<Option<ReadyRange>> {
        self.consumer_available();
        self.next_matching(|_| true).await
    }
    pub fn consumer_available(&self) {
        self.shared.state.lock().unwrap().stats.consumer_available = true;
        self.shared.emit(None, None);
    }
    pub async fn next_matching(
        &self,
        matches: impl Fn(&RangeSpec) -> bool,
    ) -> Result<Option<ReadyRange>> {
        let began = Instant::now();
        loop {
            let wake = self.shared.wake.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            self.shared.cancelled()?;
            {
                let mut st = self.shared.state.lock().unwrap();
                if st.failed {
                    return Err(Error::Transfer);
                }
                if let Some(i) = st
                    .ready
                    .iter()
                    .position(|r| matches(&r.spec) && matches!(r.location(), Location::Ram))
                    .or_else(|| st.ready.iter().position(|r| matches(&r.spec)))
                {
                    st.stats.consumer_wait_seconds += began.elapsed().as_secs_f64();
                    return Ok(st.ready.remove(i));
                }
                if st.finished {
                    return Ok(None);
                }
            }
            wake.await;
        }
    }
}
