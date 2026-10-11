//! Live counters are sampled only when state is serialized; no polling task or event log.
use crate::Manifest;
use serde::{Deserialize, Serialize, Serializer};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Read, Write},
    sync::{Arc, Mutex, Weak},
    time::Instant,
};
use tokio::sync::Semaphore;

const MAX_TRANSFERS: usize = 256;
const MAX_SHARDS: usize = 64;
const MAX_ERROR_CAUSES: usize = 16;

/// Flags and typed causes only: Display/Debug strings may contain capabilities or bodies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReqwestErrorFlags {
    pub timeout: bool,
    pub connect: bool,
    pub dns: bool,
    pub request: bool,
    pub body: bool,
    pub decode: bool,
    pub builder: bool,
    pub status: bool,
    pub redirect: bool,
    pub upgrade: bool,
}
impl From<&reqwest::Error> for ReqwestErrorFlags {
    fn from(error: &reqwest::Error) -> Self {
        Self {
            timeout: error.is_timeout(),
            connect: error.is_connect(),
            dns: error.is_dns(),
            request: error.is_request(),
            body: error.is_body(),
            decode: error.is_decode(),
            builder: error.is_builder(),
            status: error.is_status(),
            redirect: error.is_redirect(),
            upgrade: error.is_upgrade(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "category", rename_all = "snake_case")]
pub enum ErrorCause {
    Reqwest {
        flags: ReqwestErrorFlags,
    },
    Hyper {
        timeout: bool,
        closed: bool,
        canceled: bool,
        incomplete_message: bool,
        parse: bool,
        user: bool,
        body_write_aborted: bool,
    },
    Io {
        kind: String,
        raw_os_error: Option<i32>,
    },
    /// Unknown types are intentionally not formatted or named.
    Opaque,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportErrorDetails {
    pub reqwest_flags: ReqwestErrorFlags,
    pub causes: Vec<ErrorCause>,
    pub causes_truncated: bool,
}
fn error_causes(error: &(dyn std::error::Error + 'static)) -> (Vec<ErrorCause>, bool) {
    let mut causes = Vec::new();
    let mut source = Some(error);
    while let Some(error) = source {
        if causes.len() == MAX_ERROR_CAUSES {
            return (causes, true);
        }
        causes.push(
            if let Some(error) = error.downcast_ref::<reqwest::Error>() {
                ErrorCause::Reqwest {
                    flags: error.into(),
                }
            } else if let Some(error) = error.downcast_ref::<hyper::Error>() {
                ErrorCause::Hyper {
                    timeout: error.is_timeout(),
                    closed: error.is_closed(),
                    canceled: error.is_canceled(),
                    incomplete_message: error.is_incomplete_message(),
                    parse: error.is_parse(),
                    user: error.is_user(),
                    body_write_aborted: error.is_body_write_aborted(),
                }
            } else if let Some(error) = error.downcast_ref::<io::Error>() {
                ErrorCause::Io {
                    kind: format!("{:?}", error.kind()),
                    raw_os_error: error.raw_os_error(),
                }
            } else {
                ErrorCause::Opaque
            },
        );
        source = error.source();
    }
    (causes, false)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Boundary {
    #[default]
    Pending,
    PermitWait,
    HeadersWait,
    BodyWait,
    ProgressState,
    ChannelSend,
    BodyComplete,
    DecoderCompletion,
    Decoding,
    ChannelReceive,
    Archive,
    PieceRead,
    FileWrite,
    ExtractionError,
    WriteError,
    ReadError,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransferObservation {
    pub name: String,
    pub shard: usize,
    pub seq: usize,
    pub expected_bytes: u64,
    pub boundary: Boundary,
    pub boundary_since_s: f64,
    pub boundary_age_s: f64,
    pub consumer_boundary: Boundary,
    pub consumer_since_s: f64,
    pub consumer_age_s: f64,
    pub permit_wait_started_s: Option<f64>,
    pub permit_acquired_s: Option<f64>,
    pub permit_released_s: Option<f64>,
    pub permit_wait_s: f64,
    pub permit_held_s: f64,
    pub request_s: Option<f64>,
    pub headers_s: Option<f64>,
    pub first_body_s: Option<f64>,
    pub last_body_s: Option<f64>,
    pub body_eof_s: Option<f64>,
    pub consumed_s: Option<f64>,
    pub http_version: Option<String>,
    pub http_status: Option<u16>,
    pub declared_length: Option<u64>,
    pub remote_socket: Option<String>,
    pub received_bytes: u64,
    pub delivered_bytes: u64,
    pub consumed_bytes: u64,
    pub decoded_bytes: u64,
    /// Most recent channel sample; occupancy can change between samples.
    pub channel_available_slots: usize,
    pub channel_slots: usize,
    pub channel_wait_s: f64,
    pub body_wait_s: f64,
    pub longest_body_wait_s: f64,
    pub error_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport_error: Option<TransportErrorDetails>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardObservation {
    pub shard: usize,
    pub active_seq: Option<usize>,
    pub boundary: Boundary,
    pub boundary_since_s: f64,
    pub boundary_age_s: f64,
    pub decoded_bytes: u64,
    pub file_written_bytes: u64,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub elapsed_s: f64,
    pub available_permits: Option<usize>,
    pub transfers: Vec<TransferObservation>,
    pub shards: Vec<ShardObservation>,
    pub omitted_transfers: usize,
    pub omitted_shards: usize,
    pub failure: Option<FailureSnapshot>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSnapshot {
    pub elapsed_s: f64,
    pub piece: String,
    pub error_class: String,
    pub transfers: Vec<TransferObservation>,
    pub shards: Vec<ShardObservation>,
    pub host: HostSnapshot,
    /// False only for older observations without cancellation-safe capture.
    #[serde(default)]
    pub host_capture_complete: bool,
    #[serde(default)]
    pub host_capture_seconds: f64,
}
/// Deserialized observations keep their snapshot. Locally created observations sample live data.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelDiagnostics {
    #[serde(flatten)]
    snapshot: ModelSnapshot,
    #[serde(skip)]
    live: Option<Arc<Registry>>,
}
impl Serialize for ModelDiagnostics {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.snapshot().serialize(serializer)
    }
}
#[derive(Debug)]
struct Registry {
    started: Instant,
    limit: Weak<Semaphore>,
    transfers: Vec<TransferTrace>,
    shards: Vec<ShardTrace>,
    omitted_transfers: usize,
    omitted_shards: usize,
    failure: Mutex<Option<FailureSnapshot>>,
}
impl ModelDiagnostics {
    pub(crate) fn new(
        manifest: &Manifest,
        limit: &Arc<Semaphore>,
        started: Instant,
        slots: usize,
    ) -> Self {
        let transfers = manifest
            .chunks
            .iter()
            .take(MAX_TRANSFERS)
            .map(|c| TransferTrace {
                started,
                inner: Some(Arc::new(Mutex::new(TransferObservation {
                    name: c.name.clone(),
                    shard: c.shard,
                    seq: c.seq,
                    expected_bytes: c.size,
                    boundary: Boundary::Pending,
                    consumer_boundary: Boundary::Pending,
                    channel_slots: slots,
                    channel_available_slots: slots,
                    ..Default::default()
                }))),
            })
            .collect::<Vec<_>>();
        let ids = manifest
            .chunks
            .iter()
            .map(|c| c.shard)
            .collect::<std::collections::BTreeSet<_>>();
        let shards = ids
            .iter()
            .take(MAX_SHARDS)
            .map(|&shard| ShardTrace {
                started,
                inner: Some(Arc::new(Mutex::new(ShardObservation {
                    shard,
                    boundary: Boundary::Pending,
                    ..Default::default()
                }))),
            })
            .collect::<Vec<_>>();
        Self {
            snapshot: ModelSnapshot::default(),
            live: Some(Arc::new(Registry {
                started,
                limit: Arc::downgrade(limit),
                omitted_transfers: manifest.chunks.len().saturating_sub(transfers.len()),
                omitted_shards: ids.len().saturating_sub(shards.len()),
                transfers,
                shards,
                failure: Mutex::new(None),
            })),
        }
    }
    pub fn snapshot(&self) -> ModelSnapshot {
        let Some(r) = &self.live else {
            return self.snapshot.clone();
        };
        let elapsed_s = r.started.elapsed().as_secs_f64();
        ModelSnapshot {
            elapsed_s,
            available_permits: r.limit.upgrade().map(|l| l.available_permits()),
            transfers: r
                .transfers
                .iter()
                .filter_map(|t| t.snapshot(elapsed_s))
                .collect(),
            shards: r
                .shards
                .iter()
                .filter_map(|s| s.snapshot(elapsed_s))
                .collect(),
            omitted_transfers: r.omitted_transfers,
            omitted_shards: r.omitted_shards,
            failure: r.failure.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    }
    pub(crate) fn transfer(&self, name: &str) -> TransferTrace {
        let r = self.live.as_ref().unwrap();
        r.transfers
            .iter()
            .find(|t| {
                t.inner
                    .as_ref()
                    .unwrap()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .name
                    == name
            })
            .cloned()
            .unwrap_or(TransferTrace {
                started: r.started,
                inner: None,
            })
    }
    pub(crate) fn shard(&self, id: usize) -> ShardTrace {
        let r = self.live.as_ref().unwrap();
        r.shards
            .iter()
            .find(|t| {
                t.inner
                    .as_ref()
                    .unwrap()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .shard
                    == id
            })
            .cloned()
            .unwrap_or(ShardTrace {
                started: r.started,
                inner: None,
            })
    }
    pub(crate) fn failure(&self, piece: &str, error_class: &str) {
        self.failure_with_capture(piece, error_class, HostSnapshot::capture);
    }
    fn failure_with_capture(
        &self,
        piece: &str,
        error_class: &str,
        capture: impl FnOnce() -> HostSnapshot,
    ) {
        let Some(r) = &self.live else { return };
        // The first record is published only after capture completes, without an await.
        // Callers still own the sender/response here, so decoder EOF cannot cancel capture.
        let mut slot = r.failure.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return;
        }
        let elapsed_s = r.started.elapsed().as_secs_f64();
        let mut failure = FailureSnapshot {
            elapsed_s,
            piece: piece.into(),
            error_class: error_class.into(),
            transfers: r
                .transfers
                .iter()
                .filter_map(|t| t.snapshot(elapsed_s))
                .collect(),
            shards: r
                .shards
                .iter()
                .filter_map(|s| s.snapshot(elapsed_s))
                .collect(),
            host: HostSnapshot::default(),
            host_capture_complete: false,
            host_capture_seconds: 0.0,
        };
        let started = Instant::now();
        failure.host = capture();
        failure.host_capture_seconds = started.elapsed().as_secs_f64();
        failure.host_capture_complete = true;
        *slot = Some(failure);
    }
}
#[derive(Debug, Clone)]
pub(crate) struct TransferTrace {
    started: Instant,
    inner: Option<Arc<Mutex<TransferObservation>>>,
}
impl TransferTrace {
    pub(crate) fn origin(&self) -> Instant {
        self.started
    }
    pub(crate) fn error(&self, class: &str) {
        self.update(|t, _| t.error_class = Some(class.into()));
    }
    pub(crate) fn error_class(&self) -> Option<String> {
        self.inner
            .as_ref()?
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .error_class
            .clone()
    }
    pub(crate) fn request_error(&self, error: &reqwest::Error) -> &'static str {
        let class = if error.is_timeout() {
            "reqwest_timeout"
        } else if error.is_connect() {
            "connect"
        } else if error.is_body() {
            "response_body"
        } else if error.is_decode() {
            "response_decode"
        } else {
            "transport"
        };
        let (causes, causes_truncated) = error_causes(error);
        self.update(|t, _| {
            t.error_class = Some(class.into());
            t.transport_error = Some(TransportErrorDetails {
                reqwest_flags: error.into(),
                causes,
                causes_truncated,
            });
        });
        class
    }
    pub(crate) fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }
    pub(crate) fn update(&self, f: impl FnOnce(&mut TransferObservation, f64)) {
        if let Some(inner) = &self.inner {
            f(
                &mut inner.lock().unwrap_or_else(|e| e.into_inner()),
                self.now(),
            );
        }
    }
    pub(crate) fn boundary(&self, name: Boundary) {
        self.update(|t, now| {
            t.boundary = name;
            t.boundary_since_s = now;
        });
    }
    pub(crate) fn consumer(&self, name: Boundary) {
        self.update(|t, now| {
            t.consumer_boundary = name;
            t.consumer_since_s = now;
        });
    }
    fn snapshot(&self, now: f64) -> Option<TransferObservation> {
        let mut t = self
            .inner
            .as_ref()?
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        t.boundary_age_s = (now - t.boundary_since_s).max(0.0);
        t.consumer_age_s = (now - t.consumer_since_s).max(0.0);
        if let Some(start) = t.permit_wait_started_s {
            t.permit_wait_s = (t.permit_acquired_s.unwrap_or(now) - start).max(0.0);
        }
        if let Some(start) = t.permit_acquired_s {
            t.permit_held_s = (t.permit_released_s.unwrap_or(now) - start).max(0.0);
        }
        Some(t)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct ShardTrace {
    started: Instant,
    inner: Option<Arc<Mutex<ShardObservation>>>,
}
impl ShardTrace {
    pub(crate) fn update(&self, f: impl FnOnce(&mut ShardObservation, f64)) {
        if let Some(inner) = &self.inner {
            f(
                &mut inner.lock().unwrap_or_else(|e| e.into_inner()),
                self.started.elapsed().as_secs_f64(),
            );
        }
    }
    pub(crate) fn boundary(&self, name: Boundary) {
        self.update(|s, now| {
            s.boundary = name;
            s.boundary_since_s = now;
        });
    }
    fn snapshot(&self, now: f64) -> Option<ShardObservation> {
        let mut s = self
            .inner
            .as_ref()?
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        s.boundary_age_s = (now - s.boundary_since_s).max(0.0);
        Some(s)
    }
}
pub(super) struct ObservedWriter<'a, W> {
    pub inner: W,
    pub trace: &'a ShardTrace,
}
impl<W: Write> Write for ObservedWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.trace.boundary(Boundary::FileWrite);
        let result = self.inner.write(bytes);
        self.trace.update(|s, now| {
            if let Ok(n) = &result {
                s.file_written_bytes += *n as u64;
            }
            s.boundary = if result.is_ok() {
                Boundary::Archive
            } else {
                Boundary::WriteError
            };
            s.boundary_since_s = now;
        });
        result
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Fixed allowlists and caps exclude argv, environment, paths, headers and vendor error bodies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HostSnapshot {
    pub process: BTreeMap<String, u64>,
    pub memory_kib: BTreeMap<String, u64>,
    pub io: BTreeMap<String, u64>,
    pub tasks: Vec<TaskSnapshot>,
    pub tasks_truncated: bool,
    pub sockets: Vec<SocketSnapshot>,
    pub sockets_truncated: bool,
    pub unavailable: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSnapshot {
    pub tid: u32,
    pub state: String,
    pub user_ticks: u64,
    pub system_ticks: u64,
    pub wait_channel: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketSnapshot {
    pub inode: u64,
    pub local_socket_hex: String,
    pub remote_socket_hex: String,
    pub state_hex: String,
    pub tx_queue_bytes: u64,
    pub rx_queue_bytes: u64,
    pub timer_hex: String,
    pub retransmits: u64,
}
fn read_proc(path: &str) -> io::Result<String> {
    let mut value = String::new();
    fs::File::open(path)?
        .take(256 << 10)
        .read_to_string(&mut value)?;
    Ok(value)
}
fn numbers(path: &str, keys: &[&str]) -> BTreeMap<String, u64> {
    read_proc(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let (key, rest) = line.split_once(':')?;
            keys.contains(&key)
                .then(|| Some((key.into(), rest.split_whitespace().next()?.parse().ok()?)))?
        })
        .collect()
}
impl HostSnapshot {
    pub(crate) fn capture() -> Self {
        let mut s = Self {
            process: numbers(
                "/proc/self/status",
                &[
                    "VmRSS",
                    "VmSize",
                    "Threads",
                    "voluntary_ctxt_switches",
                    "nonvoluntary_ctxt_switches",
                ],
            ),
            memory_kib: numbers(
                "/proc/meminfo",
                &[
                    "MemAvailable",
                    "MemFree",
                    "Shmem",
                    "Dirty",
                    "Writeback",
                    "SwapFree",
                ],
            ),
            io: numbers(
                "/proc/self/io",
                &[
                    "rchar",
                    "wchar",
                    "syscr",
                    "syscw",
                    "read_bytes",
                    "write_bytes",
                    "cancelled_write_bytes",
                ],
            ),
            ..Self::default()
        };
        if s.process.is_empty() {
            s.unavailable.push("process_status".into());
        }
        if s.memory_kib.is_empty() {
            s.unavailable.push("memory".into());
        }
        if s.io.is_empty() {
            s.unavailable.push("process_io".into());
        }
        if let Ok(tasks) = fs::read_dir("/proc/self/task") {
            let mut ids = tasks
                .filter_map(|e| e.ok()?.file_name().to_str()?.parse::<u32>().ok())
                .collect::<Vec<_>>();
            ids.sort_unstable();
            s.tasks_truncated = ids.len() > 64;
            for tid in ids.into_iter().take(64) {
                let base = format!("/proc/self/task/{tid}");
                if let Ok(stat) = read_proc(&format!("{base}/stat"))
                    && let Some((_, values)) = stat.rsplit_once(") ")
                {
                    let f = values.split_whitespace().collect::<Vec<_>>();
                    if f.len() > 12 {
                        s.tasks.push(TaskSnapshot {
                            tid,
                            state: f[0].into(),
                            user_ticks: f[11].parse().unwrap_or(0),
                            system_ticks: f[12].parse().unwrap_or(0),
                            wait_channel: read_proc(&format!("{base}/wchan"))
                                .unwrap_or_default()
                                .trim()
                                .chars()
                                .take(80)
                                .collect(),
                        });
                    }
                }
            }
        } else {
            s.unavailable.push("tasks".into());
        }
        let inodes = fs::read_dir("/proc/self/fd")
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| {
                let target = fs::read_link(e.ok()?.path()).ok()?;
                let value = target.to_str()?;
                value
                    .strip_prefix("socket:[")?
                    .strip_suffix(']')?
                    .parse::<u64>()
                    .ok()
            })
            .collect::<std::collections::BTreeSet<_>>();
        for path in ["/proc/self/net/tcp", "/proc/self/net/tcp6"] {
            let Ok(table) = read_proc(path) else {
                s.unavailable.push(path.rsplit('/').next().unwrap().into());
                continue;
            };
            for line in table.lines().skip(1) {
                let f = line.split_whitespace().collect::<Vec<_>>();
                if f.len() < 10 {
                    continue;
                }
                let Ok(inode) = f[9].parse() else { continue };
                if !inodes.contains(&inode) {
                    continue;
                }
                if s.sockets.len() >= 128 {
                    s.sockets_truncated = true;
                    continue;
                }
                let Some((tx, rx)) = f[4].split_once(':') else {
                    continue;
                };
                s.sockets.push(SocketSnapshot {
                    inode,
                    local_socket_hex: f[1].into(),
                    remote_socket_hex: f[2].into(),
                    state_hex: f[3].into(),
                    tx_queue_bytes: u64::from_str_radix(tx, 16).unwrap_or(0),
                    rx_queue_bytes: u64::from_str_radix(rx, 16).unwrap_or(0),
                    timer_hex: f[5].into(),
                    retransmits: u64::from_str_radix(f[6], 16).unwrap_or(0),
                });
            }
        }
        s
    }
}

#[cfg(test)]
mod local;

#[cfg(test)]
mod tests;
