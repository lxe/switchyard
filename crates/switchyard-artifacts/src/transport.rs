//! Optional transport observations contain flags and typed causes, never URL/error text.
use serde::{Deserialize, Serialize};
use std::{
    error::Error as _,
    io,
    sync::{Arc, Mutex},
    time::Instant,
};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportFailure {
    pub stage: String,
    pub timeout: bool,
    pub connect: bool,
    pub body: bool,
    pub request: bool,
    pub decode: bool,
    pub status: Option<u16>,
    pub io_causes: Vec<IoCause>,
    pub hyper_causes: Vec<HyperCause>,
    pub causes_truncated: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperCause {
    pub timeout: bool,
    pub closed: bool,
    pub canceled: bool,
    pub incomplete_message: bool,
    pub parse: bool,
    pub user: bool,
    pub body_write_aborted: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoCause {
    pub kind: String,
    pub raw_os_error: Option<i32>,
}
impl TransportFailure {
    pub fn request(stage: &str, error: &reqwest::Error) -> Self {
        let mut failure = Self::local(stage);
        failure.timeout = error.is_timeout();
        failure.connect = error.is_connect();
        failure.body = error.is_body();
        failure.request = error.is_request();
        failure.decode = error.is_decode();
        failure.status = error.status().map(|s| s.as_u16());
        let mut cause = error.source();
        let mut count = 0;
        while let Some(error) = cause {
            if let Some(io) = error.downcast_ref::<io::Error>() {
                failure.io_causes.push(IoCause {
                    kind: format!("{:?}", io.kind()),
                    raw_os_error: io.raw_os_error(),
                });
            }
            if let Some(h) = error.downcast_ref::<hyper::Error>() {
                failure.hyper_causes.push(HyperCause {
                    timeout: h.is_timeout(),
                    closed: h.is_closed(),
                    canceled: h.is_canceled(),
                    incomplete_message: h.is_incomplete_message(),
                    parse: h.is_parse(),
                    user: h.is_user(),
                    body_write_aborted: h.is_body_write_aborted(),
                });
            }
            count += 1;
            if count >= 16 {
                failure.causes_truncated = error.source().is_some();
                break;
            }
            cause = error.source();
        }
        failure
    }
    pub fn io(stage: &str, error: &io::Error) -> Self {
        let mut f = Self::local(stage);
        f.io_causes.push(IoCause {
            kind: format!("{:?}", error.kind()),
            raw_os_error: error.raw_os_error(),
        });
        f
    }
    pub fn local(stage: &str) -> Self {
        Self {
            stage: stage.into(),
            timeout: false,
            connect: false,
            body: false,
            request: false,
            decode: false,
            status: None,
            io_causes: vec![],
            hyper_causes: vec![],
            causes_truncated: false,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferBoundary {
    Queued,
    Headers,
    Body,
    Storage,
    Complete,
    Failure,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferObservation {
    pub object: String,
    pub offset: u64,
    pub expected_bytes: u64,
    pub received_bytes: u64,
    pub written_bytes: u64,
    pub boundary: TransferBoundary,
    pub elapsed_seconds: f64,
    pub body_wait_seconds: f64,
    pub storage_write_seconds: f64,
    pub failure: Option<TransportFailure>,
}
#[derive(Clone)]
pub struct TransferObserver(Arc<dyn Fn(TransferObservation) + Send + Sync>);
impl std::fmt::Debug for TransferObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TransferObserver")
    }
}
impl TransferObserver {
    pub fn new(callback: impl Fn(TransferObservation) + Send + Sync + 'static) -> Self {
        Self(Arc::new(callback))
    }
}
#[derive(Clone)]
pub(crate) struct Trace {
    observer: Option<TransferObserver>,
    record: Arc<Mutex<TransferObservation>>,
    epoch: Instant,
}
impl Trace {
    pub(crate) fn new(
        object: String,
        offset: u64,
        expected_bytes: u64,
        observer: Option<TransferObserver>,
    ) -> Self {
        let trace = Self {
            observer,
            record: Arc::new(Mutex::new(TransferObservation {
                object,
                offset,
                expected_bytes,
                received_bytes: 0,
                written_bytes: 0,
                boundary: TransferBoundary::Queued,
                elapsed_seconds: 0.0,
                body_wait_seconds: 0.0,
                storage_write_seconds: 0.0,
                failure: None,
            })),
            epoch: Instant::now(),
        };
        trace.update(|_| {}, true);
        trace
    }
    fn update(&self, f: impl FnOnce(&mut TransferObservation), emit: bool) {
        let mut r = self.record.lock().unwrap();
        if r.failure.is_some() {
            return;
        }
        f(&mut r);
        r.elapsed_seconds = self.epoch.elapsed().as_secs_f64();
        let value = emit.then(|| r.clone());
        drop(r);
        if let (Some(observer), Some(value)) = (&self.observer, value) {
            (observer.0)(value);
        }
    }
    pub(crate) fn boundary(&self, b: TransferBoundary) {
        self.update(|r| r.boundary = b, true);
    }
    pub(crate) fn body(&self, bytes: u64, seconds: f64) {
        let previous = self.record.lock().unwrap().received_bytes;
        self.update(
            |r| {
                r.received_bytes += bytes;
                r.body_wait_seconds += seconds;
                r.boundary = TransferBoundary::Body;
            },
            previous >> 20 != (previous + bytes) >> 20,
        );
    }
    pub(crate) fn write(&self, bytes: u64, seconds: f64) {
        self.update(
            |r| {
                r.written_bytes += bytes;
                r.storage_write_seconds += seconds;
                r.boundary = TransferBoundary::Storage;
            },
            false,
        );
    }
    pub(crate) fn failure(&self, f: TransportFailure) {
        self.update(
            |r| {
                r.boundary = TransferBoundary::Failure;
                r.failure = Some(f);
            },
            true,
        );
    }
}
