use super::*;
use crate::{Chunk, Manifest};

fn diagnostic() -> ModelDiagnostics {
    let manifest = Manifest {
        format: "shards-v1".into(),
        compress: "none".into(),
        sha256: "a".repeat(64),
        total_size: 1,
        raw_size: 1,
        files: BTreeMap::new(),
        chunks: vec![Chunk {
            name: "first-piece".into(),
            shard: 0,
            seq: 0,
            size: 1,
            raw: 1,
            sha256: "b".repeat(64),
        }],
    };
    ModelDiagnostics::new(&manifest, &Arc::new(Semaphore::new(1)), Instant::now(), 4)
}

#[test]
fn io_errno_and_kind_are_retained_without_payload_or_unknown_error_text() {
    let timeout_errno = nix::errno::Errno::ETIMEDOUT as i32;
    let (causes, truncated) = error_causes(&io::Error::from_raw_os_error(timeout_errno));
    assert!(!truncated);
    assert!(
        matches!(&causes[0], ErrorCause::Io { kind, raw_os_error: Some(errno) }
            if kind == "TimedOut" && *errno == timeout_errno)
    );
    let error = io::Error::new(
        io::ErrorKind::ConnectionReset,
        "https://private.example/?key=secret vendor error body",
    );
    let (causes, _) = error_causes(&error);
    assert!(
        matches!(&causes[0], ErrorCause::Io { kind, raw_os_error: None } if kind == "ConnectionReset")
    );
    let encoded = serde_json::to_string(&causes).unwrap();
    assert!(!encoded.contains("secret"));
    assert!(!encoded.contains("vendor"));
    assert!(!encoded.contains("private.example"));
}

#[test]
fn opaque_recursive_error_chain_is_bounded_without_formatting() {
    #[derive(Debug)]
    struct Recursive;
    impl std::fmt::Display for Recursive {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            panic!("error text must never be formatted")
        }
    }
    impl std::error::Error for Recursive {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(self)
        }
    }
    let (causes, truncated) = error_causes(&Recursive);
    assert_eq!(causes.len(), MAX_ERROR_CAUSES);
    assert!(truncated);
    assert!(
        causes
            .iter()
            .all(|cause| matches!(cause, ErrorCause::Opaque))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_during_capture_cannot_leave_placeholder_or_replace_first_failure() {
    let observed = diagnostic();
    let recording = observed.clone();
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, waiting) = std::sync::mpsc::channel();
    let task = tokio::spawn(async move {
        recording.failure_with_capture("first-piece", "response_body", || {
            let _ = entered.send(());
            waiting.recv().unwrap();
            HostSnapshot::capture()
        });
        std::future::pending::<()>().await;
    });
    started.await.unwrap();
    task.abort();
    release.send(()).unwrap();
    assert!(task.await.unwrap_err().is_cancelled());
    observed.failure("later-piece", "consumer_closed");
    let failure = observed.snapshot().failure.unwrap();
    assert_eq!(failure.piece, "first-piece");
    assert_eq!(failure.error_class, "response_body");
    assert!(failure.host_capture_complete);
    assert!(!failure.host.process.is_empty() || !failure.host.unavailable.is_empty());
}

#[test]
fn old_failure_observations_still_deserialize_with_unconfirmed_capture() {
    let diagnostic = diagnostic();
    diagnostic.failure("first-piece", "request_timeout");
    let mut legacy = serde_json::to_value(&diagnostic).unwrap();
    let failure = legacy["failure"].as_object_mut().unwrap();
    failure.remove("host_capture_complete");
    failure.remove("host_capture_seconds");
    for transfer in failure["transfers"].as_array_mut().unwrap() {
        transfer.as_object_mut().unwrap().remove("transport_error");
    }
    let restored: ModelDiagnostics = serde_json::from_value(legacy).unwrap();
    assert!(!restored.snapshot().failure.unwrap().host_capture_complete);
}
