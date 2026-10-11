//! Explicit opt-in replay using real tickets supplied on stdin, never included in receipts.
use super::*;
use crate::{State, Ticket};
use std::{path::PathBuf, time::Duration};
use tokio::sync::RwLock;

#[derive(Deserialize)]
struct Input {
    ticket: Ticket,
    runtime: Option<switchyard_artifacts::ArtifactTicket>,
    model_root: PathBuf,
    runtime_root: PathBuf,
    receipt: PathBuf,
}

#[tokio::test]
#[ignore = "requires an explicit ticket on stdin and isolated empty destinations; downloads real artifacts"]
async fn production_pipeline_without_retries() {
    let input: Input = serde_json::from_reader(std::io::stdin()).expect("diagnostic input schema");
    assert_eq!(
        input.ticket.startup.mode,
        switchyard_artifacts::HydrationMode::TrustedStreaming
    );
    assert!(
        !input.model_root.join("model").exists(),
        "diagnostic destination must be fresh"
    );
    assert!(
        !input.runtime_root.exists(),
        "runtime destination must be fresh"
    );
    let state = Arc::new(RwLock::new(State::new(&input.ticket)));
    crate::hydrate::preflight(&input.ticket, &input.model_root, &state)
        .await
        .unwrap();
    let limit = Arc::new(Semaphore::new(input.ticket.concurrency));
    let started = Instant::now();
    let receipt = Arc::new(Mutex::new(fs::File::create(&input.receipt).unwrap()));
    let sample_state = state.clone();
    let sample_receipt = receipt.clone();
    let sampling = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(250));
        loop {
            interval.tick().await;
            let state = sample_state.read().await.clone();
            let row = serde_json::json!({"at_s": started.elapsed().as_secs_f64(), "state": state});
            let mut file = sample_receipt.lock().unwrap();
            serde_json::to_writer(&mut *file, &row).unwrap();
            writeln!(file).unwrap();
        }
    });
    let model = crate::hydrate::hydrate_with_limit(
        &input.ticket,
        &input.model_root,
        state.clone(),
        limit.clone(),
    );
    let runtime = async {
        if let Some(ticket) = input.runtime {
            switchyard_artifacts::hydrate(
                ticket,
                &input.runtime_root,
                switchyard_artifacts::HydrationOptions {
                    mode: switchyard_artifacts::HydrationMode::TrustedStreaming,
                    download_limit: Some(limit.clone()),
                    download_concurrency: input.ticket.concurrency,
                    extraction_concurrency: input.ticket.concurrency.min(20),
                    retries: 0,
                    stall_timeout_s: input.ticket.stall_s,
                    request_timeout_s: input.ticket.lifetime_s.clamp(1, 600),
                    cache_dir: input.runtime_root.with_extension("chunks"),
                    ..Default::default()
                },
                |_| {},
            )
            .await?;
        }
        Ok::<_, crate::Error>(())
    };
    let result = tokio::time::timeout(Duration::from_secs(600), async {
        tokio::try_join!(runtime, model).map(|_| ())
    })
    .await;
    sampling.abort();
    let _ = sampling.await;
    let error = match &result {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(error.to_string()),
        Err(_) => Some("diagnostic run deadline".into()),
    };
    let state = state.read().await.clone();
    let final_row = serde_json::json!({"complete": matches!(result, Ok(Ok(()))), "at_s": started.elapsed().as_secs_f64(), "error": error, "automatic_retries": 0, "state": state});
    let mut file = receipt.lock().unwrap();
    serde_json::to_writer(&mut *file, &final_row).unwrap();
    writeln!(file).unwrap();
    assert!(
        matches!(result, Ok(Ok(()))),
        "diagnostic receipt contains the failure"
    );
}
