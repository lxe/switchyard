use super::*;

async fn spawn_descendant(ignore_term: bool) -> Process {
    let mut command = Command::new("sh");
    let mut random = [0_u8; 8];
    getrandom::fill(&mut random).unwrap();
    let marker =
        std::env::temp_dir().join(format!("switchyard-child-{:x}", u64::from_le_bytes(random)));
    command.env("TEST_MARKER", &marker).args([
        "-c",
        if ignore_term {
            "sh -c 'trap \"\" TERM; while :; do sleep 1; done' & echo $! > \"$TEST_MARKER\"; wait"
        } else {
            "sleep 60 & echo $! > \"$TEST_MARKER\"; wait"
        },
    ]);
    let process = Process::spawn(&mut command).unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while !marker.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        // The child has installed its TERM handler before the group receives a signal.
        if ignore_term {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    fs::remove_file(marker).unwrap();
    process
}

#[tokio::test]
async fn shutdown_confirms_descendants_are_no_longer_writers() {
    let mut process = spawn_descendant(false).await;
    process.stop(Duration::from_secs(2)).await.unwrap();
    assert!(!live_group(process.group).unwrap());
}

#[tokio::test]
async fn surviving_descendant_causes_export_failure_and_forced_cleanup() {
    let mut process = spawn_descendant(true).await;
    assert!(process.stop(Duration::from_millis(100)).await.is_err());
    let group = process.group;
    drop(process);
    tokio::time::timeout(Duration::from_secs(2), async {
        while live_group(group).unwrap() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn cancelled_engine_owner_kills_its_entire_process_group() {
    let process = spawn_descendant(true).await;
    let group = process.group;
    let task = tokio::spawn(async move {
        let _owned = process;
        std::future::pending::<()>().await;
    });
    task.abort();
    let _ = task.await;
    tokio::time::timeout(Duration::from_secs(2), async {
        while live_group(group).unwrap() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}
