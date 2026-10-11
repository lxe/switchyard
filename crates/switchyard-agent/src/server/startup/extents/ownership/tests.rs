use super::*;
struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> (Scratch, String, switchyard_weights::VllmLoaderConfig) {
    let mut nonce = [0; 8];
    getrandom::fill(&mut nonce).unwrap();
    let root =
        Scratch(std::env::temp_dir().join(format!("loader-lease-{:x}", u64::from_le_bytes(nonce))));
    fs::create_dir(&root.0).unwrap();
    (root, "a".repeat(64), Default::default())
}
fn abandoned(root: &Path, content: &str, loader: &switchyard_weights::VllmLoaderConfig) -> PathBuf {
    let owner = sidecar(root, content, loader).unwrap();
    let path = owner.root.clone();
    let record = fs::read(record_path(&path)).unwrap();
    drop(owner);
    fs::create_dir(&path).unwrap();
    fs::write(record_path(&path), record).unwrap();
    fs::write(path.join("old-metadata"), "stale").unwrap();
    path
}
#[test]
fn abandoned_paths_are_reclaimed_with_full_identity_and_no_live_group() {
    let (root, content, loader) = fixture();
    let path = abandoned(&root.0, &content, &loader);
    let owner = sidecar(&root.0, &content, &loader).unwrap();
    assert_eq!(owner.root, path);
    assert!(!path.join("old-metadata").exists());
}
#[test]
fn unknown_foreign_namespace_and_mismatched_identity_are_preserved() {
    let (root, content, loader) = fixture();
    let path = abandoned(&root.0, &content, &loader);
    let original = record(&path).unwrap();
    for bytes in [
        b"{}".to_vec(),
        serde_json::to_vec(&Record {
            namespace: "pid:[1]".into(),
            ..context(original.identity.clone()).unwrap()
        })
        .unwrap(),
        serde_json::to_vec(&context("different-full-identity".into()).unwrap()).unwrap(),
        vec![b' '; 4097],
    ] {
        fs::write(record_path(&path), bytes).unwrap();
        assert!(sidecar(&root.0, &content, &loader).is_err());
        assert!(path.join("old-metadata").exists());
    }
}
#[tokio::test]
async fn surviving_process_group_prevents_reclaim_even_without_inherited_lease() {
    let (root, content, loader) = fixture();
    let path = abandoned(&root.0, &content, &loader);
    let mut command = Command::new("sh");
    command.args(["-c", "sleep 30"]);
    let mut process = engine::Process::spawn(&mut command).unwrap();
    let mut previous = record(&path).unwrap();
    previous.group = Some(process.child.id().unwrap() as i32);
    fs::write(record_path(&path), serde_json::to_vec(&previous).unwrap()).unwrap();
    assert!(sidecar(&root.0, &content, &loader).is_err());
    assert!(path.join("old-metadata").exists());
    process.stop(Duration::from_secs(2)).await.unwrap();
    let owner = sidecar(&root.0, &content, &loader).unwrap();
    assert!(!owner.root.join("old-metadata").exists());
}
#[tokio::test]
async fn inherited_engine_lease_and_independent_python_lease_protect_files() {
    let (root, content, loader) = fixture();
    let owner = sidecar(&root.0, &content, &loader).unwrap();
    let mut command = Command::new("python");
    command.args(["-c", "import os,fcntl,time; f=open(os.environ['SWITCHYARD_LOADER_LEASE'],'rb'); fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB); print('leased',flush=True); time.sleep(30)"])
        .stdout(Stdio::piped());
    let mut process = owner.spawn(&mut command).unwrap();
    let mut output = tokio::io::BufReader::new(process.child.stdout.take().unwrap());
    let mut line = String::new();
    use tokio::io::AsyncBufReadExt;
    tokio::time::timeout(Duration::from_secs(5), output.read_line(&mut line))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(line.trim(), "leased");
    let probe = fs::File::open(&owner.lock_path).unwrap();
    assert!(probe.try_lock().is_err());
    assert!(sidecar(&root.0, &content, &loader).is_err());
    process.stop(Duration::from_secs(2)).await.unwrap();
    drop(process);
    drop(owner);
    let _next = sidecar(&root.0, &content, &loader).unwrap();
}

#[test]
fn lock_conversion_gap_and_reused_creator_pid_preserve_the_directory() {
    let (root, content, loader) = fixture();
    let path = abandoned(&root.0, &content, &loader);
    let mut previous = record(&path).unwrap();
    previous.lease_ready = false;
    fs::write(record_path(&path), serde_json::to_vec(&previous).unwrap()).unwrap();
    assert!(sidecar(&root.0, &content, &loader).is_err());
    previous.owner_start += 1;
    fs::write(record_path(&path), serde_json::to_vec(&previous).unwrap()).unwrap();
    assert!(sidecar(&root.0, &content, &loader).is_err());
    assert!(path.join("old-metadata").exists());
}

#[test]
fn interrupted_record_update_keeps_previous_complete_ownership_proof() {
    let (root, content, loader) = fixture();
    let path = abandoned(&root.0, &content, &loader);
    fs::write(
        record_path(&path).with_extension("owner.pending"),
        b"{\"group\":",
    )
    .unwrap();
    let owner = sidecar(&root.0, &content, &loader).unwrap();
    assert_eq!(owner.root, path);
    assert!(!record_path(&path).with_extension("owner.pending").exists());
    assert!(!path.join("old-metadata").exists());
}

#[test]
fn malformed_boot_identity_never_proves_that_an_owner_is_gone() {
    let (root, content, loader) = fixture();
    let path = abandoned(&root.0, &content, &loader);
    let mut previous = record(&path).unwrap();
    previous.boot = "unknown-boot".into();
    fs::write(record_path(&path), serde_json::to_vec(&previous).unwrap()).unwrap();
    assert!(sidecar(&root.0, &content, &loader).is_err());
    assert!(path.join("old-metadata").exists());
}
