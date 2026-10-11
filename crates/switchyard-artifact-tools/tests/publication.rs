//! Exercise the complete publication command while its source manifest changes.
use axum::{
    Router,
    body::Bytes,
    extract::{Path as ObjectPath, State},
    http::{HeaderMap, StatusCode},
    routing::put,
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use switchyard_artifact_tools::cli::{Command, StorageSettings, run};
use switchyard_artifacts::{Manifest, Metadata, PackOptions, pack};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let mut random = [0_u8; 12];
        getrandom::fill(&mut random).unwrap();
        let nonce = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let path = std::env::temp_dir().join(format!("switchyard-publication-test-{nonce}"));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct Uploads {
    source_manifest: PathBuf,
    replacement: Arc<Vec<u8>>,
    objects: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
}
async fn upload(
    State(state): State<Uploads>,
    ObjectPath(key): ObjectPath<String>,
    headers: HeaderMap,
    body: Bytes,
) -> StatusCode {
    assert_eq!(headers.get("if-none-match").unwrap(), "*");
    assert_eq!(
        headers
            .get("x-amz-content-sha256")
            .unwrap()
            .to_str()
            .unwrap(),
        format!("{:x}", Sha256::digest(&body)),
    );
    let mut objects = state.objects.lock().unwrap();
    if key.ends_with("/manifest.json") {
        assert!(!objects.is_empty(), "manifest must follow verified chunks");
    } else {
        // This happens inside the object store before the shard PUT completes.
        // Publication must use the original captured manifest, even though the local
        // input now contains a valid manifest with another content identity.
        fs::write(&state.source_manifest, state.replacement.as_ref()).unwrap();
    }
    objects.insert(key, body.to_vec());
    StatusCode::OK
}

#[tokio::test]
async fn publication_uses_captured_manifest_bytes_despite_source_change_during_upload() {
    let scratch = Scratch::new();
    let source = scratch.0.join("source");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("payload"), "artifact contents").unwrap();
    let directory = scratch.0.join("packed");
    let manifest = pack(
        &source,
        &directory,
        Metadata {
            kind: "publication-test".into(),
            compatibility: BTreeMap::new(),
            attributes: BTreeMap::new(),
        },
        PackOptions::default(),
    )
    .unwrap();
    let mut original = serde_json::to_vec_pretty(&manifest).unwrap();
    original.push(b'\n');
    let manifest_path = directory.join("manifest.json");
    fs::write(&manifest_path, &original).unwrap();
    let mut changed = manifest.clone();
    changed
        .metadata
        .attributes
        .insert("revision".into(), "concurrent-change".into());
    changed.id = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                &changed.format,
                &changed.metadata,
                &changed.entries,
                &changed.chunks,
            ))
            .unwrap()
        ),
    );
    changed.validate().unwrap();
    assert_ne!(changed.id, manifest.id);
    let replacement = serde_json::to_vec(&changed).unwrap();
    let state = Uploads {
        source_manifest: manifest_path.clone(),
        replacement: Arc::new(replacement.clone()),
        objects: Arc::new(Mutex::new(BTreeMap::new())),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/{*key}", put(upload))
        .with_state(state.clone());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let config = scratch.0.join("storage.json");
    fs::write(
        &config,
        serde_json::to_vec(&serde_json::json!({"storage": {
            "endpoint": endpoint,
            "region": "auto",
            "bucket": "bucket",
            "access_key_env": "SWITCHYARD_PUBLICATION_TEST_ACCESS",
            "secret_key_env": "SWITCHYARD_PUBLICATION_TEST_SECRET",
            "url_lifetime_s": 900
        }}))
        .unwrap(),
    )
    .unwrap();
    let credentials = scratch.0.join("credentials.env");
    fs::write(&credentials, "SWITCHYARD_PUBLICATION_TEST_ACCESS=test-access\nSWITCHYARD_PUBLICATION_TEST_SECRET=test-secret\n").unwrap();
    let receipt = scratch.0.join("receipt.json");
    run(Command::Publish {
        storage: StorageSettings {
            config,
            env_file: vec![credentials],
        },
        directory,
        prefix: "artifacts".into(),
        concurrency: 2,
        receipt: receipt.clone(),
    })
    .await
    .unwrap();
    assert_eq!(fs::read(manifest_path).unwrap(), replacement);
    let objects = state.objects.lock().unwrap();
    let uploaded = &objects[&format!("bucket/artifacts/{}/manifest.json", manifest.id)];
    assert_eq!(
        uploaded, &original,
        "publish the exact original serialization"
    );
    let published: Manifest = serde_json::from_slice(uploaded).unwrap();
    published.validate().unwrap();
    assert_eq!(published.id, manifest.id);
    assert!(!objects.contains_key(&format!("bucket/artifacts/{}/manifest.json", changed.id)));
    let receipt: serde_json::Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
    assert_eq!(receipt["id"], manifest.id);
    server.abort();
}
