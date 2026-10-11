#![cfg(unix)]
//! A failed consumer cancels HTTP and leaves neither a destination nor a socket.
use axum::{Router, body::Body, routing::get};
use std::{collections::BTreeMap, fs, time::Duration};
use switchyard_artifacts::{
    Metadata,
    extents::{ExtentTicket, pack_extents},
};
use tokio::{io::AsyncWriteExt, net::UnixStream};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn invalid_bounded_command_cancels_bridge_and_cleans_staging() {
    for disconnect in [false, true] {
        let root = std::env::temp_dir().join(format!(
            "weights-bridge-{}-{disconnect}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("source")).unwrap();
        fs::write(root.join("source/model.safetensors"), [1_u8; 8192]).unwrap();
        let manifest = pack_extents(
            &root.join("source"),
            &root.join("objects"),
            Metadata {
                kind: "test".into(),
                compatibility: BTreeMap::new(),
                attributes: BTreeMap::new(),
            },
            4096,
        )
        .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new().route(
            "/{name}",
            get(|| async {
                std::future::pending::<()>().await;
                Body::empty()
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let ticket = ExtentTicket {
            urls: manifest
                .extents
                .iter()
                .map(|e| (e.object.clone(), format!("{base}/{}", e.object)))
                .collect(),
            manifest,
        };
        fs::write(
            root.join("ticket.json"),
            serde_json::to_vec(&ticket).unwrap(),
        )
        .unwrap();
        let socket = root.join("bridge.sock");
        let mut bridge = tokio::process::Command::new(env!("CARGO_BIN_EXE_switchyard-weights"))
            .args([
                "serve",
                root.join("ticket.json").to_str().unwrap(),
                root.join("model").to_str().unwrap(),
                socket.to_str().unwrap(),
                "--trusted",
            ])
            .kill_on_drop(true)
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while !socket.exists() {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .unwrap();
        let mut client = UnixStream::connect(&socket).await.unwrap();
        client
            .write_all(if disconnect { b"next\n" } else { b"bad!\n" })
            .await
            .unwrap();
        drop(client);
        let status = tokio::time::timeout(Duration::from_secs(3), bridge.wait())
            .await
            .unwrap()
            .unwrap();
        assert!(!status.success());
        assert!(!socket.exists());
        assert!(!root.join("model").exists());
        assert!(fs::read_dir(&root).unwrap().all(|e| {
            !e.unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".staging")
        }));
        server.abort();
        fs::remove_dir_all(root).unwrap();
    }
}
