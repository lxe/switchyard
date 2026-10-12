use super::*;
use crate::{
    Chunk, FileInfo, Manifest,
    host::HostedEndpoint,
    model::{LegacySource, ModelSource},
};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    sync::atomic::{AtomicUsize, Ordering},
};
use switchyard_artifacts::{ArtifactTicket, Metadata, PackOptions, pack};
use switchyard_core::topology::{EndpointPlan, HostPlan};
use tower::ServiceExt;

const TOKEN: &str = "host-test-token-at-least-32-characters";
fn plan() -> HostPlan {
    HostPlan {
        id: "pair".into(),
        gpu: "test".into(),
        gpu_memory_bytes: 4 << 20,
        reserved_gpu_memory_bytes: 1 << 20,
        contention_factor: 1.0,
        endpoints: ["replica-0", "replica-1"]
            .into_iter()
            .map(|id| EndpointPlan {
                id: id.into(),
                profile: "shared".into(),
                gpu_memory_bytes: 1 << 20,
                compute_fraction: 0.5,
            })
            .collect(),
    }
}

#[tokio::test]
async fn endpoint_proxy_keeps_authentication_queries_and_sibling_state_separate() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let upstream = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().fallback(|request: Request| async move {
                assert!(request.headers().get("authorization").is_none());
                request.uri().to_string()
            }),
        )
        .await
        .unwrap();
    });
    let mut ready = super::super::tests::app(Phase::Ready, base);
    ready.authorization_hash = Sha256::digest(format!("Bearer {TOKEN}").as_bytes()).into();
    let mut failed = ready.clone();
    failed.state = Arc::new(RwLock::new(ready.state.read().await.clone()));
    failed.state.write().await.phase = Phase::Failed;
    let routes = router(HostApp {
        plan: plan(),
        endpoints: BTreeMap::from([("ready".into(), ready), ("failed".into(), failed)]),
    });
    let request = |url: &str, auth: bool| {
        let mut r = Request::builder().uri(url);
        if auth {
            r = r.header("authorization", format!("Bearer {TOKEN}"));
        }
        r.body(Body::empty()).unwrap()
    };
    assert_eq!(
        routes
            .clone()
            .oneshot(request("/endpoints/ready/v1/models", false))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        routes
            .clone()
            .oneshot(request("/endpoints/failed/v1/models", true))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        routes
            .clone()
            .oneshot(request("/endpoints/unknown/v1/models", true))
            .await
            .unwrap()
            .status(),
        StatusCode::NOT_FOUND
    );
    let response = routes
        .oneshot(request("/endpoints/ready/v1/models?test=1", true))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap(),
        "/v1/models?test=1"
    );
    upstream.abort();
}

#[tokio::test]
async fn actual_host_hydrates_once_launches_two_processes_and_stops_them_on_expiry() {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).unwrap();
    let root =
        std::env::temp_dir().join(format!("switchyard-host-{:x}", u64::from_le_bytes(bytes)));
    fs::create_dir(&root).unwrap();
    let source = root.join("source");
    fs::create_dir(&source).unwrap();
    let script = r#"#!/usr/bin/python3
import http.server,json,os,sys
config=json.loads(sys.argv[2]); port=int(sys.argv[-1].split(':')[-1])
open(config['model_root']+'/pid-'+str(port),'w').write(str(os.getpid()))
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
  self.wfile.write(json.dumps({'model':config['model'],'port':port}).encode())
 def log_message(self,*args): pass
http.server.HTTPServer(('127.0.0.1',port),Handler).serve_forever()
"#;
    fs::write(source.join("engine"), script).unwrap();
    fs::set_permissions(source.join("engine"), fs::Permissions::from_mode(0o755)).unwrap();
    let packed = root.join("packed");
    fs::create_dir(&packed).unwrap();
    let manifest = pack(
        &source,
        &packed,
        Metadata {
            kind: "runtime".into(),
            compatibility: BTreeMap::from([("architecture".into(), std::env::consts::ARCH.into())]),
            attributes: BTreeMap::new(),
        },
        PackOptions::default(),
    )
    .unwrap();
    let mut tar = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(2);
    header.set_mode(0o644);
    header.set_cksum();
    tar.append_data(&mut header, "config.json", b"{}".as_slice())
        .unwrap();
    let model_bytes = tar.into_inner().unwrap();
    let mut bodies = BTreeMap::from([("weights".to_owned(), model_bytes.clone())]);
    for chunk in &manifest.chunks {
        bodies.insert(
            chunk.name.clone(),
            fs::read(packed.join(&chunk.name)).unwrap(),
        );
    }
    let bodies = Arc::new(bodies);
    let requests = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let counter = requests.clone();
    let http = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route(
                "/{name}",
                get(move |Path(name): Path<String>| {
                    let bodies = bodies.clone();
                    let counter = counter.clone();
                    async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        Body::from(bodies[&name].clone())
                    }
                }),
            ),
        )
        .await
        .unwrap();
    });
    let model_root = root.join("models");
    let endpoints=[("replica-0",8130), ("replica-1",8131)].into_iter().map(|(id,port)| {
        let startup=crate::startup::StartupPolicy {mode:switchyard_artifacts::HydrationMode::TrustedStreaming,model_root:Some(model_root.to_string_lossy().into()),..Default::default()};
        let config=serde_json::json!({"model":"twin","model_root":startup.model_directory(),"device":{"kind":"cuda","ordinal":0},"max_weight_bytes":1024});
        HostedEndpoint {id:id.into(),port,ticket:Ticket {
            model:"twin".into(),revision:"revision".into(),concurrency:20,stall_s:10,lifetime_s:2,
            engine_args:vec!["--config-json".into(),config.to_string(),"serve".into(),"--bind".into(),format!("127.0.0.1:{port}")],startup,
            source:ModelSource::Legacy(LegacySource {manifest:Manifest {
                format:"shards-v1".into(),compress:"none".into(),sha256:"a".repeat(64),total_size:model_bytes.len() as u64,raw_size:model_bytes.len() as u64,
                chunks:vec![Chunk{name:"weights".into(),size:model_bytes.len() as u64,raw:model_bytes.len() as u64,shard:0,seq:0,sha256:"b".repeat(64)}],
                files:BTreeMap::from([("config.json".into(),FileInfo{size:2,sha256:None})]),
            },urls:BTreeMap::from([("weights".into(),format!("{base}/weights"))])}),
            runtime:Some(crate::runtime::RuntimeTicket {hydration:None,artifact:ArtifactTicket {manifest:manifest.clone(),urls:manifest.chunks.iter().map(|c|(c.name.clone(),format!("{base}/{}",c.name))).collect()},
                compatibility:BTreeMap::new(),components:vec![],program:"engine".into(),args:vec![],environment:BTreeMap::new(),caches:vec![],cache_export:None}),
        }}
    }).collect();
    let ticket = HostTicket {
        schema_version: 1,
        storage_root: None,
        plan: plan(),
        endpoints,
    };
    ticket.validate().unwrap();
    let mut invalid = ticket.clone();
    invalid.endpoints[1].port = invalid.endpoints[0].port;
    assert!(invalid.validate().is_err());
    invalid = ticket.clone();
    invalid.endpoints[1].ticket.concurrency = 1;
    assert!(invalid.validate().is_err());
    invalid = ticket.clone();
    invalid.storage_root = Some(
        invalid.endpoints[0]
            .ticket
            .startup
            .model_directory()
            .parent()
            .unwrap()
            .to_string_lossy()
            .into(),
    );
    invalid.endpoints[0].ticket.startup.model_root = Some(format!(
        "{}/runtime",
        invalid.storage_root.as_ref().unwrap()
    ));
    assert!(invalid.validate().is_err());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let root_copy = root.clone();
    let agent = tokio::spawn(async move {
        serve_at(
            ticket,
            TOKEN.into(),
            address,
            std::time::Instant::now(),
            Some(root_copy),
        )
        .await
    });
    let client = reqwest::Client::new();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let mut ready = 0;
            for id in ["replica-0", "replica-1"] {
                if let Ok(response) = client
                    .get(format!("http://{address}/endpoints/{id}/v1/models"))
                    .bearer_auth(TOKEN)
                    .send()
                    .await
                    && response.status().is_success()
                {
                    ready += 1;
                }
            }
            if ready == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        requests.load(Ordering::SeqCst),
        manifest.chunks.len() + 1,
        "runtime and weights must hydrate once across replicas"
    );
    assert!(root.join("cache/endpoints/replica-0").exists());
    assert!(root.join("cache/endpoints/replica-1").exists());
    let pids: Vec<_> = [8130, 8131]
        .into_iter()
        .map(|port| {
            fs::read_to_string(model_root.join(format!("model/pid-{port}")))
                .unwrap()
                .parse::<u32>()
                .unwrap()
        })
        .collect();
    agent.await.unwrap().unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while pids
            .iter()
            .any(|pid| std::path::Path::new(&format!("/proc/{pid}")).exists())
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    http.abort();
    fs::remove_dir_all(root).unwrap();
}
