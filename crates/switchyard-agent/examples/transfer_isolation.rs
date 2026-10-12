//! Pod-side controlled network/decoder/filesystem isolation using production hydration.
//! Expiring URLs are fetched into memory; only safe counters are returned over HTTP.
use axum::{Json, Router, extract::State as HttpState, routing::get};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    io::{self, Read, Write},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use switchyard_agent::{State, Ticket, hydrate::HostSnapshot};
use tokio::sync::{RwLock, mpsc};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[derive(Deserialize)]
struct Job {
    cases: Vec<Case>,
}
#[derive(Clone, Deserialize)]
struct Case {
    name: String,
    mode: String,
    #[serde(default = "one")]
    connections: usize,
    #[serde(default)]
    url: String,
    #[serde(default)]
    bytes: u64,
    ticket: Option<Ticket>,
    #[serde(default = "yes")]
    sample: bool,
}
fn one() -> usize {
    1
}
fn yes() -> bool {
    true
}
#[derive(Default)]
struct Counters {
    received: AtomicU64,
    active: AtomicU64,
    peak: AtomicU64,
    wait_ns: AtomicU64,
}
type Reports = Arc<Mutex<Value>>;
fn snapshot() -> Value {
    let mut extra = serde_json::Map::new();
    for (key, path) in [
        ("cpu", "/sys/fs/cgroup/cpu.stat"),
        ("cpu_max", "/sys/fs/cgroup/cpu.max"),
        ("memory", "/sys/fs/cgroup/memory.stat"),
        ("memory_events", "/sys/fs/cgroup/memory.events"),
        ("memory_pressure", "/sys/fs/cgroup/memory.pressure"),
        ("cpu_pressure", "/sys/fs/cgroup/cpu.pressure"),
        ("io_pressure", "/sys/fs/cgroup/io.pressure"),
        ("io", "/sys/fs/cgroup/io.stat"),
        ("diskstats", "/proc/diskstats"),
        ("v1_cpu", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat"),
        (
            "v1_cpu_quota",
            "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
        ),
        (
            "v1_cpu_period",
            "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
        ),
        (
            "v1_cpu_usage_ns",
            "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage",
        ),
        ("v1_memory", "/sys/fs/cgroup/memory/memory.stat"),
        (
            "v1_memory_limit",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
        ("v1_memory_failcnt", "/sys/fs/cgroup/memory/memory.failcnt"),
        (
            "v1_io",
            "/sys/fs/cgroup/blkio/blkio.throttle.io_service_bytes",
        ),
        ("host_io_pressure", "/proc/pressure/io"),
        ("host_memory_pressure", "/proc/pressure/memory"),
    ] {
        if let Ok(s) = fs::read_to_string(path) {
            extra.insert(key.into(), json!(s));
        }
    }
    json!({"host":HostSnapshot::capture(),"system":extra})
}
struct Pipe(mpsc::Receiver<Vec<u8>>, io::Cursor<Vec<u8>>);
impl Read for Pipe {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            let n = self.1.read(out)?;
            if n != 0 {
                return Ok(n);
            }
            match self.0.blocking_recv() {
                Some(b) => self.1 = io::Cursor::new(b),
                None => return Ok(0),
            }
        }
    }
}
fn decode(reader: impl Read, output: Option<&Path>) -> Result<Value> {
    let started = Instant::now();
    let tid = nix::unistd::gettid();
    let before = fs::read_to_string(format!("/proc/self/task/{tid}/stat"))?;
    let mut decoder = zstd::stream::read::Decoder::new(reader)?;
    decoder.window_log_max(27)?;
    let mut file = output.map(fs::File::create).transpose()?;
    let mut buffer = vec![0; 64 << 10];
    let (mut bytes, mut reads, mut writes) = (0_u64, 0_f64, 0_f64);
    loop {
        let t = Instant::now();
        let n = decoder.read(&mut buffer)?;
        reads += t.elapsed().as_secs_f64();
        if n == 0 {
            break;
        }
        bytes += n as u64;
        if let Some(f) = file.as_mut() {
            let t = Instant::now();
            f.write_all(&buffer[..n])?;
            writes += t.elapsed().as_secs_f64();
        }
    }
    let after = fs::read_to_string(format!("/proc/self/task/{tid}/stat"))?;
    let ticks = |s: &str| -> u64 {
        let f: Vec<_> = s.rsplit_once(')').unwrap().1.split_whitespace().collect();
        f[11].parse::<u64>().unwrap() + f[12].parse::<u64>().unwrap()
    };
    Ok(
        json!({"decoded_bytes":bytes,"decode_read_elapsed_s":reads,"file_write_elapsed_s":writes,"worker_elapsed_s":started.elapsed().as_secs_f64(),"worker_cpu_ticks":ticks(&after)-ticks(&before),"clock_ticks_per_s":nix::unistd::sysconf(nix::unistd::SysconfVar::CLK_TCK)?.unwrap_or(100)}),
    )
}
async fn network(case: &Case, counts: Arc<Counters>) -> Result<Value> {
    let client = reqwest::Client::builder()
        .http1_only()
        .timeout(Duration::from_secs(240))
        .build()?;
    let n = case.connections;
    if !(1..=20).contains(&n) || case.bytes == 0 {
        return Err("invalid test bounds".into());
    }
    let workers = futures_util::stream::iter(0..n).map(|i| {
        let client = client.clone(); let counts = counts.clone();
        async move {
            let start = case.bytes * i as u64 / n as u64;
            let end = case.bytes * (i+1) as u64 / n as u64;
            let mut request = client.get(&case.url).header("Accept-Encoding", "identity");
            if n > 1 { request = request.header("Range", format!("bytes={start}-{}", end-1)); }
            let t = Instant::now();
            let response = request.send().await.map_err(|_| "request failed")?;
            if response.status().as_u16() != if n > 1 {206} else {200} { return Err("unexpected status".into()); }
            let headers_s = t.elapsed().as_secs_f64();
            let active = counts.active.fetch_add(1, Ordering::Relaxed)+1;
            counts.peak.fetch_max(active, Ordering::Relaxed);
            let (tx, rx) = mpsc::channel(4);
            let decoder = if case.mode.starts_with("decode") {
                let write = case.mode == "decode-write";
                Some(tokio::task::spawn_blocking(move || decode(Pipe(rx, io::Cursor::new(Vec::new())), write.then(|| Path::new("/work/decoded.tar")))))
            } else { drop(rx); None };
            let mut saved = if case.mode == "save" { Some(fs::File::create("/work/source.zst")?) } else {None};
            let mut body = response.bytes_stream();
            let (mut received, mut body_wait_s, mut send_wait_s) = (0_u64, 0_f64, 0_f64);
            loop {
                let t = Instant::now(); let next = body.next().await; body_wait_s += t.elapsed().as_secs_f64();
                let Some(b) = next else {break}; let b = b.map_err(|_| "body failed")?;
                received += b.len() as u64; counts.received.fetch_add(b.len() as u64, Ordering::Relaxed);
                if let Some(f) = saved.as_mut() { f.write_all(&b)?; }
                if decoder.is_some() {
                    for piece in b.chunks(64 << 10) {
                        let t = Instant::now(); tx.send(piece.to_vec()).await.map_err(|_| "decoder closed")?;
                        let waited = t.elapsed(); send_wait_s += waited.as_secs_f64(); counts.wait_ns.fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
                    }
                }
            }
            drop(tx); counts.active.fetch_sub(1, Ordering::Relaxed);
            if received != end-start { return Err("truncated response".into()); }
            let decoded = if let Some(worker) = decoder { worker.await?? } else {Value::Null};
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(json!({"index":i,"bytes":received,"headers_s":headers_s,"worker_elapsed_s":t.elapsed().as_secs_f64(),"body_wait_s":body_wait_s,"channel_wait_s":send_wait_s,"decoder":decoded}))
        }
    }).buffer_unordered(n);
    tokio::pin!(workers);
    let mut results = vec![];
    while let Some(result) = workers.next().await {
        results.push(result?);
    }
    Ok(json!({"workers":results,"peak_body_connections":counts.peak.load(Ordering::Relaxed)}))
}
async fn run_case(case: &Case, counts: Arc<Counters>) -> Result<Value> {
    if case.mode == "agent" {
        let ticket = case.ticket.as_ref().ok_or("missing ticket")?;
        let root = Path::new("/work").join(&case.name);
        fs::create_dir(&root)?;
        let state = Arc::new(RwLock::new(State::new(ticket)));
        switchyard_agent::hydrate::hydrate(ticket, &root, state.clone()).await?;
        let result = serde_json::to_value(&*state.read().await)?;
        fs::remove_dir_all(root)?;
        Ok(result)
    } else if case.mode.starts_with("local-decode") {
        let write = case.mode == "local-decode-write";
        tokio::task::spawn_blocking(move || {
            decode(
                fs::File::open("/work/source.zst")?,
                write.then(|| Path::new("/work/local-decoded.tar")),
            )
        })
        .await?
    } else {
        network(case, counts).await
    }
}
async fn suite(job: Job, report: Reports) {
    for case in job.cases {
        let started = Instant::now();
        let counts = Arc::new(Counters::default());
        println!(
            "{}",
            json!({"phase":"started","test":case.name,"connections":case.connections,"mode":case.mode})
        );
        let before = snapshot();
        let mut samples = vec![];
        let selected = case.clone();
        let work_counts = counts.clone();
        let work = tokio::spawn(async move { run_case(&selected, work_counts).await });
        tokio::pin!(work);
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        let result = loop {
            tokio::select! {
            r=&mut work => break r.unwrap_or_else(|_| Err("isolation worker failed".into())),
                _=tick.tick(), if case.sample => {
                let t=Instant::now(); let s=tokio::task::spawn_blocking(snapshot).await.unwrap();
                    samples.push(json!({"at_s":started.elapsed().as_secs_f64(),"sample_cost_s":t.elapsed().as_secs_f64(),"network_bytes":counts.received.load(Ordering::Relaxed),"active_bodies":counts.active.load(Ordering::Relaxed),"snapshot":s}));
                }
            }
        };
        let seconds = started.elapsed().as_secs_f64();
        let item = match result {
            Ok(details) => {
                json!({"name":case.name,"mode":case.mode,"seconds":seconds,"before":before,"after":snapshot(),"details":details,"samples":samples})
            }
            Err(_) => json!({"name":case.name,"failed":true,"seconds":seconds,"samples":samples}),
        };
        println!(
            "{}",
            json!({"phase":"finished","test":case.name,"seconds":seconds,"failed":item.get("failed")})
        );
        report.lock().unwrap()["cases"]
            .as_array_mut()
            .unwrap()
            .push(item);
        let _ = fs::remove_file("/work/decoded.tar");
        let _ = fs::remove_file("/work/local-decoded.tar");
    }
    report.lock().unwrap()["complete"] = json!(true);
}
#[tokio::main]
async fn main() -> Result<()> {
    fs::create_dir_all("/work")?;
    let job: Job = reqwest::get(std::env::var("ISOLATION_JOB_URL")?)
        .await
        .map_err(|_| "job fetch failed")?
        .json()
        .await
        .map_err(|_| "job decode failed")?;
    let report = Arc::new(Mutex::new(
        json!({"complete":false,"cases":[],"mountinfo":fs::read_to_string("/proc/self/mountinfo")?}),
    ));
    tokio::spawn(suite(job, report.clone()));
    let router = Router::new()
        .route(
            "/report",
            get(|HttpState(report): HttpState<Reports>| async move {
                Json(report.lock().unwrap().clone())
            }),
        )
        .with_state(report);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, router).await?;
    Ok(())
}
