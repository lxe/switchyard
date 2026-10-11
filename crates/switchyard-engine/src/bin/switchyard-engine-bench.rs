//! Reproducible real-endpoint benchmark. The first request is useful inference;
//! listener telemetry triggers it without an inference readiness probe.
use clap::Parser;
use futures_util::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use switchyard_engine::{config::Config, engine::Generate};
use tokio::io::{AsyncBufReadExt, BufReader};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    engine: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    requests: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 18101)]
    port: u16,
    #[arg(long)]
    reference: Option<PathBuf>,
    #[arg(long)]
    strict_parity: bool,
}
#[derive(Serialize)]
struct Call {
    token_ids: Vec<u32>,
    token_elapsed_s: Vec<f64>,
    elapsed_s: f64,
}
fn parity(calls: &[Call], gold: &serde_json::Value) -> Result<serde_json::Value> {
    let rows = gold.as_array().ok_or("Invalid reference rows")?;
    if calls.len() != rows.len() {
        return Err("Reference row count differs".into());
    }
    let mut differences = Vec::new();
    let mut matched = 0;
    let mut total = 0;
    for (i, (call, row)) in calls.iter().zip(rows).enumerate() {
        let expected: Vec<u32> = serde_json::from_value(row["token_ids"].clone())?;
        total += expected.len().max(call.token_ids.len());
        matched += expected
            .iter()
            .zip(&call.token_ids)
            .filter(|(a, b)| a == b)
            .count();
        if call.token_ids != expected {
            let first = expected
                .iter()
                .zip(&call.token_ids)
                .position(|(a, b)| a != b)
                .unwrap_or(expected.len().min(call.token_ids.len()));
            differences.push(serde_json::json!({"row":i,"first_difference":first,"expected":expected,"actual":call.token_ids}));
        }
    }
    Ok(
        serde_json::json!({"rows":calls.len(),"exact_rows":calls.len()-differences.len(),"matched_positions":matched,"positions":total,"differences":differences}),
    )
}
async fn call(client: &reqwest::Client, url: &str, model: &str, r: &Generate) -> Result<Call> {
    let start = Instant::now();
    let response = client
        .post(url)
        .json(&serde_json::json!({"model":model,"prompt":r.prompt,
        "max_tokens":r.max_tokens,"temperature":0,"stream":true,"ignore_eos":r.ignore_eos}))
        .send()
        .await?
        .error_for_status()?;
    let mut stream = response.bytes_stream();
    let mut pending = Vec::new();
    let mut ids = Vec::new();
    let mut times = Vec::new();
    let mut done = false;
    let mut stream_id = None;
    while let Some(chunk) = stream.next().await {
        pending.extend_from_slice(&chunk?);
        while let Some(end) = pending.iter().position(|&x| x == b'\n') {
            let line: Vec<u8> = pending.drain(..=end).collect();
            let line = std::str::from_utf8(&line)?.trim();
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                done = true;
                continue;
            }
            let event: serde_json::Value = serde_json::from_str(data)?;
            if event.get("error").is_some() {
                return Err(event.to_string().into());
            }
            let id = event["id"].as_str().ok_or("Missing completion ID")?;
            if id.is_empty()
                || event["created"].as_u64().is_none()
                || event["object"] != "text_completion"
            {
                return Err("Invalid OpenAI completion metadata".into());
            }
            if let Some(expected) = &stream_id {
                if expected != id {
                    return Err("Completion ID changed during stream".into());
                }
            } else {
                stream_id = Some(id.to_owned());
            }
            let tokens = event["choices"][0]["token_ids"]
                .as_array()
                .ok_or("Missing token IDs")?;
            for token in tokens {
                ids.push(token.as_u64().ok_or("Invalid token ID")? as u32);
                times.push(start.elapsed().as_secs_f64());
            }
        }
    }
    if !done || ids.is_empty() {
        return Err("Incomplete inference stream".into());
    }
    Ok(Call {
        token_ids: ids,
        token_elapsed_s: times,
        elapsed_s: start.elapsed().as_secs_f64(),
    })
}
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.strict_parity && args.reference.is_none() {
        return Err("Strict parity needs --reference".into());
    }
    let config: Config = serde_json::from_slice(&std::fs::read(&args.config)?)?;
    let requests: Vec<Generate> = serde_json::from_slice(&std::fs::read(&args.requests)?)?;
    if requests.is_empty() {
        return Err("Empty benchmark workload".into());
    }
    std::fs::create_dir_all(&args.output)?;
    let journal = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
    let mut log = std::fs::File::create(args.output.join("engine.jsonl"))?;
    let began = Instant::now();
    let mut child = tokio::process::Command::new(&args.engine)
        .args([
            "--config",
            args.config.to_str().ok_or("Invalid config path")?,
            "serve",
            "--bind",
            &format!("127.0.0.1:{}", args.port),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;
    let stderr = child.stderr.take().ok_or("Missing telemetry pipe")?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    let events = journal.clone();
    let reader = tokio::spawn(async move {
        let mut tx = Some(tx);
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = writeln!(log, "{line}");
            eprintln!("{line}");
            if let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) {
                if event["event"] == "http_listening"
                    && let Some(tx) = tx.take()
                {
                    let _ = tx.send(began.elapsed().as_secs_f64());
                }
                events.lock().unwrap().push(event);
            }
        }
    });
    let listener_s = tokio::time::timeout(Duration::from_secs(120), rx).await??;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let url = format!("http://127.0.0.1:{}/v1/completions", args.port);
    let first_started_s = began.elapsed().as_secs_f64();
    let first = call(&client, &url, &config.model, &requests[0]).await?;
    let launch_to_first_token_s = first_started_s + first.token_elapsed_s[0];
    let mut singles = Vec::new();
    for r in &requests {
        singles.push(call(&client, &url, &config.model, r).await?);
    }
    let burst_start = Instant::now();
    let burst = futures_util::future::try_join_all(
        requests
            .iter()
            .map(|r| call(&client, &url, &config.model, r)),
    )
    .await?;
    let burst_elapsed_s = burst_start.elapsed().as_secs_f64();
    child.kill().await?;
    reader.await?;
    let comparison = if let Some(path) = &args.reference {
        let gold: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        Some(
            serde_json::json!({"reference":gold,"single":parity(&singles,&gold["results"]["single"]["outputs"])?,
            "burst":parity(&burst,&gold["results"]["batch"]["outputs"])?}),
        )
    } else {
        None
    };
    let binary_sha256 = format!("{:x}", Sha256::digest(std::fs::read(&args.engine)?));
    let receipt = serde_json::json!({"schema_version":1,"scope":"resident local model files; process launch through real streamed HTTP token; no artifact hydration",
        "binary_sha256":binary_sha256,"comparison":comparison,
        "config":config,"requests":requests,"listener_s":listener_s,"launch_to_first_token_s":launch_to_first_token_s,
        "first_call":first,"single":singles,"burst":burst,"burst_elapsed_s":burst_elapsed_s,
        "burst_output_tps":burst.iter().map(|c|c.token_ids.len()).sum::<usize>() as f64 /burst_elapsed_s,
        "telemetry":*journal.lock().unwrap()});
    std::fs::write(
        args.output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!(
        "{}",
        serde_json::json!({"listener_s":listener_s,"launch_to_first_token_s":launch_to_first_token_s,
        "burst_elapsed_s":burst_elapsed_s,"burst_output_tps":receipt["burst_output_tps"],"receipt":args.output.join("receipt.json")})
    );
    if args.strict_parity
        && comparison.as_ref().is_some_and(|c| {
            c["single"]["exact_rows"] != c["single"]["rows"]
                || c["burst"]["exact_rows"] != c["burst"]["rows"]
        })
    {
        return Err("Token parity failed; differences retained in receipt.json".into());
    }
    Ok(())
}
