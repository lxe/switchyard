//! A real HTTP client for the local simulator. Run the server with clock_rate0.
//! Transport, clock advancement, and scheduling decisions are separate here.
use reqwest::{Client, Method, StatusCode};
use serde_json::{Value, json};
use std::error::Error;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use tokio::task::JoinSet;

type Failure = Box<dyn Error + Send + Sync>;

#[derive(Clone)]
struct CloudClient {
    http: Client,
    base: String,
}

struct Reply {
    status: StatusCode,
    body: Value,
}

fn failure(message: impl Into<String>) -> Failure {
    std::io::Error::other(message.into()).into()
}

impl CloudClient {
    fn new(base: String) -> Result<Self, Failure> {
        Ok(Self {
            http: Client::builder().timeout(Duration::from_secs(60)).build()?,
            base: base.trim_end_matches('/').into(),
        })
    }

    async fn call(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Reply, Failure> {
        let mut request = self.http.request(method, format!("{}{path}", self.base));
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await?;
        let status = response.status();
        let bytes = response.bytes().await?;
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes)?
        };
        Ok(Reply { status, body })
    }

    async fn get(&self, path: &str) -> Result<Value, Failure> {
        self.expect(self.call(Method::GET, path, None).await?)
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value, Failure> {
        self.expect(self.call(Method::POST, path, Some(body)).await?)
    }

    fn expect(&self, reply: Reply) -> Result<Value, Failure> {
        if reply.status.is_success() {
            Ok(reply.body)
        } else {
            Err(failure(format!("HTTP {}: {}", reply.status, reply.body)))
        }
    }

    async fn time(&self) -> Result<f64, Failure> {
        self.get("/control/time").await?["time_s"]
            .as_f64()
            .ok_or_else(|| failure("missing clock receipt"))
    }

    async fn advance(&self, seconds: f64) -> Result<(), Failure> {
        self.post("/control/advance", json!({"seconds": seconds}))
            .await?;
        Ok(())
    }

    async fn request(&self, id: u64) -> Result<Value, Failure> {
        self.get(&format!("/v1/requests/{id}")).await
    }

    async fn release(&self, id: u64) -> Result<(), Failure> {
        self.expect(
            self.call(Method::DELETE, &format!("/v1/instances/{id}"), None)
                .await?,
        )?;
        Ok(())
    }
}

struct Sample {
    receipt: Value,
    observed_submit_s: f64,
}

struct Measurements {
    mode: String,
    model: String,
    start_s: f64,
    last_completion_s: f64,
    completed: usize,
    input: u64,
    output: u64,
    ttft: Vec<f64>,
    retries: usize,
}

impl Measurements {
    fn new(mode: &str, model: &str, start_s: f64) -> Self {
        Self {
            mode: mode.into(),
            model: model.into(),
            start_s,
            last_completion_s: start_s,
            completed: 0,
            input: 0,
            output: 0,
            ttft: vec![],
            retries: 0,
        }
    }

    fn record(&mut self, sample: Sample) -> Result<(), Failure> {
        let receipt = sample.receipt;
        self.input += receipt["usage"]["prompt_tokens"]
            .as_u64()
            .ok_or_else(|| failure("missing prompt usage"))?;
        self.output += receipt["usage"]["completion_tokens"]
            .as_u64()
            .ok_or_else(|| failure("missing completion usage"))?;
        self.last_completion_s = self.last_completion_s.max(
            receipt["completed_at_s"]
                .as_f64()
                .ok_or_else(|| failure("missing completion time"))?,
        );
        let submitted = receipt["submitted_at_s"]
            .as_f64()
            .unwrap_or(sample.observed_submit_s);
        if let Some(first) = receipt["first_token_at_s"].as_f64() {
            self.ttft.push((first - submitted).max(0.0));
        }
        self.completed += 1;
        Ok(())
    }

    fn receipt(mut self) -> Value {
        self.ttft.sort_by(f64::total_cmp);
        let percentile = |q: f64| {
            self.ttft
                .get(((self.ttft.len() as f64 * q).ceil() as usize).saturating_sub(1))
                .copied()
        };
        let elapsed = self.last_completion_s - self.start_s;
        json!({ "mode": self.mode, "model": self.model, "completed_requests": self.completed,
            "virtual_elapsed_s": elapsed, "aggregate_delivered_output_tps": if elapsed > 0.0 { self.output as f64 / elapsed } else { 0.0 },
            "ttft_p50_s": percentile(0.5), "ttft_p95_s": percentile(0.95),
            "ttft_basis": "server submitted_at_s when available, otherwise client clock observed before POST",
            "usage": { "prompt_tokens": self.input, "completion_tokens": self.output, "total_tokens": self.input + self.output },
            "capacity_retries": self.retries })
    }
}

// This clock task is an experiment driver. Neither policy changes provider
// reality directly; both use the same HTTP controls as another process could.
async fn drive_clock(client: CloudClient, stop: Arc<AtomicBool>) -> Result<(), Failure> {
    while !stop.load(Ordering::Relaxed) {
        client.advance(0.25).await?;
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    Ok(())
}

async fn offerings(client: &CloudClient) -> Result<Vec<Value>, Failure> {
    client.get("/v1/offerings").await?["offerings"]
        .as_array()
        .cloned()
        .ok_or_else(|| failure("missing offerings catalog"))
}

fn chat_body(model: &str) -> Value {
    json!({ "model": model, "messages": [{ "role": "user", "content": "Synthetic scheduling experiment" }],
        "max_completion_tokens": 512, "simulator": { "input_tokens": 128, "planned_output": 256 } })
}

// Client policy: bound outstanding front-door calls and retry only capacity
// errors. Fleet selection and provisioning belong to the server runtime.
async fn frontdoor(
    client: &CloudClient,
    model: &str,
    concurrency: usize,
) -> Result<Value, Failure> {
    let mut measurements = Measurements::new("frontdoor", model, client.time().await?);
    let mut tasks = JoinSet::new();
    let mut launched = 0;
    while measurements.completed < 100 {
        while launched < 100 && tasks.len() < concurrency {
            let client = client.clone();
            let model = model.to_owned();
            tasks.spawn(async move {
                let observed_submit_s = client.time().await?;
                let mut retries = 0;
                let response = loop {
                    let reply = client
                        .call(
                            Method::POST,
                            "/v1/chat/completions",
                            Some(chat_body(&model)),
                        )
                        .await?;
                    if reply.status == StatusCode::TOO_MANY_REQUESTS {
                        retries += 1;
                        tokio::time::sleep(Duration::from_millis(2)).await;
                        continue;
                    }
                    break client.expect(reply)?;
                };
                let id = response["id"]
                    .as_str()
                    .and_then(|id| id.strip_prefix("chatcmpl-sim-"))
                    .ok_or_else(|| failure("missing chat request ID"))?
                    .parse::<u64>()?;
                Ok::<_, Failure>((
                    Sample {
                        receipt: client.request(id).await?,
                        observed_submit_s,
                    },
                    retries,
                ))
            });
            launched += 1;
        }
        let task_result = tasks
            .join_next()
            .await
            .ok_or_else(|| failure("no outstanding requests"))?;
        let (sample, retries) = task_result??;
        measurements.record(sample)?;
        measurements.retries += retries;
    }
    Ok(measurements.receipt())
}

// External policy: pick one advertised text offering, wait and probe, bound
// admission by its advertised concurrency, poll receipts, and retry429. A real
// controller could replace these decisions without changing CloudClient.
async fn direct(
    client: &CloudClient,
    profile: &Value,
    concurrency: usize,
) -> Result<Value, Failure> {
    let profile_id = profile["id"]
        .as_str()
        .ok_or_else(|| failure("missing profile ID"))?;
    let model = profile["model"]
        .as_str()
        .ok_or_else(|| failure("missing model"))?;
    let limit = concurrency.min(
        profile["max_in_flight"]
            .as_u64()
            .ok_or_else(|| failure("missing concurrency limit"))? as usize,
    );
    if limit == 0 {
        return Err(failure("offering has no inference capacity"));
    }
    let mut measurements = Measurements::new("direct", model, client.time().await?);
    let id = client
        .post("/v1/instances", json!({ "profile_id": profile_id }))
        .await?["id"]
        .as_u64()
        .ok_or_else(|| failure("missing instance ID"))?;
    let instance_path = format!("/v1/instances/{id}");
    loop {
        let instance = client.get(&instance_path).await?;
        match instance["status"].as_str() {
            Some("ready") => break,
            Some("released") => return Err(failure("instance failed before readiness")),
            _ => tokio::time::sleep(Duration::from_millis(2)).await,
        }
    }
    client
        .post(&format!("{instance_path}/probes"), json!({}))
        .await?;
    loop {
        let instance = client.get(&instance_path).await?;
        if instance["host"]["probe"].is_object() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let spec = json!({ "model": model, "input_tokens": 128, "planned_output": 256, "max_output": 512,
        "workload": profile["workload"], "upload_bytes": 0, "image_megapixels": 0.0,
        "adapter": null, "class": "standard", "cached_fraction": 0.0 });
    let mut launched = 0;
    let mut active: Vec<(u64, f64)> = vec![];
    while measurements.completed < 100 {
        while launched < 100 && active.len() < limit {
            let observed_submit_s = client.time().await?;
            let reply = client
                .call(
                    Method::POST,
                    &format!("{instance_path}/inference"),
                    Some(spec.clone()),
                )
                .await?;
            if reply.status == StatusCode::TOO_MANY_REQUESTS {
                measurements.retries += 1;
                break;
            }
            let receipt = client.expect(reply)?;
            active.push((
                receipt["id"]
                    .as_u64()
                    .ok_or_else(|| failure("missing request ID"))?,
                observed_submit_s,
            ));
            launched += 1;
        }
        let mut pending = vec![];
        for (request_id, observed_submit_s) in active {
            let receipt = client.request(request_id).await?;
            match receipt["status"].as_str() {
                Some("completed") => measurements.record(Sample {
                    receipt,
                    observed_submit_s,
                })?,
                Some("failed") => return Err(failure("instance inference failed")),
                _ => pending.push((request_id, observed_submit_s)),
            }
        }
        active = pending;
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    client.release(id).await?;
    Ok(measurements.receipt())
}

#[tokio::main]
async fn main() -> Result<(), Failure> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "frontdoor".into());
    let url = args
        .next()
        .unwrap_or_else(|| "http://127.0.0.1:8080".into());
    let selection = args.next();
    let concurrency = args
        .next()
        .map(|n| n.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    if !(1..=256).contains(&concurrency) {
        return Err(failure("concurrency must be between 1 and 256"));
    }
    let client = CloudClient::new(url)?;
    let profiles = offerings(&client).await?;
    let selected = match mode.as_str() {
        "frontdoor" => profiles.iter().find(|profile| {
            profile["workload"] == "text"
                && selection
                    .as_ref()
                    .is_none_or(|model| profile["model"] == *model)
        }),
        "direct" => profiles
            .iter()
            .filter(|profile| {
                profile["workload"] == "text"
                    && selection.as_ref().is_none_or(|id| profile["id"] == *id)
            })
            .min_by(|a, b| {
                a["usd_per_hour"]
                    .as_f64()
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&b["usd_per_hour"].as_f64().unwrap_or(f64::INFINITY))
            }),
        _ => {
            return Err(failure(
                "usage: http_scheduler <frontdoor|direct> <url> [model-or-profile] [concurrency]",
            ));
        }
    }
    .ok_or_else(|| failure("no matching text offering"))?;
    let stop = Arc::new(AtomicBool::new(false));
    let driver = tokio::spawn(drive_clock(client.clone(), stop.clone()));
    let result = if mode == "frontdoor" {
        frontdoor(
            &client,
            selected["model"]
                .as_str()
                .ok_or_else(|| failure("missing model"))?,
            concurrency,
        )
        .await
    } else {
        direct(&client, selected, concurrency).await
    };
    stop.store(true, Ordering::Relaxed);
    driver.await??;
    println!("{}", serde_json::to_string_pretty(&result?)?);
    Ok(())
}
