use crate::{
    config::{valid_component, validate_endpoint},
    *,
};
use reqwest::{Client, Method, StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Clone)]
pub struct RunPod {
    client: Client,
    base: String,
    config: Option<ComputeConfig>,
}

impl RunPod {
    pub fn new(config: &ComputeConfig) -> Result<Self> {
        config.validate()?;
        let key = std::env::var(&config.key_env)
            .map_err(|_| Error::Config(format!("missing {}", config.key_env)))?;
        Self::with_config(config, &key)
    }
    /// Construct a provisioning adapter with provider-specific capacity settings.
    pub fn with_config(config: &ComputeConfig, key: &str) -> Result<Self> {
        config.validate()?;
        let mut provider = Self::with_key(&config.api_url, key)?;
        provider.config = Some(config.clone());
        Ok(provider)
    }
    /// Construct a REST-only client for inventory, lifecycle reads, and billing.
    /// Capacity discovery and creation require `with_config` instead.
    pub fn with_key(base: &str, key: &str) -> Result<Self> {
        validate_endpoint(base)?;
        if key.is_empty() {
            return Err(Error::Config("empty provider credential".into()));
        }
        let mut auth = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| Error::Config("invalid provider credential".into()))?;
        auth.set_sensitive(true);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::AUTHORIZATION, auth);
        let client = Client::builder()
            .default_headers(headers)
            .user_agent("Switchyard/0.1")
            .timeout(Duration::from_secs(45))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| Error::Transport("client construction".into()))?;
        Ok(Self {
            client,
            base: base.trim_end_matches('/').into(),
            config: None,
        })
    }
    async fn call<T: serde::de::DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<T> {
        let mut request = self.client.request(method, format!("{}{path}", self.base));
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request
            .send()
            .await
            .map_err(|_| Error::Transport(path.into()))?;
        if !response.status().is_success() {
            return Err(Error::Http {
                status: response.status().as_u16(),
                operation: path.into(),
            });
        }
        response
            .json()
            .await
            .map_err(|_| Error::Response(path.into()))
    }
    async fn pod(&self, method: Method, path: &str, body: Option<Value>) -> Result<Pod> {
        self.call::<PodResponse>(method, path, body)
            .await?
            .try_into()
    }
    /// RunPod-specific hourly billing records over an RFC 3339 time window.
    /// Returned vendor JSON remains separate from the generic lifecycle contract.
    pub async fn billing(&self, start: &str, end: &str) -> Result<Value> {
        let url = format!("{}/billing/pods", self.base);
        let response = self
            .client
            .get(url)
            .query(&[
                ("startTime", start),
                ("endTime", end),
                ("bucketSize", "hour"),
            ])
            .send()
            .await
            .map_err(|_| Error::Transport("billing".into()))?;
        if !response.status().is_success() {
            return Err(Error::Http {
                status: response.status().as_u16(),
                operation: "billing".into(),
            });
        }
        response
            .json()
            .await
            .map_err(|_| Error::Response("billing".into()))
    }
}

#[derive(Deserialize)]
struct Catalog {
    gpus: Vec<Gpu>,
}
#[derive(Deserialize)]
struct Gpu {
    id: String,
    memory: u64,
    price: std::collections::BTreeMap<String, f64>,
    #[serde(rename = "dataCenters", default)]
    data_centers: Vec<DataCenter>,
}
#[derive(Deserialize)]
struct DataCenter {
    id: String,
    availability: String,
}
#[derive(Deserialize)]
struct Pods {
    pods: Vec<PodResponse>,
}

/// RunPod wire metadata stays inside the adapter; ignored fields may contain credentials.
#[derive(Deserialize)]
struct PodResponse {
    id: String,
    name: String,
    status: String,
    cost: f64,
    #[serde(rename = "dataCenterId", default)]
    data_center: Option<String>,
}

impl TryFrom<PodResponse> for Pod {
    type Error = Error;

    fn try_from(response: PodResponse) -> Result<Self> {
        if response.status.is_empty() {
            return Err(Error::Response("pod metadata".into()));
        }
        // https://docs.runpod.io/api-reference-v2/pods/get-a-pod
        let status = match response.status.as_str() {
            "PROVISIONING" | "STARTING" => PodStatus::Starting,
            "RUNNING" => PodStatus::Running,
            "EXITED" => PodStatus::Stopped,
            "ERROR" => PodStatus::Failed,
            "TERMINATED" | "DELETED" => PodStatus::Terminated,
            _ => PodStatus::Unknown,
        };
        let pod = Self {
            id: response.id,
            name: response.name,
            status,
            cost: response.cost,
            data_center: response.data_center,
        };
        if valid_pod(&pod) {
            Ok(pod)
        } else {
            Err(Error::Response("pod metadata".into()))
        }
    }
}

impl Provider for RunPod {
    fn endpoint_url(&self, pod: &Pod, port: u16) -> String {
        format!("https://{}-{port}.proxy.runpod.net", pod.id)
    }
    async fn offers(&self) -> Result<Vec<Offering>> {
        let config = self.config.as_ref().ok_or_else(|| {
            Error::Config("capacity discovery requires RunPod compute settings".into())
        })?;
        let response = self
            .client
            .get(format!("{}/catalog/gpus", self.base))
            .query(&[
                ("include", "AVAILABILITY"),
                ("product", "POD"),
                ("cloud", config.cloud.as_str()),
                ("count", "1"),
                ("minCudaVersion", config.min_cuda.as_str()),
            ])
            .send()
            .await
            .map_err(|_| Error::Transport("catalog".into()))?;
        if !response.status().is_success() {
            return Err(Error::Http {
                status: response.status().as_u16(),
                operation: "catalog".into(),
            });
        }
        let catalog: Catalog = response
            .json()
            .await
            .map_err(|_| Error::Response("catalog".into()))?;
        let mut out = vec![];
        for gpu in catalog.gpus {
            if !config.gpu_types.contains(&gpu.id) {
                continue;
            }
            let Some(&rate) = gpu.price.get(&config.cloud.to_lowercase()) else {
                continue;
            };
            if !rate.is_finite() || rate <= 0.0 {
                continue;
            }
            for dc in gpu.data_centers {
                if !matches!(dc.availability.as_str(), "LOW" | "MEDIUM" | "HIGH")
                    || (!config.data_centers.is_empty() && !config.data_centers.contains(&dc.id))
                {
                    continue;
                }
                out.push(Offering {
                    gpu: gpu.id.clone(),
                    gpu_memory_gb: gpu.memory,
                    data_center: dc.id,
                    availability: dc.availability,
                    gpu_usd_per_hour: rate,
                    disk_usd_per_hour: config.disk_gb as f64 * config.disk_usd_per_gb_month
                        / config.billing_hours_per_month,
                    quoted_at_unix_s: unix_s(),
                });
            }
        }
        out.sort_by(|a, b| {
            a.total_usd_per_hour()
                .total_cmp(&b.total_usd_per_hour())
                .then(a.data_center.cmp(&b.data_center))
        });
        Ok(out)
    }
    async fn create(&self, launch: Launch) -> Result<Pod> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| Error::Config("creation requires RunPod compute settings".into()))?;
        launch.container.validate()?;
        if !valid_component(&launch.name)
            || launch.container.disk_gb != config.disk_gb
            || !config.gpu_types.contains(&launch.offering.gpu)
            || (!config.data_centers.is_empty()
                && !config.data_centers.contains(&launch.offering.data_center))
        {
            return Err(Error::Config(
                "launch must match the configured capacity and quoted disk allocation".into(),
            ));
        }
        // The name is the recovery key when POST succeeds but its response is lost.
        // Refuse an existing name before allocating, so reconciliation cannot adopt another pod.
        if self.list().await?.iter().any(|pod| pod.name == launch.name) {
            return Err(Error::Config(
                "launch name already exists; recover its inventory entry instead".into(),
            ));
        }
        let ports: Vec<_> = launch
            .container
            .ports
            .iter()
            .map(|port| format!("{}/{}", port.number, if port.http { "http" } else { "tcp" }))
            .collect();
        let mut body = json!({
            "name":launch.name,"image":launch.container.image,
            "gpu":{"id":launch.offering.gpu,"count":1,"minCudaVersion":config.min_cuda},
            "cloud":config.cloud,"disk":launch.container.disk_gb,"dataCenterIds":[launch.offering.data_center],
            "ports":ports,"startSsh":false,"env":launch.container.environment,
        });
        if !launch.container.entrypoint.is_empty() {
            body["entrypoint"] = json!(launch.container.entrypoint);
        }
        if !launch.container.command.is_empty() {
            body["cmd"] = json!(launch.container.command);
        }
        match self.pod(Method::POST, "/pods", Some(body)).await {
            Ok(pod) if pod.name == launch.name && valid_pod(&pod) => Ok(pod),
            // A malformed success can also follow allocation. Do not lose ownership.
            Ok(_)
            | Err(
                Error::Transport(_)
                | Error::Response(_)
                | Error::Http {
                    status: 408 | 500..=599,
                    ..
                },
            ) => {
                for delay_ms in [0, 250, 500, 1000, 2000] {
                    if delay_ms != 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                    let Ok(pods) = self.list().await else {
                        continue;
                    };
                    let mut found = pods
                        .into_iter()
                        .filter(|pod| pod.name == launch.name && !pod.status.is_terminated());
                    match (found.next(), found.next()) {
                        (Some(pod), None) => return Ok(pod),
                        (Some(_), Some(_)) => break,
                        _ => {}
                    }
                }
                Err(Error::UnknownLaunch { name: launch.name })
            }
            Err(error) => Err(error),
        }
    }
    async fn get(&self, id: &str) -> Result<Option<Pod>> {
        validate_pod_id(id)?;
        match self.pod(Method::GET, &format!("/pods/{id}"), None).await {
            Ok(pod) if pod.id == id && valid_pod(&pod) => Ok(Some(pod)),
            Ok(_) => Err(Error::Response("pod metadata".into())),
            Err(Error::Http { status: 404, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }
    async fn list(&self) -> Result<Vec<Pod>> {
        self.call::<Pods>(Method::GET, "/pods", None)
            .await?
            .pods
            .into_iter()
            .map(Pod::try_from)
            .collect()
    }
    async fn terminate(&self, id: &str) -> Result<()> {
        validate_pod_id(id)?;
        let response = self
            .client
            .delete(format!("{}/pods/{id}", self.base))
            .send()
            .await
            .map_err(|_| Error::Transport("terminate".into()))?;
        if response.status().is_success() || response.status() == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            Err(Error::Http {
                status: response.status().as_u16(),
                operation: "terminate".into(),
            })
        }
    }
}

fn validate_pod_id(id: &str) -> Result<()> {
    if valid_component(id) {
        Ok(())
    } else {
        Err(Error::Config("invalid pod identifier".into()))
    }
}

fn valid_pod(pod: &Pod) -> bool {
    valid_component(&pod.id)
        && !pod.name.is_empty()
        && !pod.name.chars().any(char::is_control)
        && pod.cost.is_finite()
        && pod.cost >= 0.0
}
