use std::{error::Error, time::Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let epoch = std::time::Instant::now();
    let url = std::env::var("SWITCHYARD_TICKET_URL")?;
    let token = std::env::var("SWITCHYARD_AGENT_TOKEN")?;
    if token.len() < 32 {
        return Err("agent token must contain at least 32 characters".into());
    }
    let client = reqwest::Client::builder()
        .user_agent("Switchyard/0.1")
        // Complete runtime inventories can produce tickets larger than 50 MiB.
        // Keep this bounded, but allow their bodies to arrive before decoding.
        .timeout(Duration::from_secs(120))
        .build()?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|_| "ticket download failed")?;
    if !response.status().is_success() {
        return Err("ticket download was rejected".into());
    }
    let body = response.bytes().await.map_err(|error| {
        if error.is_timeout() {
            "ticket body download timed out"
        } else {
            "ticket body download interrupted"
        }
    })?;
    let downloaded_s = epoch.elapsed().as_secs_f64();
    let ticket: switchyard_agent::Ticket =
        serde_json::from_slice(&body).map_err(|_| "ticket JSON or schema is invalid")?;
    println!(
        "switchyard ticket received: bytes={} download_s={downloaded_s:.3} decode_s={:.3}",
        body.len(),
        epoch.elapsed().as_secs_f64() - downloaded_s
    );
    switchyard_agent::server::serve_started(ticket, token, "0.0.0.0:8080".parse()?, epoch).await?;
    Ok(())
}
