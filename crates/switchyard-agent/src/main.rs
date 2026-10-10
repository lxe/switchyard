use std::{error::Error, time::Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let url = std::env::var("SWITCHYARD_TICKET_URL")?;
    let token = std::env::var("SWITCHYARD_AGENT_TOKEN")?;
    if token.len() < 32 {
        return Err("agent token must contain at least 32 characters".into());
    }
    let client = reqwest::Client::builder()
        .user_agent("Switchyard/0.1")
        .timeout(Duration::from_secs(30))
        .build()?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|_| "ticket download failed")?;
    if !response.status().is_success() {
        return Err("ticket download was rejected".into());
    }
    let ticket: switchyard_agent::Ticket = response.json().await.map_err(|_| "invalid ticket")?;
    switchyard_agent::server::serve(ticket, token, "0.0.0.0:8080".parse()?).await?;
    Ok(())
}
