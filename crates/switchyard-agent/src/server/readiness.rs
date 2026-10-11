//! Readiness requires tokenization and one decoded token after engine health succeeds.
use crate::{Error, Result, startup::StartupEvent};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::time::Duration;

#[cfg(test)]
pub(super) async fn verify(client: &reqwest::Client, base: &str, model: &str) -> Result<()> {
    verify_observed(client, base, model, |_| async {}).await
}

pub(super) async fn verify_observed<F, Fut>(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    mut observe: F,
) -> Result<()>
where
    F: FnMut(StartupEvent) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let messages = json!([{"role":"user","content":"Reply with one word."}]);
    observe(StartupEvent::TokenizationStarted).await;
    let tokenized = request(
        client,
        &format!("{base}/tokenize"),
        json!({
            "model":model, "messages":messages, "add_generation_prompt":true,
            "chat_template_kwargs":{"enable_thinking":false}
        }),
    )
    .await?;
    observe(StartupEvent::TokenizationResponseReceived).await;
    if !tokenized
        .get("count")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 0)
    {
        return Err(Error::Transfer(
            "engine readiness tokenization returned no tokens".into(),
        ));
    }
    observe(StartupEvent::FirstDecodeStarted).await;
    let completion = request(
        client,
        &format!("{base}/v1/chat/completions"),
        json!({
            "model":model, "messages":messages, "max_tokens":1, "temperature":0,
            "stream":false, "chat_template_kwargs":{"enable_thinking":false}
        }),
    )
    .await?;
    observe(StartupEvent::FirstDecodeResponseReceived).await;
    let usage = completion.get("usage");
    let prompt_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Value::as_u64);
    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Value::as_u64);
    let choices = completion.get("choices").and_then(Value::as_array);
    if !prompt_tokens.is_some_and(|n| n > 0)
        || output_tokens != Some(1)
        || !choices.is_some_and(|choices| {
            choices
                .first()
                .is_some_and(|choice| choice.get("message").is_some_and(Value::is_object))
        })
    {
        return Err(Error::Transfer(
            "engine readiness completion did not confirm decoding".into(),
        ));
    }
    Ok(())
}

async fn request(client: &reqwest::Client, url: &str, body: Value) -> Result<Value> {
    // A first decode can initialize kernels after /health; the agent lifetime still bounds it.
    let response = client
        .post(url)
        .json(&body)
        .timeout(Duration::from_secs(60))
        .send()
        .await
        .map_err(|_| Error::Transfer("engine readiness request failed".into()))?;
    if !response.status().is_success() {
        return Err(Error::Transfer(format!(
            "engine readiness returned HTTP {}",
            response.status().as_u16()
        )));
    }
    // Readiness never exposes engine response bodies or prompts in failure diagnostics.
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(piece) = stream.next().await {
        let piece =
            piece.map_err(|_| Error::Transfer("engine readiness response interrupted".into()))?;
        if bytes.len().saturating_add(piece.len()) > (1 << 20) {
            return Err(Error::Transfer(
                "engine readiness response is too large".into(),
            ));
        }
        bytes.extend_from_slice(&piece);
    }
    serde_json::from_slice(&bytes)
        .map_err(|_| Error::Transfer("engine readiness returned invalid JSON".into()))
}
