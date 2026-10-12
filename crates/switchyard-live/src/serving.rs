//! OpenAI-compatible streaming transport with usage and first-content timing receipts.
use crate::{
    Error, Result,
    planner::{Message, Planner},
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

mod cache_export;
mod http_failure;

pub(crate) fn completion_payload(model: &str, messages: &[Message], max_output: u64) -> Value {
    json!({"model":model,"messages":messages,"max_tokens":max_output,"temperature":0,"stream":true,
        "stream_options":{"include_usage":true},"chat_template_kwargs":{"enable_thinking":false}})
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Completion {
    pub request: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub started_s: f64,
    pub first_content_s: f64,
    pub completed_s: f64,
    pub finish_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
    /// Digest permits output parity checks without retaining request or response text.
    #[serde(default)]
    pub content_sha256: String,
    /// Compare literal answers while allowing surrounding whitespace.
    #[serde(default)]
    pub content_trimmed_sha256: String,
}
pub enum Event {
    Progress {
        request: usize,
        generated: u64,
    },
    Completed {
        request: usize,
        result: Result<Completion>,
    },
}

#[derive(Clone)]
pub struct Endpoint {
    pub url: String,
    client: reqwest::Client,
}
impl Endpoint {
    /// Metrics failures remain explicit observations, without exposing engine error bodies.
    pub async fn metrics(&self) -> Result<String> {
        let response = self
            .client
            .get(format!("{}/metrics", self.url))
            .timeout(Duration::from_secs(4))
            .send()
            .await
            .map_err(|_| Error::Request("engine metrics transport".into()))?;
        if !response.status().is_success() {
            return Err(Error::Request(format!(
                "engine metrics HTTP {}",
                response.status().as_u16()
            )));
        }
        let mut stream = response.bytes_stream();
        let mut bytes = vec![];
        while let Some(piece) = stream.next().await {
            let piece = piece.map_err(|_| Error::Request("engine metrics interrupted".into()))?;
            if bytes.len().saturating_add(piece.len()) > 4 << 20 {
                return Err(Error::Request("engine metrics exceed limit".into()));
            }
            bytes.extend_from_slice(&piece);
        }
        String::from_utf8(bytes).map_err(|_| Error::Request("engine metrics UTF-8".into()))
    }
    pub fn new(url: String, token: &str, timeout_s: u64) -> Result<Self> {
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| Error::Config("endpoint credential".into()))?;
        value.set_sensitive(true);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::AUTHORIZATION, value);
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent("Switchyard/0.1")
            .timeout(Duration::from_secs(timeout_s))
            .build()
            .map_err(|_| Error::Request("endpoint client".into()))?;
        Ok(Self { url, client })
    }
    pub async fn state(&self) -> Result<switchyard_agent::State> {
        let response = self
            .client
            .get(format!("{}/state", self.url))
            .timeout(Duration::from_secs(4))
            .send()
            .await
            .map_err(|_| Error::Request("agent observation".into()))?;
        if !response.status().is_success() {
            return Err(Error::Request(format!(
                "agent HTTP {}",
                response.status().as_u16()
            )));
        }
        response
            .json()
            .await
            .map_err(|_| Error::Request("agent observation schema".into()))
    }
    pub async fn verify_tokenization(
        &self,
        model: &str,
        messages: &[Message],
        expected: u64,
    ) -> Result<()> {
        let response=self.client.post(format!("{}/tokenize",self.url)).json(&json!({"model":model,"messages":messages,"add_generation_prompt":true,"chat_template_kwargs":{"enable_thinking":false}}))
            .send().await.map_err(|_|Error::Request("engine tokenizer check".into()))?;
        if !response.status().is_success() {
            return Err(http_failure::failure("tokenizer", response).await);
        }
        let value: Value = response
            .json()
            .await
            .map_err(|_| Error::Request("tokenizer response".into()))?;
        if value["count"].as_u64() != Some(expected) {
            return Err(Error::Request(format!(
                "local input count {expected} disagrees with engine count {:?}",
                value["count"]
            )));
        }
        Ok(())
    }
    pub async fn complete(&self, call: Call, events: mpsc::Sender<Event>) -> Result<Completion> {
        let started = call.epoch.elapsed().as_secs_f64();
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.url))
            .json(&completion_payload(
                &call.model,
                &call.messages,
                call.max_output,
            ))
            .send()
            .await
            .map_err(|_| Error::Request("inference transport".into()))?;
        if !response.status().is_success() {
            return Err(http_failure::failure("inference", response).await);
        }
        let mut decoder = SseDecoder::default();
        let mut stream = response.bytes_stream();
        let mut first = None;
        let mut usage = None;
        let mut finish = None;
        let mut content = String::new();
        let mut last_progress = 0;
        let mut done = false;
        while let Some(bytes) = stream.next().await {
            let bytes = bytes.map_err(|_| Error::Request("inference stream interrupted".into()))?;
            for data in decoder.feed(&bytes)? {
                if data == "[DONE]" {
                    done = true;
                    continue;
                }
                let value: Value = serde_json::from_str(&data)
                    .map_err(|_| Error::Request("invalid stream event".into()))?;
                if value.get("error").is_some() {
                    return Err(Error::Request("engine reported a stream error".into()));
                }
                if let Some(text) = value["choices"][0]["delta"]["content"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                {
                    first.get_or_insert_with(|| call.epoch.elapsed().as_secs_f64());
                    content.push_str(text);
                    if content.len().saturating_sub(last_progress) >= 32 {
                        last_progress = content.len();
                        let generated = call.planner.output_tokens(&content);
                        let _ = events
                            .send(Event::Progress {
                                request: call.id,
                                generated,
                            })
                            .await;
                    }
                }
                if let Some(reason) = value["choices"][0]["finish_reason"].as_str() {
                    finish = Some(reason.to_owned());
                }
                if value["usage"].is_object() {
                    usage = Some(value["usage"].clone());
                }
            }
        }
        let usage = usage.ok_or_else(|| Error::Request("stream ended without usage".into()))?;
        let input = usage["prompt_tokens"]
            .as_u64()
            .ok_or_else(|| Error::Request("missing prompt usage".into()))?;
        let output = usage["completion_tokens"]
            .as_u64()
            .ok_or_else(|| Error::Request("missing completion usage".into()))?;
        if !done || input != call.input_tokens || output > call.max_output {
            return Err(Error::Request(
                "stream framing, input count, or output bound failed".into(),
            ));
        }
        // Empty-content responses do not manufacture a first-content measurement.
        let first = first.ok_or_else(|| Error::Request("completion had no content".into()))?;
        Ok(Completion {
            request: call.id,
            input_tokens: input,
            output_tokens: output,
            started_s: started,
            first_content_s: first,
            completed_s: call.epoch.elapsed().as_secs_f64(),
            finish_reason: finish.unwrap_or_else(|| "unknown".into()),
            cached_input_tokens: usage["prompt_tokens_details"]["cached_tokens"].as_u64(),
            content_sha256: {
                use sha2::Digest;
                format!("{:x}", sha2::Sha256::digest(content.as_bytes()))
            },
            content_trimmed_sha256: {
                use sha2::Digest;
                format!("{:x}", sha2::Sha256::digest(content.trim().as_bytes()))
            },
        })
    }
}

pub struct Call {
    pub id: usize,
    pub model: String,
    pub messages: Vec<Message>,
    pub max_output: u64,
    pub input_tokens: u64,
    pub planner: Arc<Planner>,
    pub epoch: Instant,
}

/// Incremental SSE parsing handles arbitrary TCP fragments and bounds unfinished event data.
#[derive(Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    data: Vec<String>,
}
impl SseDecoder {
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        self.buffer.extend_from_slice(bytes);
        let mut output = vec![];
        if self.buffer.len() > 2 << 20 {
            return Err(Error::Request("stream event exceeds limit".into()));
        }
        while let Some(end) = self.buffer.iter().position(|b| *b == b'\n') {
            let line: Vec<_> = self.buffer.drain(..=end).collect();
            let line = std::str::from_utf8(&line)
                .map_err(|_| Error::Request("invalid stream UTF-8".into()))?
                .trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                if !self.data.is_empty() {
                    output.push(self.data.join("\n"));
                    self.data.clear();
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data
                    .push(data.strip_prefix(' ').unwrap_or(data).to_owned());
            }
        }
        if self.data.iter().map(String::len).sum::<usize>() > 2 << 20 {
            return Err(Error::Request("stream event exceeds limit".into()));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fragment_boundary_preserves_utf8_and_crlf_events() {
        let stream =
            "data: {\"content\":\"雪🚆\"}\r\n\r\n: heartbeat\n\ndata: [DONE]\n\n".as_bytes();
        for split in 0..=stream.len() {
            let mut decoder = SseDecoder::default();
            let mut events = decoder.feed(&stream[..split]).unwrap();
            events.extend(decoder.feed(&stream[split..]).unwrap());
            assert_eq!(events, ["{\"content\":\"雪🚆\"}", "[DONE]"]);
        }
    }

    #[test]
    fn multiline_events_and_limits_are_enforced() {
        let mut decoder = SseDecoder::default();
        assert_eq!(
            decoder.feed(b"data: one\ndata:two\n\n").unwrap(),
            ["one\ntwo"]
        );
        assert!(decoder.feed(&vec![b'x'; (2 << 20) + 1]).is_err());
        assert!(SseDecoder::default().feed(b"data: \xff\n\n").is_err());
    }
}
