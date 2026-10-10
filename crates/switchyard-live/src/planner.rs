//! Count the pinned model's rendered chat prompt before acquiring a GPU.
use crate::config::ModelConfig;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
pub struct Planner {
    tokenizer: std::sync::Arc<Tokenizer>,
    template: String,
}
impl Planner {
    pub async fn load(model: &ModelConfig, cache: &Path) -> Result<Self> {
        let dir = cache.join(&model.revision);
        std::fs::create_dir_all(&dir)?;
        let client = reqwest::Client::builder()
            .user_agent("Switchyard/0.1")
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|_| Error::Request("tokenizer client".into()))?;
        for file in ["tokenizer.json", "chat_template.jinja"] {
            let path = dir.join(file);
            if !path.exists() {
                let url = format!(
                    "https://huggingface.co/{}/resolve/{}/{file}",
                    model.repository, model.revision
                );
                let r = client
                    .get(url)
                    .send()
                    .await
                    .map_err(|_| Error::Request(format!("{file} download")))?;
                if !r.status().is_success() {
                    return Err(Error::Request(format!(
                        "{file} HTTP {}",
                        r.status().as_u16()
                    )));
                }
                let b = r
                    .bytes()
                    .await
                    .map_err(|_| Error::Request(format!("{file} download")))?;
                let temp = path.with_extension("partial");
                std::fs::write(&temp, b)?;
                std::fs::rename(temp, path)?;
            }
        }
        Self::from_files(
            &dir.join("tokenizer.json"),
            &dir.join("chat_template.jinja"),
        )
    }
    /// Load already pinned local tokenizer assets, including offline test fixtures.
    pub fn from_files(tokenizer: &Path, template: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer)
            .map_err(|_| Error::Config("invalid tokenizer.json".into()))?;
        Ok(Self {
            tokenizer: std::sync::Arc::new(tokenizer),
            template: std::fs::read_to_string(template)?,
        })
    }
    pub fn input_tokens(&self, messages: &[Message]) -> Result<u64> {
        if messages.is_empty()
            || messages
                .iter()
                .any(|m| !matches!(m.role.as_str(), "system" | "user" | "assistant"))
        {
            return Err(Error::Config(
                "text messages require system, user, or assistant roles".into(),
            ));
        }
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env.add_template("chat", &self.template)
            .map_err(|_| Error::Config("unsupported model chat template".into()))?;
        let rendered=env.get_template("chat").map_err(|_|Error::Config("chat template missing".into()))?
            .render(minijinja::context! {messages=>messages,add_generation_prompt=>true,enable_thinking=>false,tools=>false,add_vision_id=>false})
            .map_err(|e|Error::Config(format!("chat template rendering: {e}")))?;
        Ok(self
            .tokenizer
            .encode(rendered, false)
            .map_err(|_| Error::Config("prompt tokenization".into()))?
            .len() as u64)
    }
    pub fn output_tokens(&self, text: &str) -> u64 {
        self.tokenizer
            .encode(text, false)
            .map_or(0, |e| e.len() as u64)
    }
}
