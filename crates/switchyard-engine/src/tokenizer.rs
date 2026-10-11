use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: String,
}
#[derive(Clone)]
pub struct Tokenizer {
    pub(crate) tokenizer: tokenizers::Tokenizer,
    template: String,
}
impl Tokenizer {
    pub fn load(root: &Path) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json"))
            .map_err(|e| Error::Config(e.to_string()))?;
        Ok(Self {
            tokenizer,
            template: std::fs::read_to_string(root.join("chat_template.jinja"))?,
        })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|x| x.get_ids().to_vec())
            .map_err(|e| Error::Config(e.to_string()))
    }
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| Error::Config(e.to_string()))
    }
    pub fn chat(&self, messages: &[Message]) -> Result<Vec<u32>> {
        if messages.is_empty()
            || messages
                .iter()
                .any(|m| !matches!(m.role.as_str(), "system" | "user" | "assistant"))
        {
            return Err(Error::Config("Expected nonempty text chat messages".into()));
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
            .map_err(|e| Error::Config(e.to_string()))?;
        let text = env.get_template("chat").map_err(|e| Error::Config(e.to_string()))?
            .render(minijinja::context!{ messages=>messages, add_generation_prompt=>true, enable_thinking=>false, tools=>false, add_vision_id=>false })
            .map_err(|e| Error::Config(e.to_string()))?;
        self.encode(&text)
    }
}
