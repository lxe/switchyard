//! Request-owned image inputs. Transport parsing never allocates CUDA state.
use crate::{Error, Result, engine::Generate, tokenizer::Tokenizer};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::{io::Cursor, path::PathBuf, sync::Arc};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServingConfig {
    /// Omit for the released HF vision tensors in model_root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<PathBuf>,
    pub max_encoded_bytes: usize,
    pub max_source_pixels: usize,
    pub max_pixels: usize,
    pub max_images: usize,
}
impl ServingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_encoded_bytes == 0
            || self.max_encoded_bytes > 8 * 1024 * 1024
            || self.max_source_pixels == 0
            || self.max_source_pixels > 16 * 1024 * 1024
            || self.max_pixels < 32 * 32
            || self.max_pixels > 262_144
            || !(1..=4).contains(&self.max_images)
        {
            return Err(Error::Config("Invalid fixed image serving bounds".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ImageInput {
    /// First expanded image-pad token in the rendered prompt.
    pub offset: usize,
    pub grid: [usize; 3],
    pub encoded: Arc<[u8]>,
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub generation: Generate,
    pub images: Vec<ImageInput>,
}
impl From<Generate> for InferenceRequest {
    fn from(generation: Generate) -> Self {
        Self {
            generation,
            images: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: Content,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
    /// The fixed server image budget determines resolution.
    #[serde(default)]
    pub detail: Option<String>,
}

/// Declared image layout of the released model, separate from serving bounds.
#[derive(Debug, Clone)]
pub struct Geometry {
    pub patch_size: usize,
    pub merge_size: usize,
    pub image_token: u32,
}
impl Geometry {
    pub fn load(root: &std::path::Path) -> Result<Self> {
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        let get = |value: &serde_json::Value, key| -> Result<usize> {
            value[key]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| Error::Config(format!("Missing model vision field {key}")))
        };
        let vision = &raw["vision_config"];
        let result = Self {
            patch_size: get(vision, "patch_size")?,
            merge_size: get(vision, "spatial_merge_size")?,
            image_token: u32::try_from(get(&raw, "image_token_id")?)
                .map_err(|_| Error::Config("Image token ID overflow".into()))?,
        };
        if result.patch_size != 16 || result.merge_size != 2 {
            return Err(Error::Config("Unsupported image patch geometry".into()));
        }
        Ok(result)
    }
    /// Qwen smart resize to multiples of patch_size * spatial_merge_size.
    pub fn grid(&self, width: usize, height: usize, max_pixels: usize) -> Result<[usize; 3]> {
        let width =
            u32::try_from(width).map_err(|_| Error::Config("Image width overflow".into()))?;
        let height =
            u32::try_from(height).map_err(|_| Error::Config("Image height overflow".into()))?;
        Ok(crate::models::vision::resize_grid(
            width, height, max_pixels,
        )?)
    }
    pub fn tokens(&self, grid: [usize; 3]) -> Result<usize> {
        if self.merge_size == 0
            || grid[0] != 1
            || grid[1] == 0
            || grid[2] == 0
            || !grid[1].is_multiple_of(self.merge_size)
            || !grid[2].is_multiple_of(self.merge_size)
        {
            return Err(Error::Config("Invalid image grid".into()));
        }
        grid[0]
            .checked_mul(grid[1] / self.merge_size)
            .and_then(|n| n.checked_mul(grid[2] / self.merge_size))
            .ok_or_else(|| Error::Config("Image token count overflow".into()))
    }
}

fn image_bytes(url: &ImageUrl, bounds: &ServingConfig) -> Result<Arc<[u8]>> {
    if url.detail.as_deref().is_some_and(|s| s != "auto") {
        return Err(Error::Config(
            "Image resolution uses fixed serving bounds; detail must be auto".into(),
        ));
    }
    let payload = url
        .url
        .strip_prefix("data:image/png;base64,")
        .or_else(|| url.url.strip_prefix("data:image/jpeg;base64,"))
        .ok_or_else(|| {
            Error::Config("Images require an inline PNG or JPEG base64 data URL".into())
        })?;
    let encoded_limit = bounds.max_encoded_bytes.div_ceil(3) * 4;
    if payload.len() > encoded_limit {
        return Err(Error::Config(
            "Encoded image exceeds serving byte bound".into(),
        ));
    }
    let bytes = STANDARD
        .decode(payload)
        .map_err(|_| Error::Config("Invalid image base64".into()))?;
    if bytes.is_empty() || bytes.len() > bounds.max_encoded_bytes {
        return Err(Error::Config(
            "Encoded image exceeds serving byte bound".into(),
        ));
    }
    Ok(bytes.into())
}

pub fn prepare_chat(
    tokenizer: &Tokenizer,
    messages: &[ChatMessage],
    vision: Option<(&ServingConfig, &Geometry)>,
    max_context: usize,
    max_tokens: usize,
) -> Result<InferenceRequest> {
    if let Some((bounds, _)) = vision {
        bounds.validate()?;
    }
    if messages.is_empty()
        || messages
            .iter()
            .any(|m| !matches!(m.role.as_str(), "system" | "user" | "assistant"))
    {
        return Err(Error::Config(
            "Expected nonempty supported chat roles".into(),
        ));
    }
    let mut inputs = Vec::new();
    for message in messages {
        if let Content::Parts(parts) = &message.content {
            for part in parts {
                if let ContentPart::ImageUrl { image_url } = part {
                    let (bounds, geometry) =
                        vision.ok_or_else(|| Error::Config("Vision serving is disabled".into()))?;
                    if message.role != "user" || inputs.len() >= bounds.max_images {
                        return Err(Error::Config(
                            "Images require a user message within the fixed image count".into(),
                        ));
                    }
                    let encoded = image_bytes(image_url, bounds)?;
                    let reader = image::ImageReader::new(Cursor::new(encoded.as_ref()))
                        .with_guessed_format()
                        .map_err(|_| Error::Config("Invalid image header".into()))?;
                    if !matches!(
                        reader.format(),
                        Some(image::ImageFormat::Png | image::ImageFormat::Jpeg)
                    ) {
                        return Err(Error::Config(
                            "Only PNG and JPEG images are supported".into(),
                        ));
                    }
                    let (width, height) = reader
                        .into_dimensions()
                        .map_err(|_| Error::Config("Invalid image dimensions".into()))?;
                    if (width as usize)
                        .checked_mul(height as usize)
                        .is_none_or(|n| n == 0 || n > bounds.max_source_pixels)
                    {
                        return Err(Error::Config(
                            "Source image exceeds fixed pixel bound".into(),
                        ));
                    }
                    let grid = geometry.grid(width as usize, height as usize, bounds.max_pixels)?;
                    inputs.push(ImageInput {
                        offset: 0,
                        grid,
                        encoded,
                    });
                }
            }
        }
    }
    let tokens = tokenizer.chat_parts(messages)?;
    let mut prompt = Vec::new();
    let mut images = inputs.into_iter();
    let mut prepared = Vec::new();
    for token in tokens {
        if vision.is_some_and(|(_, geometry)| token == geometry.image_token) {
            let mut image = images.next().ok_or_else(|| {
                Error::Config("Image placeholders must originate from image parts".into())
            })?;
            let count = vision.unwrap().1.tokens(image.grid)?;
            image.offset = prompt.len();
            if prompt
                .len()
                .checked_add(count)
                .is_none_or(|n| n > max_context)
            {
                return Err(Error::Config(
                    "Expanded image prompt exceeds context bound".into(),
                ));
            }
            prompt.extend(std::iter::repeat_n(token, count));
            prepared.push(image);
        } else {
            if prompt.len() == max_context {
                return Err(Error::Config("Prompt exceeds context bound".into()));
            }
            prompt.push(token);
        }
    }
    if images.next().is_some()
        || prompt
            .len()
            .checked_add(max_tokens)
            .is_none_or(|n| n > max_context)
    {
        return Err(Error::Config(
            "Image placeholders or context bounds do not match".into(),
        ));
    }
    Ok(InferenceRequest {
        generation: Generate {
            prompt,
            max_tokens,
            ignore_eos: false,
        },
        images: prepared,
    })
}
