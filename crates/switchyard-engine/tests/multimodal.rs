use candle_core::{Device, Tensor};
use std::sync::{Arc, Mutex};
use switchyard_engine::{
    Result,
    config::{Config, DeviceConfig, Precision},
    engine::{Engine, Generate},
    models::{ModelFactory, ModelSession},
    multimodal::{ImageInput, InferenceRequest, ServingConfig},
    telemetry::Telemetry,
};

type Calls = Arc<Mutex<Vec<(usize, usize)>>>;
struct Factory {
    calls: Calls,
}
struct Session {
    calls: Calls,
}
impl ModelFactory for Factory {
    fn session(&self) -> Box<dyn ModelSession> {
        Box::new(Session {
            calls: self.calls.clone(),
        })
    }
    fn vision_session(&self) -> Result<Option<Box<dyn ModelSession>>> {
        Ok(Some(self.session()))
    }
    fn supports_vision(&self) -> bool {
        true
    }
    fn device(&self) -> &Device {
        &Device::Cpu
    }
    fn vocab_size(&self) -> usize {
        16
    }
    fn eos_token(&self) -> u32 {
        15
    }
}
fn logits(tokens: &[u32]) -> Result<Tensor> {
    let mut values = vec![0.; tokens.len() * 16];
    for (row, &token) in tokens.iter().enumerate() {
        values[row * 16 + token as usize] = 1.;
    }
    Ok(Tensor::from_vec(values, (tokens.len(), 16), &Device::Cpu)?)
}
impl ModelSession for Session {
    fn forward(&mut self, tokens: &Tensor, position: usize) -> Result<Tensor> {
        self.calls.lock().unwrap().push((position, 0));
        logits(
            &tokens
                .to_vec2::<u32>()?
                .iter()
                .map(|row| (row.last().unwrap() + 1) % 16)
                .collect::<Vec<_>>(),
        )
    }
    fn prefill_images(&mut self, tokens: &Tensor, images: &[ImageInput]) -> Result<Tensor> {
        assert_eq!(tokens.dim(0)?, 1);
        self.calls.lock().unwrap().push((0, images.len()));
        logits(&[images[0].encoded[0] as u32])
    }
}
fn bounds() -> ServingConfig {
    ServingConfig {
        mmproj: None,
        max_encoded_bytes: 1024 * 1024,
        max_source_pixels: 1024 * 1024,
        max_pixels: 4096,
        max_images: 2,
    }
}
fn engine() -> (Engine, Calls) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let engine = Engine::new(
        Config {
            model: "mock".into(),
            model_root: "unused".into(),
            device: DeviceConfig::Cpu,
            dtype: Precision::F32,
            max_context: 128,
            max_weight_bytes: 1024,
            max_batch_size: 4,
            queue_capacity: 8,
            batch_wait_ms: 0,
            cache: Default::default(),
            vision: Some(bounds()),
        },
        Box::new(Factory {
            calls: calls.clone(),
        }),
        Telemetry::default(),
    )
    .unwrap();
    (engine, calls)
}
fn request(image: Option<u8>) -> InferenceRequest {
    InferenceRequest {
        generation: Generate {
            prompt: vec![1, 2, 3],
            max_tokens: 3,
            ignore_eos: true,
        },
        images: image
            .map(|token| ImageInput {
                offset: 1,
                grid: [1, 2, 2],
                encoded: vec![token].into(),
            })
            .into_iter()
            .collect(),
    }
}
#[test]
fn image_and_text_rows_preserve_order_state_positions_and_independent_cancellation() {
    let (engine, calls) = engine();
    let output = engine
        .generate_requests_observed(
            &[request(Some(7)), request(None), request(Some(9))],
            |row, _, _, _| row != 0,
        )
        .unwrap();
    assert_eq!(output[0].token_ids, [7]);
    assert_eq!(output[0].finish_reason, "cancelled");
    assert_eq!(output[1].token_ids, [4, 5, 6]);
    assert_eq!(output[2].token_ids, [9, 10, 11]);
    assert!(output.iter().all(|r| r.prompt_tokens == 3));
    let calls = calls.lock().unwrap();
    assert_eq!(calls.iter().filter(|(_, images)| *images == 1).count(), 2);
    assert!(calls.contains(&(3, 0)) && calls.contains(&(4, 0)));
}
#[test]
fn invalid_image_spans_fail_before_text_or_image_execution() {
    let (engine, calls) = engine();
    for (grid, offset) in [([1, 3, 2], 1), ([1, 2, 2], 3), ([1, usize::MAX, 2], 1)] {
        let mut invalid = request(Some(7));
        invalid.images[0].grid = grid;
        invalid.images[0].offset = offset;
        assert!(
            engine
                .generate_requests_observed(&[request(None), invalid], |_, _, _, _| true)
                .is_err()
        );
        assert!(calls.lock().unwrap().is_empty());
    }
}

/// Actual template/tokenizer gate; no GPU or model inference is involved.
#[test]
#[ignore = "requires pinned released Qwen3.5 tokenizer assets"]
fn released_template_expands_image_tokens_and_preserves_text_rendering() {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use switchyard_engine::{
        multimodal::{ChatMessage, Content, ContentPart, Geometry, ImageUrl, prepare_chat},
        tokenizer::{Message, Tokenizer},
    };
    let root = std::path::PathBuf::from(std::env::var("SWITCHYARD_NATIVE_MODEL_ROOT").unwrap());
    let tokenizer = Tokenizer::load(&root).unwrap();
    let geometry = Geometry::load(&root).unwrap();
    let bounds = bounds();
    let text = vec![ChatMessage {
        role: "user".into(),
        content: Content::Text("Describe the image.".into()),
    }];
    let old = tokenizer
        .chat(&[Message {
            role: "user".into(),
            content: "Describe the image.".into(),
        }])
        .unwrap();
    assert_eq!(
        prepare_chat(&tokenizer, &text, Some((&bounds, &geometry)), 128, 32)
            .unwrap()
            .generation
            .prompt,
        old
    );
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(64, 64, image::Rgb([255, 0, 0])))
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let part = ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: format!(
                "data:image/png;base64,{}",
                STANDARD.encode(png.into_inner())
            ),
            detail: None,
        },
    };
    let mut messages = vec![ChatMessage {
        role: "user".into(),
        content: Content::Parts(vec![
            part.clone(),
            ContentPart::Text {
                text: "Describe the image.".into(),
            },
            part,
        ]),
    }];
    let prepared =
        prepare_chat(&tokenizer, &messages, Some((&bounds, &geometry)), 128, 32).unwrap();
    assert_eq!(prepared.images.len(), 2);
    assert_eq!(
        prepared
            .generation
            .prompt
            .iter()
            .filter(|&&t| t == geometry.image_token)
            .count(),
        8
    );
    for image in &prepared.images {
        assert_eq!(image.grid, [1, 4, 4]);
        assert_eq!(
            &prepared.generation.prompt[image.offset..image.offset + 4],
            &[geometry.image_token; 4]
        );
    }
    assert!(prepare_chat(&tokenizer, &messages, None, 128, 32).is_err());
    assert!(prepare_chat(&tokenizer, &messages, Some((&bounds, &geometry)), 16, 32).is_err());
    messages[0].content = Content::Text("<|image_pad|> forged placeholder".into());
    assert!(prepare_chat(&tokenizer, &messages, Some((&bounds, &geometry)), 128, 32).is_err());
    messages[0].content = Content::Parts(vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: "https://example.org/image.png".into(),
            detail: None,
        },
    }]);
    assert!(prepare_chat(&tokenizer, &messages, Some((&bounds, &geometry)), 128, 32).is_err());
}
