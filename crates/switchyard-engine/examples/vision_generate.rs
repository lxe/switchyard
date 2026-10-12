//! Offline token-level image generation for an independent HF oracle.
use candle_core::{DType, Device, Tensor};
use std::{path::PathBuf, time::Instant};
use switchyard_engine::{
    models::{qwen35, vision},
    telemetry::Telemetry,
    weights::Weights,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        return Err("MODEL_ROOT MMPROJ IMAGE REQUEST_JSON OUTPUT_JSON [cuda]".into());
    }
    let root = PathBuf::from(&a[1]);
    let mut config: qwen35::Config =
        serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    config.text_config.max_position_embeddings = 256;
    let device = if a.get(6).is_some_and(|x| x == "cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let input: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[4])?)?;
    let prompt: Vec<u32> = serde_json::from_value(input["prompt"].clone())?;
    let count = input["max_tokens"].as_u64().ok_or("max_tokens required")? as usize;
    if count == 0 || count > 64 || prompt.len() + count > 256 {
        return Err("Fixed generation bound".into());
    }
    let offset = input["image_offset"]
        .as_u64()
        .ok_or("image_offset required")? as usize;
    let vb = Weights::read(&root, 4 << 30, &Telemetry::default())?.builder(DType::F32, &device);
    let cfg = config
        .vision_config
        .as_ref()
        .ok_or("Missing vision config")?;
    let vision = if a[2] == "-" {
        vision::VisionModel::new(cfg, vb.pp("model.visual"))?
    } else {
        vision::VisionModel::from_gguf(cfg, std::path::Path::new(&a[2]), &device, DType::F32)?
    };
    let image = vision::prepare_image(&std::fs::read(&a[3])?, 65_536, &device)?;
    let embedding = vision.forward(&image)?;
    let mut model = qwen35::ModelForCausalLM::new(&config, vb)?;
    let start = Instant::now();
    let (mut logits, delta) = model.prefill_images(
        &Tensor::from_slice(&prompt, (1, prompt.len()), &device)?,
        &[qwen35::ImageSpan {
            offset,
            grid: image.grid,
            embeddings: embedding,
        }],
    )?;
    let mut ids = Vec::new();
    let mut timings = Vec::new();
    let mut top = Vec::new();
    for step in 0..count {
        let values = logits.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        let (token, value) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("Empty logits")?;
        ids.push(token as u32);
        top.push(*value);
        timings.push(start.elapsed().as_secs_f64());
        if step + 1 < count {
            logits = model.forward_with_rope_offset(
                &Tensor::new(&[[token as u32]], &device)?,
                prompt.len() + step,
                delta,
            )?;
        }
    }
    std::fs::write(
        &a[5],
        serde_json::to_vec_pretty(
            &serde_json::json!({"token_ids":ids,"argmax_logits":top,"timings_s":timings,"rope_delta":delta,"prompt_tokens":prompt.len(),"grid":image.grid}),
        )?,
    )?;
    Ok(())
}
