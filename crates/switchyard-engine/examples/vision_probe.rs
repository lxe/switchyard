//! Offline image-encoder oracle: cargo run -p switchyard-engine --example vision_probe
//! -- MODEL_ROOT MMPROJ IMAGE OUTPUT_DIR [cuda]. Use `-` for the native HF vision weights.
use candle_core::{DType, Device};
use std::{path::PathBuf, time::Instant};
use switchyard_engine::{
    models::{qwen35, vision},
    telemetry::Telemetry,
    weights::Weights,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err("MODEL_ROOT MMPROJ IMAGE OUTPUT_DIR [cuda]".into());
    }
    let root = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[4]);
    std::fs::create_dir_all(&output)?;
    let config: qwen35::Config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    let config = config.vision_config.ok_or("Missing vision configuration")?;
    let device = if args.get(5).is_some_and(|x| x == "cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let start = Instant::now();
    let model = if args[2] == "-" {
        vision::VisionModel::new(
            &config,
            Weights::read(&root, 4 << 30, &Telemetry::default())?
                .builder(DType::F32, &device)
                .pp("model.visual"),
        )?
    } else {
        vision::VisionModel::from_gguf(
            &config,
            std::path::Path::new(&args[2]),
            &device,
            DType::F32,
        )?
    };
    let load_s = start.elapsed().as_secs_f64();
    let image = vision::prepare_image(&std::fs::read(&args[3])?, 65_536, &device)?;
    image
        .pixels
        .to_device(&Device::Cpu)?
        .write_npy(output.join("native-pixels.npy"))?;
    let start = Instant::now();
    let embedding = model.forward(&image)?;
    device.synchronize()?;
    let forward_s = start.elapsed().as_secs_f64();
    embedding
        .to_device(&Device::Cpu)?
        .write_npy(output.join("native-embeddings.npy"))?;
    std::fs::write(
        output.join("native-image.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"grid":image.grid,"shape":embedding.dims(),"load_s":load_s,"forward_s":forward_s,"dtype":"f32"}),
        )?,
    )?;
    Ok(())
}
