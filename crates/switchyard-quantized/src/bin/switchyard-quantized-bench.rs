//! Offline CPU projection benchmark; neither a serving endpoint nor model-quality test.
use candle_core::quantized::{QMatMul, QTensor, gguf_file};
use candle_core::{Device, Module, Tensor};
use clap::Parser;
use std::{io::Cursor, sync::Arc, time::Instant};
use switchyard_quantized::{
    gguf::{Encoding, Index, IndexLimits},
    projection::ProjectionGroup,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 512)]
    width: usize,
    #[arg(long, default_value_t = 256)]
    output_width: usize,
    #[arg(long, value_delimiter = ',', default_value = "1,8,64")]
    rows: Vec<usize>,
    #[arg(long, default_value_t = 2)]
    encoding_code: u32,
    #[arg(long, default_value_t = 20)]
    iterations: usize,
    #[arg(long, default_value_t = 5)]
    samples: usize,
}
fn elapsed(
    iterations: usize,
    mut operation: impl FnMut() -> candle_core::Result<Vec<Tensor>>,
) -> candle_core::Result<f64> {
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(operation()?);
    }
    Ok(start.elapsed().as_secs_f64() / iterations as f64)
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let encoding = Encoding::from_code(args.encoding_code)?;
    if !encoding.quantized()
        || args.width == 0
        || args.width > 8192
        || args.output_width == 0
        || args.output_width > 8192
        || !args.width.is_multiple_of(encoding.backend().block_size())
        || args.iterations == 0
        || args.samples == 0
        || args.samples > 100
        || args.rows.is_empty()
        || args.rows.iter().any(|&n| n == 0 || n > 1024)
    {
        return Err("invalid CPU benchmark dimensions or repetition count".into());
    }
    let make = |seed: usize| -> candle_core::Result<QTensor> {
        let values = (0..args.output_width * args.width)
            .map(|i| ((i * 17 + seed * 37) as f32 * 0.071).sin())
            .collect::<Vec<_>>();
        QTensor::quantize(
            &Tensor::from_vec(values, (args.output_width, args.width), &Device::Cpu)?,
            encoding.backend(),
        )
    };
    let q = [make(3)?, make(9)?];
    let mut file = Cursor::new(Vec::new());
    gguf_file::write(&mut file, &[], &[("gate", &q[0]), ("up", &q[1])])?;
    file.get_mut()[4..8].copy_from_slice(&3u32.to_le_bytes());
    let index = Index::read(&mut file, IndexLimits::default())?;
    let grouped =
        ProjectionGroup::read(&mut file, &index, &["gate", "up"], 64 << 20, &Device::Cpu)?;
    let independent = q
        .into_iter()
        .map(|q| QMatMul::QTensor(Arc::new(q)))
        .collect::<Vec<_>>();
    let mut cases = Vec::new();
    for rows in args.rows {
        let input = Tensor::from_vec(
            (0..rows * args.width)
                .map(|i| (i as f32 * 0.013).cos())
                .collect::<Vec<_>>(),
            (rows, args.width),
            &Device::Cpu,
        )?;
        let reference = independent
            .iter()
            .map(|op| op.forward(&input))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let actual = grouped.forward(&input)?;
        let mut max_error = 0f32;
        for (expected, actual) in reference.iter().zip(&actual) {
            max_error = max_error.max((expected - actual)?.abs()?.max_all()?.to_scalar::<f32>()?);
        }
        let single_rows = (0..rows)
            .map(|r| input.narrow(0, r, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        for (r, row) in single_rows.iter().enumerate() {
            for (op, expected) in independent.iter().zip(&reference) {
                let single = op.forward(row)?;
                max_error = max_error.max(
                    (expected.narrow(0, r, 1)? - &single)?
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?,
                );
            }
        }
        if max_error > 1e-5 {
            return Err(format!("CPU projection max abs error {max_error} exceeds 1e-5").into());
        }
        for _ in 0..3 {
            std::hint::black_box(grouped.forward(&input)?);
            for op in &independent {
                std::hint::black_box(op.forward(&input)?);
            }
        }
        let mut per_row = Vec::new();
        let mut batch = Vec::new();
        let mut fusion = Vec::new();
        for sample in 0..args.samples {
            // Alternate order to reduce a consistent temperature/order bias.
            for mode in if sample % 2 == 0 {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            } {
                match mode {
                    0 => per_row.push(elapsed(args.iterations, || {
                        single_rows
                            .iter()
                            .flat_map(|row| independent.iter().map(|op| op.forward(row)))
                            .collect()
                    })?),
                    1 => batch.push(elapsed(args.iterations, || {
                        independent.iter().map(|op| op.forward(&input)).collect()
                    })?),
                    _ => fusion.push(elapsed(args.iterations, || {
                        grouped
                            .forward(&input)
                            .map_err(|error| candle_core::Error::Msg(error.to_string()))
                    })?),
                }
            }
        }
        cases.push(serde_json::json!({"rows":rows,"max_abs_difference":max_error,
            "per_row_separate_seconds":median(&per_row),"batched_separate_seconds":median(&batch),
            "batched_grouped_seconds":median(&fusion),
            "samples":{"per_row_separate":per_row,"batched_separate":batch,"batched_grouped":fusion}}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope":"CPU matrix projections only; not model token throughput or quality",
            "candle_revision":"c68b24997319b8d76e5a4e775dab197c7f26322a",
            "encoding":encoding,"width":args.width,"output_width":args.output_width,
            "iterations":args.iterations,"sample_count":args.samples,"packed_bytes":grouped.packed_bytes(),
            "rayon_threads_env":std::env::var("RAYON_NUM_THREADS").ok(),"max_abs_tolerance":1e-5,
            "cases":cases
        }))?
    );
    Ok(())
}
