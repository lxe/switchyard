use super::support;
use candle_core::{
    Device, Module, Tensor,
    quantized::{GgmlDType, QMatMul, ggml_file::qtensor_from_ggml, gguf_file::Value},
};
use std::io::{Read, Seek, SeekFrom};
use switchyard_quantized::{
    gguf::{Index, IndexLimits},
    projection::ProjectionGroup,
};

#[test]
fn independent_q4_block_fixture_preserves_row_order() {
    let mut payload = vec![0, 0x3c];
    payload.extend([0x98; 16]);
    payload.extend([0, 0x38]);
    payload.extend([0x7a; 16]);
    let mut file = support::raw_fixture(
        &[("general.architecture", Value::String("qwen35".into()))],
        &[("matrix", &[32, 2], 2, 0)],
        &payload,
    );
    let index = Index::read(&mut file, IndexLimits::default()).unwrap();
    index.require_architecture("qwen35").unwrap();
    assert!(index.require_architecture("qwen38").is_err());
    assert_eq!(index.tensor("matrix").unwrap().backend_shape(), [2, 32]);
    let group = ProjectionGroup::read(&mut file, &index, &["matrix"], 36, &Device::Cpu).unwrap();
    let out = group
        .forward(&Tensor::ones((1, 32), candle_core::DType::F32, &Device::Cpu).unwrap())
        .unwrap();
    let values = out[0].to_vec2::<f32>().unwrap();
    let dense = qtensor_from_ggml(GgmlDType::Q4_0, &payload, vec![2, 32], &Device::Cpu)
        .unwrap()
        .dequantize(&Device::Cpu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(&dense[0][..16], &[0.; 16]);
    assert_eq!(&dense[0][16..], &[1.; 16]);
    assert_eq!(&dense[1][..16], &[1.; 16]);
    assert_eq!(&dense[1][16..], &[-0.5; 16]);
    // Candle's CPU Q4 dot quantizes the lhs as Q8_0. The all-ones scale
    // 1/127 rounds to FP16 129/16384, so this is the packed-kernel oracle.
    let lhs_scale = 127. * 129. / 16_384.;
    assert_eq!(values, vec![vec![16. * lhs_scale, 8. * lhs_scale]]);
}

#[test]
fn header_admission_does_not_read_tensor_payload() {
    struct Count {
        inner: std::io::Cursor<Vec<u8>>,
        bytes: usize,
    }
    impl Read for Count {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(b)?;
            self.bytes += n;
            Ok(n)
        }
    }
    impl Seek for Count {
        fn seek(&mut self, p: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(p)
        }
    }
    let file = support::raw_fixture(&[], &[("large", &[32, 1 << 20], 2, 0)], &vec![0; 18 << 20]);
    let mut counted = Count {
        inner: file,
        bytes: 0,
    };
    let index = Index::read(&mut counted, IndexLimits::default()).unwrap();
    assert_eq!(counted.bytes as u64, index.header_bytes());
    assert!(counted.bytes < 128);
}

#[test]
fn malformed_dimensions_types_offsets_and_lengths_are_rejected() {
    for tensors in [
        vec![("bad", &[16, 2][..], 2, 0)], // divisible total, invalid row block boundary
        vec![("bad", &[32, 1][..], 16, 0)], // unsupported IQ encoding
        vec![("bad", &[u64::MAX, 2][..], 0, 0)],
        vec![("bad", &[32, 1][..], 2, 1)],
        vec![("bad", &[32, 1][..], 2, 1024)],
        vec![("a", &[32, 1][..], 2, 0), ("b", &[32, 1][..], 2, 0)],
        vec![("a", &[32, 1][..], 2, 0), ("a", &[32, 1][..], 2, 32)],
    ] {
        let mut file = support::raw_fixture(&[], &tensors, &[0; 128]);
        assert!(
            Index::read(&mut file, IndexLimits::default()).is_err(),
            "{tensors:?}"
        );
    }
    let mut invalid_alignment = support::raw_fixture(
        &[("general.alignment", Value::U32(0))],
        &[("a", &[32, 1], 2, 0)],
        &[0; 18],
    );
    assert!(Index::read(&mut invalid_alignment, IndexLimits::default()).is_err());
    let mut truncated = support::raw_fixture(&[], &[("a", &[32, 1], 2, 0)], &[0; 17]);
    assert!(Index::read(&mut truncated, IndexLimits::default()).is_err());
}

#[test]
fn metadata_limits_are_checked_before_large_allocation() {
    let tensors = [("w", support::matrix(2, 32, 3, GgmlDType::Q4_0))];
    let metadata = [("array", Value::Array(vec![Value::U32(1); 16]))];
    let mut file = support::fixture(&metadata, &tensors);
    assert!(
        Index::read(
            &mut file,
            IndexLimits {
                values: 8,
                ..Default::default()
            }
        )
        .is_err()
    );
    let mut file = support::fixture(
        &[("text", Value::String("abcdefghijklmnop".into()))],
        &tensors,
    );
    assert!(
        Index::read(
            &mut file,
            IndexLimits {
                string_bytes: 8,
                ..Default::default()
            }
        )
        .is_err()
    );
    let mut file = support::fixture(&[], &tensors);
    assert!(
        Index::read(
            &mut file,
            IndexLimits {
                header_bytes: 32,
                ..Default::default()
            }
        )
        .is_err()
    );
    let nested = Value::Array(vec![Value::Array(vec![Value::U32(1)])]);
    let mut file = support::fixture(&[("nested", nested)], &tensors);
    assert!(
        Index::read(
            &mut file,
            IndexLimits {
                array_depth: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    let mut invalid_boolean = support::fixture(&[("bool", Value::Bool(true))], &tensors);
    // Fixed header24 + string prefix8 + key4 + type4 = boolean byte40.
    invalid_boolean.get_mut()[40] = 2;
    assert!(Index::read(&mut invalid_boolean, IndexLimits::default()).is_err());
}

#[test]
fn bounded_header_corruption_never_panics() {
    let original = support::raw_fixture(
        &[("general.architecture", Value::String("qwen35".into()))],
        &[("w", &[32, 2], 2, 0)],
        &[0; 36],
    )
    .into_inner();
    let limits = IndexLimits {
        header_bytes: 4096,
        string_bytes: 256,
        metadata_entries: 64,
        tensors: 64,
        values: 64,
        array_depth: 2,
    };
    let mut random = 7u64;
    for _ in 0..512 {
        let mut bytes = original.clone();
        for _ in 0..3 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let index = random as usize % bytes.len();
            bytes[index] ^= (random >> 32) as u8;
        }
        assert!(
            std::panic::catch_unwind(|| Index::read(&mut std::io::Cursor::new(bytes), limits))
                .is_ok()
        );
    }
}

#[test]
fn grouped_projection_rejects_mixed_encodings_widths_and_dense_weights() {
    let tensors = [
        ("q4", support::matrix(32, 256, 3, GgmlDType::Q4_0)),
        ("q8", support::matrix(32, 256, 9, GgmlDType::Q8_0)),
        ("narrow", support::matrix(32, 128, 9, GgmlDType::Q4_0)),
        ("dense", support::matrix(32, 256, 9, GgmlDType::F32)),
    ];
    let mut file = support::fixture(&[], &tensors);
    let index = Index::read(&mut file, IndexLimits::default()).unwrap();
    for names in [&["q4", "q8"][..], &["q4", "narrow"][..], &["dense"][..]] {
        assert!(ProjectionGroup::read(&mut file, &index, names, 1 << 20, &Device::Cpu).is_err());
    }
    let group = ProjectionGroup::read(&mut file, &index, &["q4"], 1 << 20, &Device::Cpu).unwrap();
    let input = Tensor::zeros((2, 256), candle_core::DType::F32, &Device::Cpu).unwrap();
    assert!(group.forward(&input.t().unwrap()).is_err());
    assert!(
        group
            .forward(&input.to_dtype(candle_core::DType::F64).unwrap())
            .is_err()
    );
}

#[test]
fn fused_projection_matches_independent_calls_for_batch_and_prefill() {
    for (encoding, rmse_limit) in [
        (GgmlDType::Q4_0, 0.08),
        (GgmlDType::Q8_0, 0.006),
        (GgmlDType::Q4K, 0.1),
        (GgmlDType::Q6K, 0.02),
    ] {
        let tensors = [
            ("gate", support::matrix(32, 256, 3, encoding)),
            ("up", support::matrix(48, 256, 9, encoding)),
        ];
        let mut file = support::fixture(&[], &tensors);
        let index = Index::read(&mut file, IndexLimits::default()).unwrap();
        let group =
            ProjectionGroup::read(&mut file, &index, &["up", "gate"], 1 << 20, &Device::Cpu)
                .unwrap();
        assert_eq!(group.output_names().collect::<Vec<_>>(), ["up", "gate"]);
        for dimensions in [vec![5, 256], vec![2, 3, 256]] {
            let values = (0..dimensions.iter().product())
                .map(|i| (i as f32 * 0.013).cos())
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values, dimensions.as_slice(), &Device::Cpu).unwrap();
            let fused = group.forward(&input).unwrap();
            for (actual, name) in fused.iter().zip(["up", "gate"]) {
                let descriptor = index.tensor(name).unwrap();
                let mut raw = vec![0; descriptor.bytes() as usize];
                file.seek(SeekFrom::Start(descriptor.file_offset()))
                    .unwrap();
                file.read_exact(&mut raw).unwrap();
                let quantized =
                    qtensor_from_ggml(encoding, &raw, descriptor.backend_shape(), &Device::Cpu)
                        .unwrap();
                let independent = QMatMul::QTensor(std::sync::Arc::new(quantized))
                    .forward(&input)
                    .unwrap();
                let error = (actual - &independent)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(
                    error <= 1e-5,
                    "{encoding:?} grouped output max abs difference {error}"
                );
            }
        }
        // Local matrix reconstruction tolerance, not a model-quality claim.
        for ((_, qtensor), seed) in tensors.iter().zip([3, 9]) {
            let dense = qtensor
                .dequantize(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let rmse = (dense
                .iter()
                .enumerate()
                .map(|(i, &v)| (v - ((i * 17 + seed * 37) as f32 * 0.071).sin()).powi(2) as f64)
                .sum::<f64>()
                / dense.len() as f64)
                .sqrt();
            assert!(
                rmse < rmse_limit,
                "{encoding:?}: weight reconstruction RMSE {rmse}"
            );
        }
        assert!(
            ProjectionGroup::read(
                &mut file,
                &index,
                &["up", "gate"],
                group.packed_bytes() - 1,
                &Device::Cpu
            )
            .is_err()
        );
        assert!(
            ProjectionGroup::read(&mut file, &index, &["up", "up"], 1 << 20, &Device::Cpu).is_err()
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "Main thread owns the GPU validation slot; CPU tests do not run this"]
fn cuda_grouped_projection_matches_separate_calls() {
    let device = Device::new_cuda(0).unwrap();
    for encoding in [
        GgmlDType::Q4_0,
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
    ] {
        let tensors = [
            ("gate", support::matrix(32, 256, 3, encoding)),
            ("up", support::matrix(48, 256, 9, encoding)),
        ];
        let mut file = support::fixture(&[], &tensors);
        let index = Index::read(&mut file, IndexLimits::default()).unwrap();
        let group =
            ProjectionGroup::read(&mut file, &index, &["gate", "up"], 1 << 20, &device).unwrap();
        let mut separate = Vec::new();
        for name in ["gate", "up"] {
            let tensor = index.tensor(name).unwrap();
            let mut raw = vec![0; tensor.bytes() as usize];
            file.seek(SeekFrom::Start(tensor.file_offset())).unwrap();
            file.read_exact(&mut raw).unwrap();
            separate.push(QMatMul::QTensor(std::sync::Arc::new(
                qtensor_from_ggml(encoding, &raw, tensor.backend_shape(), &device).unwrap(),
            )));
        }
        for rows in [1, 8, 64] {
            let input = Tensor::from_vec(
                (0..rows * 256)
                    .map(|i| (i as f32 * 0.013).cos())
                    .collect::<Vec<_>>(),
                (rows, 256),
                &device,
            )
            .unwrap();
            let fused = group.forward(&input).unwrap();
            for (output, op) in fused.iter().zip(&separate) {
                let error = (output - &op.forward(&input).unwrap())
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                eprintln!("encoding={encoding:?} rows={rows} max_abs_difference={error}");
                assert!(
                    error <= 1e-4,
                    "GPU grouped/separate proposed tolerance failed"
                );
            }
        }
    }
}
