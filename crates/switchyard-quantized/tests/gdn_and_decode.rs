use super::support;
use candle_core::quantized::{GgmlDType, gguf_file::Value};
use switchyard_quantized::{
    decode_plan::{DecodePlan, DecodeRow, StateLease},
    gdn::{HeadOrder, Qwen35Requirements},
    gguf::{Index, IndexLimits},
};

#[test]
fn grouped_and_tiled_head_maps_are_inverse_for_ratios_one_two_and_three() {
    for (keys, values, width) in [(16, 16, 128), (16, 32, 128), (16, 48, 128)] {
        let order = HeadOrder::new(keys, values, width).unwrap();
        for i in 0..values * width {
            assert_eq!(
                order
                    .tiled_to_grouped(order.grouped_to_tiled(i).unwrap())
                    .unwrap(),
                i
            );
        }
        assert!(order.grouped_to_tiled(values * width).is_err());
    }
    let order = HeadOrder::new(2, 6, 1).unwrap();
    assert_eq!(
        (0..6)
            .map(|i| order.grouped_to_tiled(i).unwrap())
            .collect::<Vec<_>>(),
        [0, 2, 4, 1, 3, 5]
    );
    assert!(HeadOrder::new(0, 4, 2).is_err());
    assert!(HeadOrder::new(3, 4, 2).is_err());
}

#[test]
fn qwen_gdn_layout_is_checked_but_native_admission_remains_closed() {
    let metadata = [
        ("general.architecture", Value::String("qwen35".into())),
        ("qwen35.embedding_length", Value::U32(256)),
        ("qwen35.block_count", Value::U32(1)),
        ("qwen35.ssm.group_count", Value::U32(2)),
        ("qwen35.ssm.time_step_rank", Value::U32(6)),
        ("qwen35.ssm.state_size", Value::U32(32)),
        ("qwen35.ssm.inner_size", Value::U32(192)),
        ("qwen35.ssm.conv_kernel", Value::U32(4)),
        (
            "qwen35.attention.recurrent_layers",
            Value::Array(vec![Value::Bool(true)]),
        ),
    ];
    let specs = [
        ("attn_qkv.weight", 256, 320),
        ("attn_gate.weight", 256, 192),
        ("ssm_beta.weight", 256, 6),
        ("ssm_alpha.weight", 256, 6),
        ("ssm_conv1d.weight", 4, 320),
        ("ssm_dt.bias", 6, 1),
        ("ssm_a", 6, 1),
        ("ssm_norm.weight", 32, 1),
        ("ssm_out.weight", 192, 256),
    ];
    let names = specs
        .iter()
        .map(|(n, _, _)| format!("blk.0.{n}"))
        .collect::<Vec<_>>();
    let tensors = specs
        .iter()
        .enumerate()
        .map(|(i, (_, columns, rows))| {
            let data = vec![0.25f32; columns * rows];
            let shape = if *rows == 1 {
                vec![*columns]
            } else {
                vec![*rows, *columns]
            };
            let t = candle_core::Tensor::from_vec(data, shape, &candle_core::Device::Cpu).unwrap();
            (
                names[i].as_str(),
                candle_core::quantized::QTensor::quantize(&t, GgmlDType::F32).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let mut file = support::fixture(&metadata, &tensors);
    let index = Index::read(&mut file, IndexLimits::default()).unwrap();
    let report = Qwen35Requirements::inspect(&index).unwrap();
    assert!(!report.native_serving_admitted);
    assert_eq!((report.key_heads, report.value_heads), (2, 6));
    assert!(
        report
            .required_conversions
            .iter()
            .any(|s| s.contains("negative exp"))
    );
    let mut file = support::fixture(&metadata, &tensors[..8]);
    let missing = Index::read(&mut file, IndexLimits::default()).unwrap();
    assert!(Qwen35Requirements::inspect(&missing).is_err());
    let mut altered = metadata.to_vec();
    altered[6].1 = Value::U32(191);
    let mut file = support::fixture(&altered, &tensors);
    let wrong = Index::read(&mut file, IndexLimits::default()).unwrap();
    assert!(Qwen35Requirements::inspect(&wrong).is_err());
}

#[test]
fn varied_prefills_advance_independently_and_completed_state_leases_are_returned() {
    let rows = vec![
        DecodeRow {
            request_id: 9,
            state: StateLease {
                slot: 2,
                generation: 1,
            },
            token: 4,
            position: 3,
            remaining_outputs: 3,
        },
        DecodeRow {
            request_id: 3,
            state: StateLease {
                slot: 5,
                generation: 4,
            },
            token: 8,
            position: 17,
            remaining_outputs: 1,
        },
        DecodeRow {
            request_id: 5,
            state: StateLease {
                slot: 1,
                generation: 7,
            },
            token: 1,
            position: 8,
            remaining_outputs: 4,
        },
    ];
    let mut plan = DecodePlan::new(rows.clone(), 3, 32).unwrap();
    let released = plan.advance(&[5, 9, 2], &[false, false, false]).unwrap();
    assert_eq!(released, [rows[1].state]);
    assert_eq!(
        plan.rows()
            .iter()
            .map(|r| (r.request_id, r.position, r.token))
            .collect::<Vec<_>>(),
        [(9, 4, 5), (5, 9, 2)]
    );
    assert!(plan.advance(&[6], &[false]).is_err());
    let released = plan.advance(&[6, 3], &[true, false]).unwrap();
    assert_eq!(released, [rows[0].state]);
    assert_eq!(plan.rows()[0].state, rows[2].state);
    assert_eq!(plan.rows()[0].position, 10);
    assert!(DecodePlan::new(rows.clone(), 2, 32).is_err());
    let mut duplicate = rows;
    duplicate[2].state.slot = 2;
    assert!(DecodePlan::new(duplicate, 3, 32).is_err());
}
