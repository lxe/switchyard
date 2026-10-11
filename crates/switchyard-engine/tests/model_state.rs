//! Tiny CPU model tests exercise actual attention and recurrent-state math.
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use switchyard_engine::models::qwen35::{Config, ModelForCausalLM};

fn model() -> ModelForCausalLM {
    let cfg: Config = serde_json::from_value(serde_json::json!({"text_config":{
        "vocab_size":32,"hidden_size":16,"intermediate_size":24,
        "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":8,"max_position_embeddings":16,"rms_norm_eps":1e-6,
        "rope_parameters":{"rope_theta":10000.,"partial_rotary_factor":0.5},
        "attention_bias":false,"hidden_act":"silu",
        "layer_types":["linear_attention","full_attention"],
        "linear_num_key_heads":2,"linear_num_value_heads":2,
        "linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":4
    }}))
    .unwrap();
    let vars = VarMap::new();
    ModelForCausalLM::new(
        &cfg,
        VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
    )
    .unwrap()
}
fn close(a: &Tensor, b: &Tensor) {
    let error = (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .fold(0f32, f32::max);
    assert!(error < 2e-5, "maximum logit error {error}");
}
#[test]
fn cached_decode_matches_full_causal_prefix() {
    let weights = model();
    let mut cached = weights.clone();
    let mut full = weights.clone();
    let prefix = Tensor::new(&[[1u32, 2, 3]], &Device::Cpu).unwrap();
    cached.forward(&prefix, 0).unwrap();
    let decoded = cached
        .forward(&Tensor::new(&[[4u32]], &Device::Cpu).unwrap(), 3)
        .unwrap();
    let expected = full
        .forward(&Tensor::new(&[[1u32, 2, 3, 4]], &Device::Cpu).unwrap(), 0)
        .unwrap();
    close(&decoded, &expected);
}
#[test]
fn batch_rows_match_independent_sessions_and_weights_keep_empty_state() {
    let weights = model();
    let mut batched = weights.clone();
    let batch = batched
        .forward(
            &Tensor::new(&[[1u32, 2, 3], [3u32, 2, 1]], &Device::Cpu).unwrap(),
            0,
        )
        .unwrap();
    for (i, prompt) in [[1u32, 2, 3], [3u32, 2, 1]].iter().enumerate() {
        let mut single = weights.clone();
        let output = single
            .forward(&Tensor::new(&[*prompt], &Device::Cpu).unwrap(), 0)
            .unwrap();
        close(&batch.narrow(0, i, 1).unwrap(), &output);
    }
    batched.clear_kv_cache();
    let replay = batched
        .forward(
            &Tensor::new(&[[1u32, 2, 3], [3u32, 2, 1]], &Device::Cpu).unwrap(),
            0,
        )
        .unwrap();
    close(&batch, &replay);
}
