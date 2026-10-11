use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use switchyard_engine::{
    Result,
    models::{
        MixedModelSession, ModelFactory, ModelSession, QwenMixedSession, QwenSession,
        qwen35::{Config, ModelForCausalLM},
    },
};

pub fn model(value_heads: usize) -> ModelForCausalLM {
    let cfg: Config = serde_json::from_value(serde_json::json!({"text_config":{
        "vocab_size":32,"hidden_size":16,"intermediate_size":24,
        "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":8,"max_position_embeddings":160,"rms_norm_eps":1e-6,
        "rope_parameters":{"rope_theta":10000.,"partial_rotary_factor":0.5},
        "attention_bias":false,"hidden_act":"silu",
        "layer_types":["linear_attention","full_attention","linear_attention","full_attention"],
        "linear_num_key_heads":2,"linear_num_value_heads":value_heads,
        "linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":4
    }}))
    .unwrap();
    let vars = VarMap::new();
    let _shape_model = ModelForCausalLM::new(
        &cfg,
        VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
    )
    .unwrap();
    let tensors = vars
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| {
            let seed = name
                .bytes()
                .fold(0u32, |n, b| n.wrapping_mul(31).wrapping_add(u32::from(b)));
            let data = (0..var.elem_count())
                .map(|i| ((i as f32 * 0.37 + seed as f32 * 0.0001).sin()) * 0.03)
                .collect::<Vec<_>>();
            (
                name.clone(),
                Tensor::from_vec(data, var.shape(), &Device::Cpu).unwrap(),
            )
        })
        .collect();
    ModelForCausalLM::new(
        &cfg,
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
    )
    .unwrap()
}
pub fn prompt(length: usize, seed: u32) -> Vec<u32> {
    (0..length).map(|i| ((i as u32 * 7) + seed) % 32).collect()
}
pub fn tensor(prompt: &[u32]) -> Tensor {
    Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), &Device::Cpu).unwrap()
}
pub fn close(a: &Tensor, b: &Tensor) -> f32 {
    let error = (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(error <= 2e-5, "F32 maximum logit error {error}");
    error
}
pub struct Factory {
    pub weights: ModelForCausalLM,
    pub eos: u32,
}
impl ModelFactory for Factory {
    fn session(&self) -> Box<dyn ModelSession> {
        Box::new(QwenSession::new(self.weights.clone()))
    }
    fn mixed_session(&self, max_rows: usize) -> Result<Option<Box<dyn MixedModelSession>>> {
        Ok(Some(Box::new(QwenMixedSession::new(
            self.weights.clone(),
            max_rows,
        )?)))
    }
    fn device(&self) -> &Device {
        &Device::Cpu
    }
    fn vocab_size(&self) -> usize {
        32
    }
    fn eos_token(&self) -> u32 {
        self.eos
    }
}
