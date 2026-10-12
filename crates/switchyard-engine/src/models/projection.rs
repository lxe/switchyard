//! Weight-format-neutral projections. GGUF matrices remain packed during execution.
use super::gguf::Source;
use candle_core::{DType, Device, Module, Result, Shape, Tensor, quantized::QMatMul};
use candle_nn::VarBuilder;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum Linear {
    Dense(candle_nn::Linear),
    Packed {
        operator: QMatMul,
        input_order: Option<Tensor>,
        output_order: Option<Tensor>,
    },
}
impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(input),
            Self::Packed {
                operator,
                input_order,
                output_order,
            } => {
                let input = input.to_dtype(DType::F32)?.contiguous()?;
                let input = match input_order {
                    Some(order) => input.index_select(order, input.rank() - 1)?.contiguous()?,
                    None => input,
                };
                let output = operator.forward(&input)?;
                match output_order {
                    Some(order) => output.index_select(order, output.rank() - 1),
                    None => Ok(output),
                }
            }
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) enum Embedding {
    Dense(candle_nn::Embedding),
    Packed(QMatMul),
}
impl Module for Embedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(embedding) => embedding.forward(ids),
            Self::Packed(operator) => operator.embedding(ids),
        }
    }
}
impl Embedding {
    pub fn tied_output(&self) -> Linear {
        match self {
            Self::Dense(embedding) => {
                Linear::Dense(candle_nn::Linear::new(embedding.embeddings().clone(), None))
            }
            Self::Packed(operator) => Linear::Packed {
                operator: operator.clone(),
                input_order: None,
                output_order: None,
            },
        }
    }
}
/// A dense builder is also used for GGUF's small auxiliaries;
/// projection lookup deliberately bypasses it to avoid dense matrix expansion.
#[derive(Clone)]
pub(crate) struct ModelWeights<'a> {
    vb: VarBuilder<'a>,
    source: Option<Arc<Source>>,
}
impl<'a> ModelWeights<'a> {
    pub fn dense(vb: VarBuilder<'a>) -> Self {
        Self { vb, source: None }
    }
    pub fn gguf(vb: VarBuilder<'a>, source: Arc<Source>) -> Self {
        Self {
            vb,
            source: Some(source),
        }
    }
    pub fn pp(&self, prefix: impl ToString) -> Self {
        Self {
            vb: self.vb.pp(prefix),
            source: self.source.clone(),
        }
    }
    pub fn get(&self, shape: impl Into<Shape>, name: &str) -> Result<Tensor> {
        self.vb.get(shape, name)
    }
    pub fn builder(&self) -> VarBuilder<'a> {
        self.vb.clone()
    }
    pub fn device(&self) -> &Device {
        self.vb.device()
    }
    pub fn dtype(&self) -> DType {
        self.vb.dtype()
    }
    pub fn set_dtype(self, dtype: DType) -> Self {
        Self {
            vb: self.vb.set_dtype(dtype),
            ..self
        }
    }
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.vb.contains_tensor(name)
    }
    pub fn norm_weight(&self, dim: usize) -> Result<Tensor> {
        let weight = self.vb.get(dim, "weight")?.to_dtype(DType::F32)?;
        if self.source.is_some() {
            Ok(weight)
        } else {
            weight + 1.0
        }
    }
    pub fn negative_a(&self, dim: usize) -> Result<Tensor> {
        let weight = self.vb.get(dim, "A_log")?.to_dtype(DType::F32)?;
        if self.source.is_some() {
            Ok(weight)
        } else {
            weight.exp()? * -1.0
        }
    }
    pub fn embedding(&self, rows: usize, width: usize) -> Result<Embedding> {
        match &self.source {
            Some(source) => {
                let projection =
                    source.projection("token_embd.weight", width, rows, None, self.device())?;
                let Linear::Packed { operator, .. } = projection else {
                    unreachable!("GGUF projection returns packed operator")
                };
                Ok(Embedding::Packed(operator))
            }
            None => Ok(Embedding::Dense(candle_nn::embedding(
                rows,
                width,
                self.vb.clone(),
            )?)),
        }
    }
}
pub(crate) fn linear_no_bias(
    input: usize,
    output: usize,
    weights: ModelWeights<'_>,
) -> Result<Linear> {
    linear_b(input, output, false, weights)
}
pub(crate) fn linear_b(
    input: usize,
    output: usize,
    bias: bool,
    weights: ModelWeights<'_>,
) -> Result<Linear> {
    match &weights.source {
        Some(source) => {
            if bias {
                candle_core::bail!("GGUF Qwen3.5 biased projections unsupported");
            }
            let prefix = weights.vb.prefix();
            let (name, permutation) = source.projection_name(&prefix)?;
            source.projection(&name, input, output, permutation, weights.device())
        }
        None => Ok(Linear::Dense(candle_nn::linear_b(
            input, output, bias, weights.vb,
        )?)),
    }
}
