//! Greedy selection has a defined tie rule: the lowest vocabulary ID wins.
use crate::Result;
use candle_core::{Device, Tensor};

pub struct Greedy {
    indices: Tensor,
    sentinel: Tensor,
}
impl Greedy {
    pub fn new(vocab: usize, device: &Device) -> Result<Self> {
        let vocab = u32::try_from(vocab)
            .map_err(|_| crate::Error::Config("Vocabulary exceeds token ID range".into()))?;
        Ok(Self {
            indices: Tensor::arange(0u32, vocab, device)?,
            sentinel: Tensor::full(u32::MAX, vocab as usize, device)?,
        })
    }
    pub fn select(&self, logits: &Tensor) -> Result<Tensor> {
        let (rows, vocab) = logits.dims2()?;
        let equal = logits.broadcast_eq(&logits.max_keepdim(1)?)?;
        let indices = self.indices.broadcast_as((rows, vocab))?;
        let sentinel = self.sentinel.broadcast_as((rows, vocab))?;
        Ok(equal.where_cond(&indices, &sentinel)?.min(1)?)
    }
}
