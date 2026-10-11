//! Same-input projections share one packed matmul. Output slices retain caller order.
use crate::{Error, Result, gguf::Index};
use candle_core::{
    DType, Device, Module, Tensor,
    quantized::{QMatMul, ggml_file::qtensor_from_ggml},
};
use std::{
    collections::BTreeSet,
    io::{Read, Seek, SeekFrom},
    sync::Arc,
};

pub struct ProjectionGroup {
    operator: QMatMul,
    outputs: Vec<(String, usize, usize)>,
    input_width: usize,
    packed_bytes: u64,
}
impl ProjectionGroup {
    /// Reads only selected packed matrices, with an explicit aggregate payload cap.
    /// Backend construction may make an additional packed copy. No full F32 weights.
    pub fn read(
        reader: &mut (impl Read + Seek),
        index: &Index,
        names: &[&str],
        max_packed_bytes: u64,
        device: &Device,
    ) -> Result<Self> {
        if names.is_empty() || names.len() > 16 {
            return Err(Error::Limit("projection count outside 1..=16".into()));
        }
        let mut unique = BTreeSet::new();
        let first = index.tensor(names[0])?;
        let encoding = first.encoding();
        if !encoding.quantized() || first.dimensions().len() != 2 {
            return Err(Error::Invalid(
                "projection group requires quantized rank-2 matrices".into(),
            ));
        }
        let input_width = first.dimensions()[0];
        let mut packed_bytes = 0u64;
        let mut rows = 0usize;
        let mut outputs = Vec::new();
        for name in names {
            if !unique.insert(name) {
                return Err(Error::Invalid("duplicate projection".into()));
            }
            let tensor = index.tensor(name)?;
            if tensor.encoding() != encoding
                || tensor.dimensions().len() != 2
                || tensor.dimensions()[0] != input_width
            {
                return Err(Error::Invalid(
                    "projection encodings and input widths must match".into(),
                ));
            }
            packed_bytes = packed_bytes
                .checked_add(tensor.bytes())
                .filter(|&n| n <= max_packed_bytes)
                .ok_or_else(|| Error::Limit("aggregate packed projection bytes".into()))?;
            outputs.push((name.to_string(), rows, tensor.dimensions()[1]));
            rows = rows
                .checked_add(tensor.dimensions()[1])
                .ok_or_else(|| Error::Limit("projection rows".into()))?;
        }
        let length = usize::try_from(packed_bytes)
            .map_err(|_| Error::Limit("packed address space".into()))?;
        rows.checked_mul(input_width)
            .ok_or_else(|| Error::Limit("projection shape address space".into()))?;
        let mut packed = vec![0u8; length];
        let mut cursor = 0;
        for name in names {
            let tensor = index.tensor(name)?;
            let length = tensor.bytes() as usize;
            reader.seek(SeekFrom::Start(tensor.file_offset()))?;
            reader.read_exact(&mut packed[cursor..cursor + length])?;
            cursor += length;
        }
        let qtensor =
            qtensor_from_ggml(encoding.backend(), &packed, vec![rows, input_width], device)?;
        // Construct this variant directly: CANDLE_DEQUANTIZE_ALL must not silently
        // turn a supposedly quantized operation into a full dense allocation.
        let operator = QMatMul::QTensor(Arc::new(qtensor));
        Ok(Self {
            operator,
            outputs,
            input_width,
            packed_bytes,
        })
    }
    pub fn packed_bytes(&self) -> u64 {
        self.packed_bytes
    }
    pub fn output_names(&self) -> impl Iterator<Item = &str> {
        self.outputs.iter().map(|(name, _, _)| name.as_str())
    }
    pub fn forward(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        if !(2..=3).contains(&input.rank())
            || input.dims().last() != Some(&self.input_width)
            || input.dtype() != DType::F32
            || !input.is_contiguous()
        {
            return Err(Error::Invalid(
                "projection input must be contiguous F32 [rows,width] or [batch,tokens,width]"
                    .into(),
            ));
        }
        let output = self.operator.forward(input)?;
        self.outputs
            .iter()
            .map(|(_, start, rows)| Ok(output.narrow(output.rank() - 1, *start, *rows)?))
            .collect()
    }
}
