//! Little-endian GGUF v3 indexing. Tensor payloads are never read while indexing.
use crate::{Error, Result};
use candle_core::quantized::{GgmlDType, gguf_file::Value};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
};

#[derive(Clone, Copy, Debug)]
pub struct IndexLimits {
    pub header_bytes: u64,
    pub string_bytes: u64,
    pub metadata_entries: u64,
    pub tensors: u64,
    pub values: u64,
    pub array_depth: usize,
}
impl Default for IndexLimits {
    fn default() -> Self {
        Self {
            header_bytes: 64 << 20,
            string_bytes: 1 << 20,
            metadata_entries: 10_000,
            tensors: 100_000,
            values: 2_000_000,
            array_depth: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Encoding {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q8_0,
    Q4K,
    Q6K,
}
impl Encoding {
    pub fn from_code(code: u32) -> Result<Self> {
        Ok(match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            30 => Self::Bf16,
            _ => return Err(Error::Invalid(format!("unsupported GGML type {code}"))),
        })
    }
    pub fn backend(self) -> GgmlDType {
        match self {
            Self::F32 => GgmlDType::F32,
            Self::F16 => GgmlDType::F16,
            Self::Bf16 => GgmlDType::BF16,
            Self::Q4_0 => GgmlDType::Q4_0,
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q4K => GgmlDType::Q4K,
            Self::Q6K => GgmlDType::Q6K,
        }
    }
    pub fn quantized(self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::Bf16)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorDescriptor {
    name: String,
    /// GGML dimension order: contiguous input width comes first. This is not a transpose.
    dimensions: Vec<usize>,
    encoding: Encoding,
    file_offset: u64,
    bytes: u64,
}
impl TensorDescriptor {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }
    pub fn file_offset(&self) -> u64 {
        self.file_offset
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn backend_shape(&self) -> Vec<usize> {
        self.dimensions.iter().rev().copied().collect()
    }
}

#[derive(Debug)]
pub struct Index {
    metadata: BTreeMap<String, Value>,
    tensors: BTreeMap<String, TensorDescriptor>,
    file_bytes: u64,
    header_bytes: u64,
}
impl Index {
    pub fn read(reader: &mut (impl Read + Seek), limits: IndexLimits) -> Result<Self> {
        if !cfg!(target_endian = "little") {
            return Err(Error::Invalid("little-endian host required".into()));
        }
        let file_bytes = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        let mut parser = Parser {
            reader,
            limits,
            file_bytes,
            position: 0,
            values: 0,
        };
        if parser.bytes::<4>()? != *b"GGUF" || parser.u32()? != 3 {
            return Err(Error::Invalid(
                "only little-endian GGUF version 3 is admitted".into(),
            ));
        }
        let tensor_count = parser.u64()?;
        let metadata_count = parser.u64()?;
        if tensor_count == 0
            || tensor_count > limits.tensors
            || metadata_count > limits.metadata_entries
        {
            return Err(Error::Limit("tensor or metadata count".into()));
        }
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = parser.string()?;
            if key.is_empty() || metadata.contains_key(&key) {
                return Err(Error::Invalid("empty or duplicate metadata key".into()));
            }
            let kind = parser.u32()?;
            metadata.insert(key, parser.value(kind, 0)?);
        }
        let alignment = match metadata.get("general.alignment") {
            None => 32u64,
            Some(Value::U32(value)) => u64::from(*value),
            _ => return Err(Error::Invalid("general.alignment must be U32".into())),
        };
        if !alignment.is_power_of_two() || alignment > 4096 {
            return Err(Error::Invalid(
                "alignment must be a power of two no greater than 4096".into(),
            ));
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = parser.string()?;
            if name.is_empty() || tensors.contains_key(&name) {
                return Err(Error::Invalid("empty or duplicate tensor name".into()));
            }
            let rank = parser.u32()?;
            if !(1..=4).contains(&rank) {
                return Err(Error::Invalid("tensor rank outside 1..=4".into()));
            }
            let mut dimensions = Vec::new();
            let mut elements = 1u64;
            for _ in 0..rank {
                let dim = parser.u64()?;
                elements = elements
                    .checked_mul(dim)
                    .filter(|_| dim > 0)
                    .ok_or_else(|| Error::Invalid("zero dimension or shape overflow".into()))?;
                dimensions.push(
                    usize::try_from(dim).map_err(|_| {
                        Error::Invalid("dimension exceeds host address space".into())
                    })?,
                );
            }
            let encoding = Encoding::from_code(parser.u32()?)?;
            let block = encoding.backend().block_size() as u64;
            if !(dimensions[0] as u64).is_multiple_of(block) {
                return Err(Error::Invalid(format!(
                    "{name}: input row does not contain whole quantization blocks"
                )));
            }
            let bytes = (elements / block)
                .checked_mul(encoding.backend().type_size() as u64)
                .ok_or_else(|| Error::Invalid("tensor byte overflow".into()))?;
            let offset = parser.u64()?;
            if !offset.is_multiple_of(alignment) {
                return Err(Error::Invalid("unaligned tensor offset".into()));
            }
            tensors.insert(
                name.clone(),
                TensorDescriptor {
                    name,
                    dimensions,
                    encoding,
                    file_offset: offset,
                    bytes,
                },
            );
        }
        let header_bytes = parser.position;
        let data_start = header_bytes
            .checked_add(alignment - 1)
            .map(|n| n / alignment * alignment)
            .ok_or_else(|| Error::Invalid("data offset overflow".into()))?;
        let mut regions = Vec::new();
        for tensor in tensors.values_mut() {
            tensor.file_offset = data_start
                .checked_add(tensor.file_offset)
                .ok_or_else(|| Error::Invalid("tensor offset overflow".into()))?;
            let end = tensor
                .file_offset
                .checked_add(tensor.bytes)
                .filter(|&n| n <= file_bytes)
                .ok_or_else(|| Error::Invalid("tensor payload exceeds file boundary".into()))?;
            regions.push((tensor.file_offset, end));
        }
        regions.sort_unstable();
        if regions.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err(Error::Invalid("overlapping tensor payloads".into()));
        }
        Ok(Self {
            metadata,
            tensors,
            file_bytes,
            header_bytes,
        })
    }
    pub fn tensor(&self, name: &str) -> Result<&TensorDescriptor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("missing tensor {name}")))
    }
    pub fn tensors(&self) -> impl Iterator<Item = &TensorDescriptor> {
        self.tensors.values()
    }
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }
    pub fn header_bytes(&self) -> u64 {
        self.header_bytes
    }
    pub fn require_architecture(&self, expected: &str) -> Result<()> {
        match self.metadata.get("general.architecture") {
            Some(Value::String(architecture)) if architecture == expected => Ok(()),
            _ => Err(Error::Invalid(format!(
                "general.architecture must equal {expected}; file names are not architecture evidence"
            ))),
        }
    }
}

struct Parser<'a, R> {
    reader: &'a mut R,
    limits: IndexLimits,
    file_bytes: u64,
    position: u64,
    values: u64,
}
impl<R: Read> Parser<'_, R> {
    fn claim(&mut self, bytes: u64) -> Result<()> {
        self.position = self
            .position
            .checked_add(bytes)
            .ok_or_else(|| Error::Limit("header overflow".into()))?;
        if self.position > self.limits.header_bytes || self.position > self.file_bytes {
            return Err(Error::Limit("header byte budget or file boundary".into()));
        }
        Ok(())
    }
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.claim(N as u64)?;
        let mut bytes = [0; N];
        self.reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String> {
        let length = self.u64()?;
        if length > self.limits.string_bytes {
            return Err(Error::Limit("string byte budget".into()));
        }
        self.claim(length)?;
        let mut bytes = vec![
            0;
            usize::try_from(length)
                .map_err(|_| Error::Limit("string address space".into()))?
        ];
        self.reader.read_exact(&mut bytes)?;
        if bytes.contains(&0) {
            return Err(Error::Invalid("NUL in GGUF string".into()));
        }
        String::from_utf8(bytes).map_err(|_| Error::Invalid("invalid UTF-8".into()))
    }
    fn value(&mut self, kind: u32, depth: usize) -> Result<Value> {
        self.values += 1;
        if self.values > self.limits.values {
            return Err(Error::Limit("metadata value budget".into()));
        }
        Ok(match kind {
            0 => Value::U8(self.bytes::<1>()?[0]),
            1 => Value::I8(self.bytes::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => match self.bytes::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err(Error::Invalid("boolean must be 0 or 1".into())),
            },
            8 => Value::String(self.string()?),
            9 => {
                if depth >= self.limits.array_depth {
                    return Err(Error::Limit("array nesting budget".into()));
                }
                let element_kind = self.u32()?;
                let length = self.u64()?;
                if length > self.limits.values - self.values {
                    return Err(Error::Limit("array value budget".into()));
                }
                let mut values = Vec::new();
                for _ in 0..length {
                    values.push(self.value(element_kind, depth + 1)?);
                }
                Value::Array(values)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            _ => return Err(Error::Invalid(format!("unsupported metadata type {kind}"))),
        })
    }
}
