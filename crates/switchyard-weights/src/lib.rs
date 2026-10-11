//! Safetensors metadata and readiness, outside generic filesystem transport.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use switchyard_artifacts::{
    Entry, Error, Result,
    extents::{CompletedRange, ExtentHydration},
};

/// Absolute file offsets are derived from safetensors' data-relative offsets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub entry: usize,
    pub offset: u64,
    pub length: u64,
    pub file_size: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorInfo {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Parse a completed header without touching incomplete tensor data.
/// This adapter deliberately supports byte-aligned scalar dtypes only.
pub fn parse_header(entry: usize, file_size: u64, header: &[u8]) -> Result<Vec<Tensor>> {
    let data_start = 8_u64
        .checked_add(header.len() as u64)
        .ok_or_else(|| Error::Invalid("safetensors header overflow".into()))?;
    if data_start > file_size || header.first() != Some(&b'{') {
        return Err(Error::Invalid("safetensors header boundary".into()));
    }
    // Safetensors forbids duplicate keys. A custom map visitor rejects them instead
    // of allowing serde_json's normal map deserializer to keep the final value.
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<String, TensorInfo>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unique safetensors keys")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut values = BTreeMap::new();
            let mut seen = std::collections::BTreeSet::new();
            while let Some(key) = map.next_key::<String>()? {
                if !seen.insert(key.clone()) {
                    return Err(serde::de::Error::custom("duplicate safetensors key"));
                }
                if key == "__metadata__" {
                    let _: BTreeMap<String, String> = map.next_value()?;
                } else {
                    // Deserialize the struct directly so duplicate dtype/shape/offset
                    // fields are rejected too, before a generic JSON Value can erase them.
                    values.insert(key, map.next_value::<TensorInfo>()?);
                }
            }
            Ok(values)
        }
    }
    use serde::de::Deserializer;
    let mut de = serde_json::Deserializer::from_slice(header);
    let values = (&mut de).deserialize_map(Unique)?;
    de.end()?;
    let mut tensors = vec![];
    for (name, info) in values {
        if name.len() > 4096 || info.shape.len() > 128 {
            return Err(Error::Invalid("safetensors name or rank limit".into()));
        }
        let width = match info.dtype.as_str() {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1_u64,
            "F16" | "BF16" | "I16" | "U16" => 2,
            "F32" | "I32" | "U32" => 4,
            "F64" | "I64" | "U64" => 8,
            _ => return Err(Error::Invalid("unsupported safetensors dtype".into())),
        };
        let count = info
            .shape
            .iter()
            .try_fold(1_u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::Invalid("safetensors shape overflow".into()))?;
        let length = count
            .checked_mul(width)
            .ok_or_else(|| Error::Invalid("safetensors tensor overflow".into()))?;
        let [start, end] = info.data_offsets;
        if start.checked_add(length) != Some(end) || end > file_size - data_start {
            return Err(Error::Invalid(
                "safetensors tensor boundary or shape".into(),
            ));
        }
        tensors.push(Tensor {
            name,
            dtype: info.dtype,
            shape: info.shape,
            entry,
            offset: data_start + start,
            length,
            file_size,
        });
    }
    tensors.sort_by(|a, b| (a.offset, a.length, &a.name).cmp(&(b.offset, b.length, &b.name)));
    let mut cursor = data_start;
    for tensor in &tensors {
        if tensor.offset != cursor {
            return Err(Error::Invalid(
                "safetensors overlapping or missing data".into(),
            ));
        }
        cursor += tensor.length;
    }
    if cursor != file_size {
        return Err(Error::Invalid(
            "safetensors trailing or missing data".into(),
        ));
    }
    Ok(tensors)
}

/// Retains a completion lease through the consumer's copy/assignment lifetime.
pub struct ReadyTensor {
    pub tensor: Tensor,
    pub bytes: CompletedRange,
}

pub struct WeightPlan {
    pub tensors: Vec<Tensor>,
}
impl WeightPlan {
    pub async fn discover(hydration: &ExtentHydration, max_header_bytes: u64) -> Result<Self> {
        if max_header_bytes == 0 || max_header_bytes > 100_000_000 {
            return Err(Error::Invalid("safetensors header limit".into()));
        }
        let mut tensors = vec![];
        for (entry, file) in hydration.manifest().entries.iter().enumerate() {
            let Entry::File { path, size, .. } = file else {
                continue;
            };
            if !path.ends_with(".safetensors") {
                continue;
            }
            let prefix = hydration.wait_range(entry, 0, 8).await?;
            let mut length = [0_u8; 8];
            prefix.read_into(0, &mut length)?;
            let length = u64::from_le_bytes(length);
            if length > max_header_bytes || length.checked_add(8).is_none_or(|end| end > *size) {
                return Err(Error::Invalid("safetensors header length".into()));
            }
            let completed = hydration.wait_range(entry, 8, length).await?;
            let mut header = vec![0; length as usize];
            completed.read_into(0, &mut header)?;
            tensors.extend(parse_header(entry, *size, &header)?);
        }
        let mut names = std::collections::BTreeSet::new();
        if tensors.is_empty() || tensors.iter().any(|t| !names.insert(t.name.clone())) {
            return Err(Error::Invalid(
                "missing or duplicate checkpoint tensors".into(),
            ));
        }
        Ok(Self { tensors })
    }
    /// Pick the first completed remaining tensor; no incomplete mapped data is read.
    pub async fn next_ready(
        &self,
        hydration: &ExtentHydration,
        remaining: &[usize],
    ) -> Result<Option<(usize, ReadyTensor)>> {
        if remaining.is_empty() {
            return Ok(None);
        }
        let ranges = remaining
            .iter()
            .map(|i| {
                let t = &self.tensors[*i];
                (t.entry, t.offset, t.length)
            })
            .collect::<Vec<_>>();
        let selected = hydration.wait_any(&ranges).await?;
        let index = remaining[selected];
        let tensor = self.tensors[index].clone();
        let bytes = hydration
            .wait_range(tensor.entry, tensor.offset, tensor.length)
            .await?;
        Ok(Some((index, ReadyTensor { tensor, bytes })))
    }
}

/// Incremental discovery permits a completed tensor from one file to load while
/// another file's header is still arriving. Every selection uses the same completion gate.
pub struct WeightStream {
    headers: Vec<(usize, u64, Option<u64>)>,
    pending: Vec<Tensor>,
    names: std::collections::BTreeSet<String>,
    max_header_bytes: u64,
}
impl WeightStream {
    pub fn new(hydration: &ExtentHydration, max_header_bytes: u64) -> Result<Self> {
        if max_header_bytes == 0 || max_header_bytes > 100_000_000 {
            return Err(Error::Invalid("safetensors header limit".into()));
        }
        let headers = hydration
            .manifest()
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| match e {
                Entry::File { path, size, .. } if path.ends_with(".safetensors") => {
                    Some((i, *size, None))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if headers.is_empty() {
            return Err(Error::Invalid("missing checkpoint files".into()));
        }
        Ok(Self {
            headers,
            pending: vec![],
            names: std::collections::BTreeSet::new(),
            max_header_bytes,
        })
    }
    pub async fn next_ready(&mut self, hydration: &ExtentHydration) -> Result<Option<ReadyTensor>> {
        loop {
            let mut ranges = self
                .pending
                .iter()
                .map(|t| (t.entry, t.offset, t.length))
                .collect::<Vec<_>>();
            ranges.extend(self.headers.iter().map(|(entry, _, length)| match length {
                Some(n) => (*entry, 8, *n),
                None => (*entry, 0, 8),
            }));
            if ranges.is_empty() {
                if self.names.is_empty() {
                    return Err(Error::Invalid("empty checkpoint".into()));
                }
                return Ok(None);
            }
            let selected = hydration.wait_any(&ranges).await?;
            let (entry, offset, length) = ranges[selected];
            let bytes = hydration.wait_range(entry, offset, length).await?;
            if selected < self.pending.len() {
                return Ok(Some(ReadyTensor {
                    tensor: self.pending.remove(selected),
                    bytes,
                }));
            }
            let index = selected - self.pending.len();
            let (_, size, header_length) = self.headers[index];
            if header_length.is_none() {
                let mut encoded = [0_u8; 8];
                bytes.read_into(0, &mut encoded)?;
                let n = u64::from_le_bytes(encoded);
                if n > self.max_header_bytes || n.checked_add(8).is_none_or(|end| end > size) {
                    return Err(Error::Invalid("safetensors header length".into()));
                }
                self.headers[index].2 = Some(n);
            } else {
                let mut header = vec![0_u8; length as usize];
                bytes.read_into(0, &mut header)?;
                let tensors = parse_header(entry, size, &header)?;
                if tensors.iter().any(|t| !self.names.insert(t.name.clone())) {
                    return Err(Error::Invalid("duplicate checkpoint tensor".into()));
                }
                self.pending.extend(tensors);
                self.headers.remove(index);
            }
        }
    }
}
