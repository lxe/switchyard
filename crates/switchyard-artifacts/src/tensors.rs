//! Serving tensor byte destinations, independent of engines and object stores.
use crate::{Entry, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
pub const ATTRIBUTE: &str = "tensor_destinations_v1";
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorManifest {
    pub format: String,
    pub worker: WorkerPartition,
    pub tensors: Vec<TensorDestination>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerPartition {
    pub rank: usize,
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDestination {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub layout: String,
    pub stride: Vec<u64>,
    pub bytes: u64,
    pub ranges: Vec<DestinationRange>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DestinationRange {
    pub entry: usize,
    pub source_offset: u64,
    pub destination_offset: u64,
    pub length: u64,
}
pub fn dtype_bytes(dtype: &str) -> Result<u64> {
    Ok(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F32" | "I32" | "U32" => 4,
        "F64" | "I64" | "U64" => 8,
        _ => return Err(Error::Invalid("tensor dtype".into())),
    })
}
pub fn dense_stride(shape: &[u64]) -> Result<Vec<u64>> {
    let mut stride = vec![0; shape.len()];
    let mut n = 1_u64;
    for (i, d) in shape.iter().enumerate().rev() {
        stride[i] = n;
        n = n
            .checked_mul(*d)
            .ok_or_else(|| Error::Invalid("tensor shape overflow".into()))?;
    }
    Ok(stride)
}
impl TensorManifest {
    pub fn validate(&self, entries: &[Entry]) -> Result<()> {
        if self.format != "switchyard-tensor-destinations-v1"
            || self.tensors.is_empty()
            || self.tensors.len() > 1_000_000
            || self.worker.tensor_parallel_size == 0
            || self.worker.pipeline_parallel_size == 0
            || self.worker.rank
                >= self.worker.tensor_parallel_size * self.worker.pipeline_parallel_size
        {
            return Err(Error::Invalid("tensor manifest or worker partition".into()));
        }
        let mut names = BTreeSet::new();
        let mut sources = vec![];
        for t in &self.tensors {
            let width = dtype_bytes(&t.dtype)?;
            let count = t
                .shape
                .iter()
                .try_fold(1_u64, |n, d| n.checked_mul(*d))
                .ok_or_else(|| Error::Invalid("tensor shape overflow".into()))?;
            // This first format packs dense C storage. Noncontiguous destinations
            // must be transformed at bake or use a future explicitly defined layout.
            if t.name.is_empty()
                || t.name.len() > 4096
                || !names.insert(&t.name)
                || t.layout != "dense_c"
                || t.stride != dense_stride(&t.shape)?
                || count.checked_mul(width) != Some(t.bytes)
            {
                return Err(Error::Invalid("tensor destination layout".into()));
            }
            let mut cursor = 0;
            for r in &t.ranges {
                let size = match entries.get(r.entry) {
                    Some(Entry::File { size, .. }) => *size,
                    _ => return Err(Error::Invalid("tensor source entry".into())),
                };
                if r.length == 0
                    || r.destination_offset != cursor
                    || r.source_offset
                        .checked_add(r.length)
                        .is_none_or(|n| n > size)
                    || r.length % width != 0
                    || r.destination_offset % width != 0
                {
                    return Err(Error::Invalid("tensor byte mapping".into()));
                }
                cursor = cursor
                    .checked_add(r.length)
                    .ok_or_else(|| Error::Invalid("tensor mapping overflow".into()))?;
                sources.push((r.entry, r.source_offset, r.length));
            }
            if cursor != t.bytes {
                return Err(Error::Invalid(
                    "incomplete tensor destination coverage".into(),
                ));
            }
        }
        sources.sort();
        for pair in sources.windows(2) {
            if pair[0].0 == pair[1].0 && pair[0].1 + pair[0].2 > pair[1].1 {
                return Err(Error::Invalid("overlapping tensor source mappings".into()));
            }
        }
        Ok(())
    }
}
