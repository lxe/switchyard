use candle_core::{
    Device, Tensor,
    quantized::{
        GgmlDType, QTensor,
        gguf_file::{self, Value},
    },
};
use std::io::Cursor;

pub fn matrix(rows: usize, columns: usize, seed: usize, dtype: GgmlDType) -> QTensor {
    let data = (0..rows * columns)
        .map(|i| ((i * 17 + seed * 37) as f32 * 0.071).sin())
        .collect::<Vec<_>>();
    QTensor::quantize(
        &Tensor::from_vec(data, (rows, columns), &Device::Cpu).unwrap(),
        dtype,
    )
    .unwrap()
}
pub fn fixture(metadata: &[(&str, Value)], tensors: &[(&str, QTensor)]) -> Cursor<Vec<u8>> {
    let mut file = Cursor::new(Vec::new());
    let metadata = metadata.iter().map(|(n, v)| (*n, v)).collect::<Vec<_>>();
    let tensors = tensors.iter().map(|(n, t)| (*n, t)).collect::<Vec<_>>();
    gguf_file::write(&mut file, &metadata, &tensors).unwrap();
    // Candle's writer emits v2. The fields used by these LE fixtures have the
    // same wire representation in v3; only the explicit version word differs.
    file.get_mut()[4..8].copy_from_slice(&3u32.to_le_bytes());
    file.set_position(0);
    file
}
pub fn raw_fixture(
    metadata: &[(&str, Value)],
    tensors: &[(&str, &[u64], u32, u64)],
    payload: &[u8],
) -> Cursor<Vec<u8>> {
    fn string(bytes: &mut Vec<u8>, s: &str) {
        bytes.extend((s.len() as u64).to_le_bytes());
        bytes.extend(s.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        string(&mut bytes, key);
        match value {
            Value::String(text) => {
                bytes.extend(8u32.to_le_bytes());
                string(&mut bytes, text);
            }
            Value::U32(number) => {
                bytes.extend(4u32.to_le_bytes());
                bytes.extend(number.to_le_bytes());
            }
            _ => panic!("raw fixture value type"),
        }
    }
    for (name, dims, kind, offset) in tensors {
        string(&mut bytes, name);
        bytes.extend((dims.len() as u32).to_le_bytes());
        for d in *dims {
            bytes.extend(d.to_le_bytes());
        }
        bytes.extend(kind.to_le_bytes());
        bytes.extend(offset.to_le_bytes());
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(payload);
    Cursor::new(bytes)
}
