//! Safetensors header boundaries are prepared in the serving layer.
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Read,
    path::Path,
};
use switchyard_artifacts::{
    Entry, Error, Metadata, Result,
    extents::{ExtentManifest, pack_extents_with_prefixes},
};

/// Preserve canonical checkpoint bytes and isolate each length field plus JSON header.
/// Headers are parsed and validated at preparation; no engine-specific tensor conversion occurs.
pub fn pack_weights(
    source: &Path,
    output: &Path,
    metadata: Metadata,
    extent_bytes: u64,
) -> Result<ExtentManifest> {
    let mut prefixes = BTreeMap::new();
    inspect(source, source, &mut prefixes)?;
    if prefixes.is_empty() {
        return Err(Error::Invalid("missing safetensors checkpoint".into()));
    }
    // Bake source-to-final byte destinations into immutable artifact metadata.
    // Only a completed final-state export may advertise final serving layout.
    let mut provisional =
        pack_extents_with_prefixes(source, output, metadata, extent_bytes, &prefixes)?;
    if source.join("switchyard-final-state.json").is_file() {
        let mut tensors = vec![];
        for (entry, item) in provisional.entries.iter().enumerate() {
            let Entry::File { path, size, .. } = item else {
                continue;
            };
            if !path.ends_with(".safetensors") {
                continue;
            }
            let mut input = File::open(source.join(path))?;
            let mut length = [0; 8];
            input.read_exact(&mut length)?;
            let mut header = vec![0; u64::from_le_bytes(length) as usize];
            input.read_exact(&mut header)?;
            for t in super::parse_header(entry, *size, &header)? {
                tensors.push(switchyard_artifacts::tensors::TensorDestination {
                    stride: switchyard_artifacts::tensors::dense_stride(&t.shape)?,
                    name: t.name,
                    dtype: t.dtype,
                    shape: t.shape,
                    layout: "dense_c".into(),
                    bytes: t.length,
                    ranges: vec![switchyard_artifacts::tensors::DestinationRange {
                        entry,
                        source_offset: t.offset,
                        destination_offset: 0,
                        length: t.length,
                    }],
                });
            }
        }
        let destinations = switchyard_artifacts::tensors::TensorManifest {
            format: "switchyard-tensor-destinations-v1".into(),
            worker: switchyard_artifacts::tensors::WorkerPartition {
                rank: 0,
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
            },
            tensors,
        };
        destinations.validate(&provisional.entries)?;
        provisional.metadata.attributes.insert(
            switchyard_artifacts::tensors::ATTRIBUTE.into(),
            serde_json::to_string(&destinations)?,
        );
        // Objects are content addressed and reused. Only identities/metadata change.
    }
    provisional.content_id = provisional.content_identity()?;
    provisional.id = provisional.identity()?;
    provisional.validate()?;
    Ok(provisional)
}
fn inspect(root: &Path, directory: &Path, prefixes: &mut BTreeMap<String, u64>) -> Result<()> {
    for child in fs::read_dir(directory)? {
        let child = child?;
        let kind = child.file_type()?;
        if kind.is_symlink() {
            return Err(Error::Invalid("weight source contains symlink".into()));
        }
        if kind.is_dir() {
            inspect(root, &child.path(), prefixes)?;
        } else if kind.is_file()
            && child
                .file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
        {
            let path = child.path();
            let size = child.metadata()?.len();
            let mut file = File::open(&path)?;
            let mut length = [0; 8];
            file.read_exact(&mut length)?;
            let n = u64::from_le_bytes(length);
            if n == 0 || n > 100_000_000 || n.checked_add(8).is_none_or(|n| n > size) {
                return Err(Error::Invalid("safetensors header length".into()));
            }
            let mut header = vec![0; n as usize];
            file.read_exact(&mut header)?;
            super::parse_header(0, size, &header)?;
            prefixes.insert(
                path.strip_prefix(root)
                    .map_err(|_| Error::Invalid("weight source path".into()))?
                    .to_string_lossy()
                    .into_owned(),
                n + 8,
            );
        }
    }
    Ok(())
}
/// Metadata files and each checkpoint's first extent precede bulk payload extents.
/// Generic transport receives only an ordered list of extent indices.
pub fn priority_extents(manifest: &ExtentManifest) -> Vec<usize> {
    let mut selected = manifest
        .extents
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match &manifest.entries[e.entry] {
            Entry::File { path, .. } if !path.ends_with(".safetensors") || e.offset == 0 => Some(i),
            _ => None,
        })
        .collect::<Vec<_>>();
    selected.sort_by_key(|i| match &manifest.entries[manifest.extents[*i].entry] {
        Entry::File { path, .. } => path.ends_with(".safetensors"),
        _ => false,
    });
    selected
}
