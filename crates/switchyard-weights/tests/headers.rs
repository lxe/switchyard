use switchyard_weights::parse_header;

#[test]
fn exact_tensor_shapes_offsets_and_empty_tensors() {
    let header=br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"empty":{"dtype":"BF16","shape":[0],"data_offsets":[8,8]},"b":{"dtype":"BF16","shape":[2,2],"data_offsets":[8,16]},"__metadata__":{"format":"pt"}}"#;
    let tensors = parse_header(3, 8 + header.len() as u64 + 16, header).unwrap();
    assert_eq!(tensors.len(), 3);
    assert_eq!(tensors[0].name, "a");
    assert_eq!(tensors[0].offset, 8 + header.len() as u64);
    assert_eq!(tensors[1].length, 0);
    assert_eq!(tensors[2].shape, [2, 2]);
}
#[test]
fn malformed_headers_fail_closed() {
    for header in [
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,8]}}"#,
        r#"{"a":{"dtype":"U3","shape":[1],"data_offsets":[0,1]}}"#,
        r#"{"__metadata__":{"format":4}}"#,
        r#"{"a":{"dtype":"F32","dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#,
    ] {
        assert!(parse_header(0, 8 + header.len() as u64 + 8, header.as_bytes()).is_err());
    }
}

#[test]
fn prepared_header_has_an_independent_object_without_altering_checkpoint_bytes() {
    use std::{collections::BTreeMap, fs};
    use switchyard_artifacts::Metadata;
    let root =
        std::env::temp_dir().join(format!("switchyard-weight-header-{}", std::process::id()));
    fs::create_dir_all(root.join("source")).unwrap();
    let header = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(42.0_f32.to_le_bytes());
    fs::write(root.join("source/model.safetensors"), &bytes).unwrap();
    fs::write(root.join("source/config.json"), b"{}").unwrap();
    let metadata = Metadata {
        kind: "weights".into(),
        compatibility: BTreeMap::new(),
        attributes: BTreeMap::new(),
    };
    let manifest = switchyard_weights::pack_weights(
        &root.join("source"),
        &root.join("objects"),
        metadata,
        4096,
    )
    .unwrap();
    let entry = manifest
        .entries
        .iter()
        .position(|e| e.path() == "model.safetensors")
        .unwrap();
    let pieces = manifest
        .extents
        .iter()
        .filter(|e| e.entry == entry)
        .collect::<Vec<_>>();
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[0].length, 8 + header.len() as u64);
    assert_eq!(pieces[1].length, 4);
    let reconstructed = pieces
        .iter()
        .flat_map(|e| fs::read(root.join("objects").join(&e.object)).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(reconstructed, bytes);
    let priority = switchyard_weights::priority_extents(&manifest);
    assert!(
        priority.contains(
            &manifest
                .extents
                .iter()
                .position(|e| e.entry == entry && e.offset == 0)
                .unwrap()
        )
    );
    fs::remove_dir_all(root).unwrap();
}
