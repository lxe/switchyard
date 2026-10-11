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
