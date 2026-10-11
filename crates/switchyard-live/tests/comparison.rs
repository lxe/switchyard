use switchyard_live::workloads::Experiment;

#[test]
fn comparison_corpus_preserves_shared_system_prefix_and_bounds_mixed_outputs() {
    let plan: Experiment = serde_json::from_str(include_str!(
        "../../../configs/runpod/serving-comparison/workloads.json"
    ))
    .unwrap();
    plan.validate().unwrap();
    let shared = plan
        .cases
        .iter()
        .find(|c| c.name == "shared-prefix-burst-32")
        .unwrap();
    let control = plan
        .cases
        .iter()
        .find(|c| c.name == "unique-prefix-burst-32")
        .unwrap();
    assert_eq!(
        shared.request("m", 0).messages[0].content,
        shared.request("m", 1).messages[0].content
    );
    assert_ne!(
        control.request("m", 0).messages[0].content,
        control.request("m", 1).messages[0].content
    );
    let mixed = plan
        .cases
        .iter()
        .find(|c| c.name == "shared-prefix-queue-64")
        .unwrap();
    assert_eq!(
        mixed
            .requests("m")
            .iter()
            .map(|r| r.max_output)
            .take(4)
            .collect::<Vec<_>>(),
        [8, 16, 32, 96]
    );
    assert!(
        mixed
            .requests("m")
            .iter()
            .all(|r| r.planned_output <= r.max_output)
    );
    let mut invalid = plan.clone();
    invalid.cases[0].output_bounds = vec![0];
    assert!(invalid.validate().is_err());
}
