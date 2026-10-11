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

#[test]
fn native_gauntlet_cycles_input_and_output_work_without_changing_service_classes() {
    let plan: Experiment =
        serde_json::from_str(include_str!("../../../configs/runpod/native-gauntlet.json")).unwrap();
    plan.validate().unwrap();
    let mixed = plan
        .cases
        .iter()
        .find(|c| c.name == "mixed-length-burst")
        .unwrap();
    let requests = mixed.requests("native-model");
    assert_ne!(
        requests[0].messages[0].content,
        requests[1].messages[0].content
    );
    assert_eq!(
        requests
            .iter()
            .take(3)
            .map(|r| r.max_output)
            .collect::<Vec<_>>(),
        [8, 16, 32]
    );
    assert!(requests.iter().all(|r| r.class == "standard"));
    let deps = plan
        .cases
        .iter()
        .find(|c| c.name == "dependencies")
        .unwrap()
        .requests("native-model");
    assert_eq!(deps[1].depends_on, [0]);
    assert_eq!(plan.cases.iter().map(|c| c.calls).sum::<usize>(), 1470);
}
