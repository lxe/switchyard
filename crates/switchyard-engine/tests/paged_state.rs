//! Prefix forks retain immutable attention pages and complete hybrid state.
#[path = "support/mixed.rs"]
mod support;
use candle_core::Tensor;

#[test]
fn restored_prefix_accepts_multitoken_suffix_across_page_boundaries() {
    for value_heads in [2, 4, 6] {
        for page_tokens in [1, 2, 7, 16] {
            let mut weights = support::model(value_heads);
            weights.base_model.set_kv_page_tokens(page_tokens).unwrap();
            let prefix = support::prompt(13, 1);
            let mut prepared = weights.clone();
            prepared.prefill(&support::tensor(&prefix), 0).unwrap();
            let state = prepared.base_model.sequence_states().unwrap().remove(0);
            let before_bytes = state.resident_bytes();
            assert!(before_bytes > 0);
            for suffix in [support::prompt(10, 2), support::prompt(1, 3)] {
                let mut resumed = weights.clone();
                resumed.base_model.restore_sequences(&[&state]).unwrap();
                let actual = resumed
                    .prefill(&support::tensor(&suffix), prefix.len())
                    .unwrap();
                let full = prefix.iter().chain(&suffix).copied().collect::<Vec<_>>();
                let expected = weights.clone().prefill(&support::tensor(&full), 0).unwrap();
                support::close(&actual, &expected);
                assert_eq!(state.position(), 13);
                assert_eq!(state.resident_bytes(), before_bytes);
                assert_eq!(
                    resumed.base_model.state_kv_width().unwrap(),
                    Some(full.len())
                );
            }
        }
    }
}

#[test]
fn forked_prefix_rows_diverge_without_mutating_each_other_or_cached_state() {
    let mut weights = support::model(6);
    weights.base_model.set_kv_page_tokens(7).unwrap();
    let prefix = support::prompt(13, 4);
    let mut prepared = weights.clone();
    prepared.prefill(&support::tensor(&prefix), 0).unwrap();
    let states = prepared.base_model.sequence_states().unwrap();
    let mut merged = weights.clone();
    merged
        .base_model
        .restore_sequences(&[&states[0], &states[0]])
        .unwrap();
    let mut independent = [prepared.clone(), prepared.clone()];
    for step in 0..10 {
        let tokens = [(step % 32) as u32, ((step + 11) % 32) as u32];
        let input = Tensor::new(&[[tokens[0]], [tokens[1]]], &candle_core::Device::Cpu).unwrap();
        let actual = merged.decode_rows(&input, &[13 + step, 13 + step]).unwrap();
        for row in 0..2 {
            let expected = independent[row]
                .forward(&support::tensor(&[tokens[row]]), 13 + step)
                .unwrap();
            support::close(&actual.narrow(0, row, 1).unwrap(), &expected);
        }
    }
    assert_eq!(states[0].position(), 13);
    let mut restored = weights.clone();
    restored
        .base_model
        .restore_sequences(&[&states[0]])
        .unwrap();
    support::close(
        &restored.forward(&support::tensor(&[8]), 13).unwrap(),
        &prepared.forward(&support::tensor(&[8]), 13).unwrap(),
    );
}

#[test]
fn page_size_and_position_validation_preserve_owned_state() {
    let mut model = support::model(2);
    let _factory = support::Factory {
        weights: model.clone(),
        eos: 31,
    };
    assert!(model.base_model.set_kv_page_tokens(0).is_err());
    assert!(model.base_model.set_kv_page_tokens(4097).is_err());
    model.base_model.set_kv_page_tokens(2).unwrap();
    model.prefill(&support::tensor(&[1, 2, 3]), 0).unwrap();
    assert!(model.base_model.set_kv_page_tokens(4).is_err());
    assert!(model.prefill(&support::tensor(&[4]), 2).is_err());
    assert_eq!(model.base_model.state_positions(), &[3]);
    assert_eq!(model.base_model.state_kv_width().unwrap(), Some(3));
    model.clear_kv_cache();
    model.base_model.set_kv_page_tokens(4).unwrap();
}
