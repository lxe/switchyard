#[path = "support/mixed.rs"]
mod support;
use candle_core::{Device, Tensor};
use std::collections::BTreeMap;
use switchyard_engine::{
    config::{Config, DeviceConfig, Precision},
    engine::{Engine, Generate},
    models::{MixedModelSession, QwenMixedSession},
    telemetry::Telemetry,
};

#[test]
fn mixed_8_32_96_prefills_match_independent_f32_sessions_after_compaction() {
    for value_heads in [2, 4, 6] {
        let weights = support::model(value_heads);
        let mut session = QwenMixedSession::new(weights.clone(), 3).unwrap();
        let mut independent = BTreeMap::new();
        let mut lengths = BTreeMap::new();
        for (id, length) in [(71, 8), (19, 32), (4, 96)] {
            let prompt = support::prompt(length, id as u32);
            let mut model = weights.clone();
            let expected = model
                .forward(&support::tensor(&prompt), 0)
                .unwrap()
                .squeeze(1)
                .unwrap();
            let actual = session.prefill(&[id], &support::tensor(&prompt)).unwrap();
            support::close(&actual, &expected);
            independent.insert(id, model);
            lengths.insert(id, length);
        }
        let mut rows = vec![4, 71, 19];
        for step in 0..6 {
            let tokens = rows
                .iter()
                .map(|&id| ((id + step * 3) % 32) as u32)
                .collect::<Vec<_>>();
            let positions = rows.iter().map(|id| lengths[id] + step).collect::<Vec<_>>();
            let input = Tensor::from_vec(tokens.clone(), (rows.len(), 1), &Device::Cpu).unwrap();
            let actual = session.decode(&rows, &input, &positions).unwrap();
            for (row, &id) in rows.iter().enumerate() {
                let expected = independent
                    .get_mut(&id)
                    .unwrap()
                    .forward(&support::tensor(&[tokens[row]]), positions[row])
                    .unwrap()
                    .squeeze(1)
                    .unwrap();
                support::close(&actual.narrow(0, row, 1).unwrap(), &expected);
            }
            if step == 1 {
                session.release(&[4]).unwrap();
                rows.retain(|&id| id != 4);
                assert_eq!(session.state_kv_width().unwrap(), Some(34));
            }
        }
        session.release(&rows).unwrap();
        assert_eq!(session.state_rows(), 0);
        assert_eq!(session.state_kv_width().unwrap(), None);
    }
}

#[test]
fn shared_prefill_cohort_exports_distinct_rows_before_mixed_decode() {
    let weights = support::model(6);
    let mut mixed = QwenMixedSession::new(weights.clone(), 4).unwrap();
    let lengths = [8, 32, 8, 96];
    let prompts = lengths
        .iter()
        .enumerate()
        .map(|(row, &length)| support::prompt(length, row as u32))
        .collect::<Vec<_>>();
    let mut independent = Vec::new();
    for prompt in &prompts {
        let mut model = weights.clone();
        model.forward(&support::tensor(prompt), 0).unwrap();
        independent.push(model);
    }
    for rows in [&[0, 2][..], &[1][..], &[3][..]] {
        let input = Tensor::from_vec(
            rows.iter()
                .flat_map(|&row| prompts[row].iter().copied())
                .collect::<Vec<_>>(),
            (rows.len(), lengths[rows[0]]),
            &Device::Cpu,
        )
        .unwrap();
        mixed.prefill(rows, &input).unwrap();
    }
    let mut rows = vec![0, 1, 2, 3];
    for step in 0..3 {
        let ids = rows
            .iter()
            .map(|&row| (row + step + 7) as u32)
            .collect::<Vec<_>>();
        let positions = rows
            .iter()
            .map(|&row| lengths[row] + step)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(ids.clone(), (rows.len(), 1), &Device::Cpu).unwrap();
        let actual = mixed.decode(&rows, &input, &positions).unwrap();
        for (row, &id) in rows.iter().enumerate() {
            let expected = independent[id]
                .forward(&support::tensor(&[ids[row]]), positions[row])
                .unwrap()
                .squeeze(1)
                .unwrap();
            support::close(&actual.narrow(0, row, 1).unwrap(), &expected);
        }
        if step == 0 {
            mixed.release(&[0]).unwrap();
            rows.remove(0);
        }
    }
    mixed.release(&rows).unwrap();
    assert_eq!(mixed.state_rows(), 0);
}

#[test]
fn opaque_state_owner_positions_and_selection_are_checked_without_mutation() {
    let weights = support::model(6);
    let mut a = weights.clone();
    let mut b = weights.clone();
    a.forward(&support::tensor(&support::prompt(8, 1)), 0)
        .unwrap();
    b.forward(&support::tensor(&support::prompt(32, 2)), 0)
        .unwrap();
    let a_state = a.base_model.sequence_states().unwrap();
    let b_state = b.base_model.sequence_states().unwrap();
    let mut merged = weights.clone();
    merged
        .base_model
        .restore_sequences(&[&a_state[0], &b_state[0]])
        .unwrap();
    let before = merged.base_model.state_positions().to_vec();
    assert!(
        merged
            .decode_rows(
                &Tensor::new(&[[1u32], [2]], &Device::Cpu).unwrap(),
                &[8, 31]
            )
            .is_err()
    );
    assert!(merged.base_model.select_sequences(&[0, 0]).is_err());
    assert_eq!(merged.base_model.state_positions(), before);
    let mut foreign = support::model(6);
    assert!(
        foreign
            .base_model
            .restore_sequences(&[&a_state[0]])
            .is_err()
    );
    assert!(foreign.base_model.state_positions().is_empty());
    merged.base_model.select_sequences(&[1, 0]).unwrap();
    let actual = merged
        .decode_rows(
            &Tensor::new(&[[2u32], [1]], &Device::Cpu).unwrap(),
            &[32, 8],
        )
        .unwrap();
    support::close(
        &actual.narrow(0, 0, 1).unwrap(),
        &b.forward(&support::tensor(&[2]), 32).unwrap(),
    );
    support::close(
        &actual.narrow(0, 1, 1).unwrap(),
        &a.forward(&support::tensor(&[1]), 8).unwrap(),
    );
    // Export after packed decode strips left padding and preserves each row's
    // recurrent/conv history, so either row can resume as an independent session.
    let scattered = merged.base_model.sequence_states().unwrap();
    for (row, position, token, independent) in [(0, 33, 5, &mut b), (1, 9, 6, &mut a)] {
        assert_eq!(scattered[row].position(), position);
        let mut resumed = weights.clone();
        resumed
            .base_model
            .restore_sequences(&[&scattered[row]])
            .unwrap();
        let actual = resumed
            .decode_rows(&support::tensor(&[token]), &[position])
            .unwrap();
        support::close(
            &actual,
            &independent
                .forward(&support::tensor(&[token]), position)
                .unwrap(),
        );
    }
    merged.base_model.select_sequences(&[]).unwrap();
    assert!(merged.base_model.sequence_states().is_err());
    let replay = merged
        .forward(&support::tensor(&support::prompt(8, 1)), 0)
        .unwrap();
    support::close(
        &replay,
        &weights
            .clone()
            .forward(&support::tensor(&support::prompt(8, 1)), 0)
            .unwrap(),
    );
}

#[test]
fn mixed_session_rejects_stale_duplicate_unknown_and_late_rows() {
    let weights = support::model(4);
    let mut session = QwenMixedSession::new(weights, 2).unwrap();
    let input = support::tensor(&support::prompt(8, 1));
    session.prefill(&[9], &input).unwrap();
    assert!(session.prefill(&[9], &input).is_err());
    assert!(session.release(&[88]).is_err());
    assert!(session.decode(&[9], &support::tensor(&[1]), &[7]).is_err());
    assert_eq!(session.state_rows(), 1);
    session.decode(&[9], &support::tensor(&[1]), &[8]).unwrap();
    assert!(session.prefill(&[2], &input).is_err());
    assert!(session.decode(&[8], &support::tensor(&[1]), &[9]).is_err());
    session.release(&[9]).unwrap();
    assert_eq!(session.state_rows(), 0);
    assert!(session.decode(&[9], &support::tensor(&[1]), &[9]).is_err());
    let mut session = QwenMixedSession::new(support::model(2), 1).unwrap();
    session.prefill(&[1], &input).unwrap();
    session.release(&[1]).unwrap();
    assert!(session.prefill(&[2], &input).is_err());
}

fn config() -> Config {
    Config {
        model: "tiny".into(),
        model_root: "unused".into(),
        device: DeviceConfig::Cpu,
        dtype: Precision::F32,
        max_context: 160,
        max_weight_bytes: 1024,
        max_batch_size: 3,
        queue_capacity: 6,
        batch_wait_ms: 0,
        cache: Default::default(),
    }
}
fn engine(weights: switchyard_engine::models::qwen35::ModelForCausalLM, eos: u32) -> Engine {
    Engine::new(
        config(),
        Box::new(support::Factory { weights, eos }),
        Telemetry::default(),
    )
    .unwrap()
}

#[test]
fn merged_teacher_forcing_matches_independent_rows_with_different_continuation_lengths() {
    let factory = support::Factory {
        weights: support::model(6),
        eos: 31,
    };
    let traces = [8, 32, 96]
        .into_iter()
        .enumerate()
        .map(|(row, length)| switchyard_engine::probe::Trace {
            prompt: support::prompt(length, row as u32),
            continuation: (0..row * 2 + 2)
                .map(|step| (row as u32 + step as u32 * 3) % 32)
                .collect(),
        })
        .collect::<Vec<_>>();
    let expected = switchyard_engine::probe::probe(&factory, &config(), &traces).unwrap();
    let actual = switchyard_engine::probe::probe_merged(&factory, &config(), &traces).unwrap();
    for (a, e) in actual.iter().zip(&expected) {
        assert_eq!(a.len(), e.len());
        for (a, e) in a.iter().zip(e) {
            assert_eq!(a.position, e.position);
            assert_eq!(a.expected, e.expected);
            assert_eq!(a.top[0].token, e.top[0].token);
            assert!((a.expected_logprob - e.expected_logprob).abs() <= 1e-4);
        }
    }
}

#[test]
fn real_mixed_engine_preserves_output_order_eos_cancellation_and_context_limits() {
    let weights = support::model(6);
    let reference = engine(weights.clone(), 31);
    let requests = [8, 32, 96]
        .into_iter()
        .enumerate()
        .map(|(row, length)| Generate {
            prompt: support::prompt(length, row as u32),
            max_tokens: row + 3,
            ignore_eos: true,
        })
        .collect::<Vec<_>>();
    let singles = requests
        .iter()
        .map(|request| {
            reference
                .generate(std::slice::from_ref(request))
                .unwrap()
                .remove(0)
        })
        .collect::<Vec<_>>();
    let eos = singles[1].token_ids[0];
    let mixed = engine(weights.clone(), eos);
    let mut adjusted = requests.clone();
    adjusted[1].ignore_eos = false;
    let mut callbacks = vec![Vec::new(); 3];
    let result = mixed
        .generate_observed(&adjusted, |row, token, _, _| {
            callbacks[row].push(token);
            row != 2 || callbacks[row].len() < 2
        })
        .unwrap();
    assert_eq!(result[0].token_ids, singles[0].token_ids);
    assert_eq!(result[1].token_ids, [eos]);
    assert_eq!(result[1].finish_reason, "stop");
    assert_eq!(result[2].token_ids, singles[2].token_ids[..2]);
    assert_eq!(result[2].finish_reason, "cancelled");
    for (row, output) in result.iter().enumerate() {
        assert_eq!(callbacks[row], output.token_ids);
        assert!(
            output
                .token_latency_s
                .windows(2)
                .all(|times| times[0] <= times[1])
        );
    }
    let fresh = mixed.generate(&requests).unwrap();
    for (actual, expected) in fresh.iter().zip(&singles) {
        assert_eq!(actual.token_ids, expected.token_ids);
    }
    let mut cohort_callbacks = vec![Vec::new(); requests.len()];
    let cohorts = mixed
        .generate_cohorts_observed(&requests, |row, token, _, _| {
            cohort_callbacks[row].push(token);
            true
        })
        .unwrap();
    for (row, (actual, expected)) in cohorts.iter().zip(&singles).enumerate() {
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.token_ids, cohort_callbacks[row]);
    }
    let mut invalid = requests;
    invalid[1].max_tokens = usize::MAX;
    assert!(mixed.generate(&invalid).is_err());
}

#[test]
fn mixed_engine_keeps_global_observer_rows_across_bounded_cohorts() {
    let weights = support::model(4);
    let engine = engine(weights, 31);
    let requests = [8, 32, 96, 8, 32]
        .into_iter()
        .enumerate()
        .map(|(row, length)| Generate {
            prompt: support::prompt(length, row as u32),
            max_tokens: 3,
            ignore_eos: true,
        })
        .collect::<Vec<_>>();
    let expected = requests
        .iter()
        .map(|r| {
            engine
                .generate(std::slice::from_ref(r))
                .unwrap()
                .remove(0)
                .token_ids
        })
        .collect::<Vec<_>>();
    let mut observed = vec![Vec::new(); 5];
    let output = engine
        .generate_observed(&requests, |row, token, _, _| {
            observed[row].push(token);
            row != 4
        })
        .unwrap();
    for row in 0..5 {
        let count = if row == 4 { 1 } else { 3 };
        assert_eq!(output[row].token_ids, expected[row][..count]);
        assert_eq!(observed[row], output[row].token_ids);
    }
    assert_eq!(output[4].finish_reason, "cancelled");
}
