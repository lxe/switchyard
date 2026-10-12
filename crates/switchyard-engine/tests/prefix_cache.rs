#[path = "support/mixed.rs"]
#[allow(dead_code)]
mod support;
use candle_core::{Device, Tensor};
use switchyard_engine::{
    config::CacheConfig,
    models::{MixedModelSession, ModelSession, PrefixCache, QwenMixedSession, QwenSession},
    telemetry::Telemetry,
};

fn make_cache(entries: usize, bytes: usize) -> PrefixCache {
    PrefixCache::new(
        CacheConfig {
            page_tokens: 4,
            checkpoint_tokens: 4,
            max_entries: entries,
            max_bytes: bytes,
        },
        Telemetry::default(),
    )
    .unwrap()
}
fn session(
    weights: &switchyard_engine::models::qwen35::ModelForCausalLM,
    cache: &PrefixCache,
) -> QwenSession {
    QwenSession::with_cache(weights.clone(), cache.clone())
}

#[test]
fn cached_shared_prefixes_diverge_and_exact_hits_preserve_logits_and_decode() {
    for value_heads in [2, 4, 6] {
        let weights = support::model(value_heads);
        let cache = make_cache(32, 4 * 1024 * 1024);
        let prefix = support::prompt(9, 2);
        let mut first = session(&weights, &cache);
        let first_logits = first.forward(&support::tensor(&prefix), 0).unwrap();
        let mut exact = session(&weights, &cache);
        support::close(
            &exact.forward(&support::tensor(&prefix), 0).unwrap(),
            &first_logits,
        );
        for suffix in [vec![3], vec![4, 5, 6], vec![9, 1, 7, 2, 3]] {
            let mut prompt = prefix.clone();
            prompt.extend(suffix);
            let mut reference = weights.clone();
            let expected = reference
                .forward(&support::tensor(&prompt), 0)
                .unwrap()
                .squeeze(1)
                .unwrap();
            let mut cached = session(&weights, &cache);
            support::close(
                &cached.forward(&support::tensor(&prompt), 0).unwrap(),
                &expected,
            );
            for step in 0..4 {
                let input = support::tensor(&[step as u32 + 3]);
                let actual = cached.forward(&input, prompt.len() + step).unwrap();
                let expected = reference
                    .forward(&input, prompt.len() + step)
                    .unwrap()
                    .squeeze(1)
                    .unwrap();
                support::close(&actual, &expected);
            }
        }
        // Diverging readers did not alter either the checkpoint or the first session.
        let input = support::tensor(&[8]);
        support::close(
            &first.forward(&input, prefix.len()).unwrap(),
            &exact.forward(&input, prefix.len()).unwrap(),
        );
    }
}

#[test]
fn within_batch_prefix_deduplication_matches_independent_rows() {
    let weights = support::model(6);
    let cache = make_cache(16, 4 * 1024 * 1024);
    let mut prompts = vec![support::prompt(12, 2); 4];
    prompts[1][8] = 7;
    prompts[2][10] = 4;
    let input = Tensor::from_vec(
        prompts.iter().flatten().copied().collect::<Vec<_>>(),
        (4, 12),
        &Device::Cpu,
    )
    .unwrap();
    let actual = session(&weights, &cache).forward(&input, 0).unwrap();
    for (row, prompt) in prompts.iter().enumerate() {
        let expected = weights
            .clone()
            .forward(&support::tensor(prompt), 0)
            .unwrap()
            .squeeze(1)
            .unwrap();
        support::close(&actual.narrow(0, row, 1).unwrap(), &expected);
    }
    assert!(cache.stats().unwrap().entries < 4 * 3);
}

#[test]
fn eviction_and_clear_do_not_invalidate_live_readers_and_retention_is_bounded() {
    let weights = support::model(4);
    let cache = make_cache(1, 128 * 1024);
    let prompt = support::prompt(9, 2);
    let mut live = session(&weights, &cache);
    live.forward(&support::tensor(&prompt), 0).unwrap();
    let mut reference = weights.clone();
    reference.forward(&support::tensor(&prompt), 0).unwrap();
    for seed in 3..9 {
        session(&weights, &cache)
            .forward(&support::tensor(&support::prompt(13, seed)), 0)
            .unwrap();
        let stats = cache.stats().unwrap();
        assert!(stats.entries <= 1 && stats.charged_bytes <= 128 * 1024);
    }
    assert!(cache.stats().unwrap().evictions > 0);
    cache.clear().unwrap();
    assert_eq!(cache.stats().unwrap().charged_bytes, 0);
    support::close(
        &live.forward(&support::tensor(&[4]), prompt.len()).unwrap(),
        &reference
            .forward(&support::tensor(&[4]), prompt.len())
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let tiny = make_cache(8, 1);
    session(&weights, &tiny)
        .forward(&support::tensor(&prompt), 0)
        .unwrap();
    assert_eq!(tiny.stats().unwrap().entries, 0);
}

#[test]
fn cached_mixed_rows_preserve_release_and_private_recurrence() {
    let weights = support::model(6);
    let cache = make_cache(16, 4 * 1024 * 1024);
    let mut mixed = QwenMixedSession::with_cache(weights.clone(), 3, cache).unwrap();
    let mut independent = Vec::new();
    for (row, length) in [8, 12, 16].into_iter().enumerate() {
        let prompt = support::prompt(length, 2);
        let mut reference = weights.clone();
        let expected = reference
            .forward(&support::tensor(&prompt), 0)
            .unwrap()
            .squeeze(1)
            .unwrap();
        support::close(
            &mixed.prefill(&[row], &support::tensor(&prompt)).unwrap(),
            &expected,
        );
        independent.push(reference);
    }
    mixed.release(&[1]).unwrap();
    for step in 0..4 {
        let ids = Tensor::new(&[[3u32], [5]], &Device::Cpu).unwrap();
        let actual = mixed.decode(&[0, 2], &ids, &[8 + step, 16 + step]).unwrap();
        for (index, row, token, position) in [(0, 0, 3, 8 + step), (1, 2, 5, 16 + step)] {
            let expected = independent[row]
                .forward(&support::tensor(&[token]), position)
                .unwrap()
                .squeeze(1)
                .unwrap();
            support::close(&actual.narrow(0, index, 1).unwrap(), &expected);
        }
    }
    mixed.release(&[0, 2]).unwrap();
    assert_eq!(mixed.state_rows(), 0);
}

#[test]
fn prefix_bounds_reject_partial_disable_and_invalid_granularity() {
    for cfg in [
        CacheConfig {
            max_entries: 1,
            ..Default::default()
        },
        CacheConfig {
            page_tokens: 0,
            ..Default::default()
        },
        CacheConfig {
            checkpoint_tokens: 4097,
            ..Default::default()
        },
    ] {
        assert!(cfg.validate().is_err());
    }
}

#[test]
fn foreign_model_cannot_read_cached_logits_or_state() {
    let cache = make_cache(8, 4 * 1024 * 1024);
    let input = support::tensor(&support::prompt(8, 2));
    session(&support::model(2), &cache)
        .forward(&input, 0)
        .unwrap();
    assert!(
        session(&support::model(2), &cache)
            .forward(&input, 0)
            .is_err()
    );
}

#[test]
fn bf16_cached_multitoken_suffix_has_compatible_causal_mask_and_prompt_normalization() {
    let weights = support::model_dtype(6, candle_core::DType::BF16);
    let cache = make_cache(16, 4 * 1024 * 1024);
    let prompt = support::prompt(13, 2);
    let mut cached = session(&weights, &cache);
    let actual = cached.forward(&support::tensor(&prompt), 0).unwrap();
    let expected = weights
        .clone()
        .prefill(&support::tensor(&prompt), 0)
        .unwrap()
        .squeeze(1)
        .unwrap();
    let error = (actual.to_dtype(candle_core::DType::F32).unwrap()
        - expected.to_dtype(candle_core::DType::F32).unwrap())
    .unwrap()
    .abs()
    .unwrap()
    .max_all()
    .unwrap()
    .to_scalar::<f32>()
    .unwrap();
    assert!(error <= 0.002, "BF16 cached prefill error {error}");
    let mut exact = session(&weights, &cache);
    let hit = exact.forward(&support::tensor(&prompt), 0).unwrap();
    assert_eq!(
        hit.to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap(),
        actual
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
    );
}

#[test]
fn bounded_shared_prefix_protection_survives_sibling_ending_evictions() {
    let weights = support::model(6);
    let mut measured = weights.clone();
    measured
        .prefill(&support::tensor(&support::prompt(8, 2)), 0)
        .unwrap();
    let charge =
        measured.base_model.sequence_states().unwrap()[0].resident_bytes() + 32 * 4 + 8 * 4;
    let cache = make_cache(4, charge * 4);
    let prompts = (3u32..8)
        .map(|tail| vec![1, 2, 3, 4, tail, 9, 10, 11])
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(
        prompts.iter().flatten().copied().collect::<Vec<_>>(),
        (5, 8),
        &Device::Cpu,
    )
    .unwrap();
    let first = session(&weights, &cache).forward(&input, 0).unwrap();
    let before = cache.stats().unwrap();
    assert!(before.evictions > 0 && before.protected_entries > 0);
    let repeat = session(&weights, &cache).forward(&input, 0).unwrap();
    let after = cache.stats().unwrap();
    assert_eq!(after.hit_rows - before.hit_rows, 5);
    assert!(after.entries <= 4 && after.charged_bytes <= charge * 4);
    support::close(&first, &repeat);
}
