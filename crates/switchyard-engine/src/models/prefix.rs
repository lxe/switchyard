//! Model-local prefix checkpoints. Retained states include hybrid recurrence;
//! attention pages remain immutable when requests diverge or entries are evicted.
use super::qwen35::{ModelForCausalLM, SequenceState};
use crate::{Error, Result, config::CacheConfig, telemetry::Telemetry};
use candle_core::Tensor;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
struct Checkpoint {
    state: SequenceState,
    logits: Tensor,
}
struct Entry {
    checkpoint: Checkpoint,
    charge: usize,
    touched: u64,
    protected: bool,
}
#[derive(Default)]
struct Retained {
    entries: BTreeMap<Vec<u32>, Entry>,
    bytes: usize,
    clock: u64,
    evictions: u64,
    hits: u64,
}
/// The byte budget charges retained tensor storage and token keys; shared pages
/// are conservatively charged per entry. Host object metadata is bounded by the
/// entry and context limits, rather than measured by this byte counter. Active
/// request state is bounded separately by batch/context admission.
#[derive(Clone)]
pub struct PrefixCache {
    config: CacheConfig,
    retained: Arc<Mutex<Retained>>,
    telemetry: Telemetry,
}
#[derive(Debug, Serialize)]
pub struct PrefixCacheStats {
    pub entries: usize,
    pub charged_bytes: usize,
    pub evictions: u64,
    pub hit_rows: u64,
    pub protected_entries: usize,
}
impl PrefixCache {
    pub(crate) fn enabled(&self) -> bool {
        self.config.enabled()
    }
    pub fn new(config: CacheConfig, telemetry: Telemetry) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            retained: Arc::new(Mutex::new(Retained::default())),
            telemetry,
        })
    }
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Retained>> {
        self.retained
            .lock()
            .map_err(|_| Error::Config("Prefix cache ownership poisoned".into()))
    }
    pub fn stats(&self) -> Result<PrefixCacheStats> {
        let retained = self.lock()?;
        Ok(PrefixCacheStats {
            entries: retained.entries.len(),
            charged_bytes: retained.bytes,
            evictions: retained.evictions,
            hit_rows: retained.hits,
            protected_entries: retained.entries.values().filter(|e| e.protected).count(),
        })
    }
    /// Used by offline comparisons; clearing entries cannot invalidate an active
    /// session because every reader owns its immutable page/state handles.
    pub fn clear(&self) -> Result<()> {
        *self.lock()? = Retained::default();
        Ok(())
    }
    fn longest(&self, tokens: &[u32]) -> Result<Option<Checkpoint>> {
        let mut retained = self.lock()?;
        let key = retained
            .entries
            .keys()
            .filter(|key| tokens.starts_with(key))
            .max_by_key(|key| key.len())
            .cloned();
        let Some(key) = key else {
            return Ok(None);
        };
        retained.clock = retained.clock.wrapping_add(1);
        retained.hits += 1;
        let clock = retained.clock;
        let entry = retained
            .entries
            .get_mut(&key)
            .expect("Selected owned checkpoint");
        entry.touched = clock;
        Ok(Some(entry.checkpoint.clone()))
    }
    fn insert(&self, tokens: &[u32], checkpoint: Checkpoint, shared: bool) -> Result<()> {
        let charge = checkpoint
            .state
            .resident_bytes()
            .checked_add(checkpoint.logits.elem_count() * checkpoint.logits.dtype().size_in_bytes())
            .and_then(|n| n.checked_add(std::mem::size_of_val(tokens)));
        let Some(charge) = charge.filter(|&n| n <= self.config.max_bytes) else {
            return Ok(());
        };
        let mut retained = self.lock()?;
        if let Some(old) = retained.entries.remove(tokens) {
            retained.bytes -= old.charge;
        }
        // Shared prefixes receive a protected LRU segment capped at half of
        // each retention bound. Individual endings cannot immediately evict a
        // common system prompt, while new shared workloads age old ones out.
        let protected =
            shared && charge <= self.config.max_bytes / 2 && self.config.max_entries >= 2;
        if protected {
            loop {
                let entries = retained
                    .entries
                    .values()
                    .filter(|e| e.protected)
                    .collect::<Vec<_>>();
                if entries.len() < self.config.max_entries / 2
                    && entries.iter().map(|e| e.charge).sum::<usize>()
                        <= self.config.max_bytes / 2 - charge
                {
                    break;
                }
                let key = retained
                    .entries
                    .iter()
                    .filter(|(_, e)| e.protected)
                    .min_by_key(|(_, e)| e.touched)
                    .map(|(key, _)| key.clone())
                    .expect("Protected segment before demotion");
                retained
                    .entries
                    .get_mut(&key)
                    .expect("Selected demotion")
                    .protected = false;
            }
        }
        while retained.entries.len() >= self.config.max_entries
            || charge > self.config.max_bytes - retained.bytes
        {
            let key = retained
                .entries
                .iter()
                .min_by_key(|(_, e)| (e.protected, e.touched))
                .map(|(key, _)| key.clone())
                .expect("Nonempty cache before eviction");
            let old = retained.entries.remove(&key).expect("Selected eviction");
            retained.bytes -= old.charge;
            retained.evictions += 1;
        }
        retained.clock = retained.clock.wrapping_add(1);
        let touched = retained.clock;
        retained.entries.insert(
            tokens.to_vec(),
            Entry {
                checkpoint,
                charge,
                touched,
                protected,
            },
        );
        retained.bytes += charge;
        Ok(())
    }
    /// Input rows have equal prompt length. A stage deduplicates exact token
    /// prefixes, executes distinct suffixes as a batch, then splits checkpoints.
    /// The short lock protects bookkeeping only; no GPU work occurs under it.
    pub(crate) fn prefill(
        &self,
        template: &ModelForCausalLM,
        tokens: &Tensor,
    ) -> Result<(Tensor, Vec<SequenceState>)> {
        if !self.config.enabled() {
            let mut model = template.clone();
            model.clear_kv_cache();
            let logits = model.forward(tokens, 0)?.squeeze(1)?;
            return Ok((logits, model.base_model.sequence_states()?));
        }
        let prompts = tokens.to_vec2::<u32>()?;
        let (batch, length) = tokens.dims2()?;
        if batch == 0 || length == 0 {
            return Err(Error::Config("Empty cached prefill".into()));
        }
        let mut rows = prompts
            .iter()
            .map(|p| self.longest(p))
            .collect::<Result<Vec<_>>>()?;
        if rows
            .iter()
            .flatten()
            .any(|c| !c.state.belongs_to(&template.base_model))
        {
            return Err(Error::Config(
                "Prefix cache belongs to a different model instance".into(),
            ));
        }
        let retained_hits = rows.iter().filter(|r| r.is_some()).count();
        let mut processed = 0usize;
        loop {
            let position = rows
                .iter()
                .filter_map(|r| {
                    let position = r.as_ref().map_or(0, |c| c.state.position());
                    (position < length).then_some(position)
                })
                .min();
            let Some(position) = position else {
                break;
            };
            let end = ((position / self.config.checkpoint_tokens + 1)
                * self.config.checkpoint_tokens)
                .min(length);
            let mut groups: BTreeMap<&[u32], Vec<usize>> = BTreeMap::new();
            for (row, prompt) in prompts.iter().enumerate() {
                if rows[row].as_ref().map_or(0, |c| c.state.position()) == position {
                    groups.entry(&prompt[..end]).or_default().push(row);
                }
            }
            let representatives = groups.values().map(|group| group[0]).collect::<Vec<_>>();
            let mut cohort = template.clone();
            cohort.clear_kv_cache();
            cohort
                .base_model
                .set_kv_page_tokens(self.config.page_tokens)?;
            if position > 0 {
                let states = representatives
                    .iter()
                    .map(|&r| &rows[r].as_ref().expect("Owned prefix").state)
                    .collect::<Vec<_>>();
                cohort.base_model.restore_sequences(&states)?;
            }
            let ids = representatives
                .iter()
                .flat_map(|&r| prompts[r][position..end].iter().copied())
                .collect::<Vec<_>>();
            let suffix = Tensor::from_vec(
                ids,
                (representatives.len(), end - position),
                tokens.device(),
            )?;
            let logits = cohort.prefill(&suffix, position)?.squeeze(1)?;
            let states = cohort.base_model.sequence_states()?;
            processed += representatives.len() * (end - position);
            for (index, (prefix, members)) in groups.into_iter().enumerate() {
                let checkpoint = Checkpoint {
                    state: states[index].clone(),
                    logits: logits.narrow(0, index, 1)?.force_contiguous()?,
                };
                self.insert(prefix, checkpoint.clone(), members.len() > 1)?;
                for row in members {
                    rows[row] = Some(checkpoint.clone());
                }
            }
        }
        let rows = rows
            .into_iter()
            .map(|r| r.expect("Every prefill reached prompt end"))
            .collect::<Vec<_>>();
        self.telemetry.event(
            "prefix_prefill",
            serde_json::json!({
                "rows":batch, "input_tokens":batch * length, "computed_tokens":processed,
                "reused_tokens":batch * length - processed, "retained_hit_rows":retained_hits,
                "cache":self.stats()?,
            }),
        );
        Ok((
            Tensor::cat(&rows.iter().map(|r| &r.logits).collect::<Vec<_>>(), 0)?,
            rows.into_iter().map(|r| r.state).collect(),
        ))
    }
}
