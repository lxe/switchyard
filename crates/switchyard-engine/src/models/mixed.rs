//! Per-cohort request/state ownership, separate from Qwen mathematical operations.
use super::{
    MixedModelSession,
    qwen35::{ModelForCausalLM, SequenceState},
};
use crate::{Error, Result};
use candle_core::Tensor;
use std::collections::{BTreeMap, BTreeSet};

pub struct QwenMixedSession {
    model: ModelForCausalLM,
    pending: BTreeMap<usize, SequenceState>,
    seen: BTreeSet<usize>,
    active: Vec<usize>,
    max_rows: usize,
    decoding: bool,
}
impl QwenMixedSession {
    pub fn new(mut model: ModelForCausalLM, max_rows: usize) -> Result<Self> {
        if max_rows == 0 || max_rows > u32::MAX as usize {
            return Err(Error::Config("Invalid mixed session row capacity".into()));
        }
        model.clear_kv_cache();
        Ok(Self {
            model,
            pending: BTreeMap::new(),
            seen: BTreeSet::new(),
            active: Vec::new(),
            max_rows,
            decoding: false,
        })
    }
    pub fn state_rows(&self) -> usize {
        if self.decoding {
            self.active.len()
        } else {
            self.pending.len()
        }
    }
    pub fn state_kv_width(&self) -> Result<Option<usize>> {
        Ok(self.model.base_model.state_kv_width()?)
    }
    fn unique(rows: &[usize]) -> bool {
        let mut seen = BTreeSet::new();
        rows.iter().all(|&r| seen.insert(r))
    }
}
impl MixedModelSession for QwenMixedSession {
    fn prefill(&mut self, rows: &[usize], tokens: &Tensor) -> Result<Tensor> {
        let (batch, length) = tokens.dims2()?;
        if self.decoding
            || batch == 0
            || length == 0
            || batch != rows.len()
            || !Self::unique(rows)
            || rows.iter().any(|r| self.seen.contains(r))
            || self
                .seen
                .len()
                .checked_add(batch)
                .is_none_or(|n| n > self.max_rows)
        {
            return Err(Error::Config(
                "Mixed prefill requires fresh distinct IDs within cohort capacity, before decode"
                    .into(),
            ));
        }
        let mut cohort = self.model.clone();
        cohort.clear_kv_cache();
        let logits = cohort.forward(tokens, 0)?.squeeze(1)?;
        let states = cohort.base_model.sequence_states()?;
        for (&id, state) in rows.iter().zip(states) {
            self.pending.insert(id, state);
            self.seen.insert(id);
        }
        Ok(logits)
    }
    fn decode(&mut self, rows: &[usize], tokens: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (batch, length) = tokens.dims2()?;
        if length != 1
            || batch == 0
            || batch != rows.len()
            || positions.len() != batch
            || !Self::unique(rows)
            || positions
                .iter()
                .any(|&p| p == 0 || p >= self.model.base_model.context_limit())
        {
            return Err(Error::Config(
                "Mixed decode row/position/token shape mismatch".into(),
            ));
        }
        if !self.decoding {
            if rows.len() != self.pending.len() {
                return Err(Error::Config(
                    "Release omitted prefill rows before joining decode".into(),
                ));
            }
            let states = rows
                .iter()
                .map(|id| {
                    self.pending
                        .get(id)
                        .ok_or_else(|| Error::Config("Unknown prefill row".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            if states
                .iter()
                .zip(positions)
                .any(|(state, &p)| state.position() != p)
            {
                return Err(Error::Config("Prefill/decode position drift".into()));
            }
            self.model.base_model.restore_sequences(&states)?;
            self.active = rows.to_vec();
            self.pending.clear();
            self.decoding = true;
        } else if rows != self.active {
            return Err(Error::Config(
                "Mixed decode rows must retain active order; release finished rows explicitly"
                    .into(),
            ));
        }
        Ok(self.model.decode_rows(tokens, positions)?.squeeze(1)?)
    }
    fn release(&mut self, rows: &[usize]) -> Result<()> {
        if !Self::unique(rows)
            || rows.iter().any(|id| {
                if self.decoding {
                    !self.active.contains(id)
                } else {
                    !self.pending.contains_key(id)
                }
            })
        {
            return Err(Error::Config("Unknown or duplicate state release".into()));
        }
        if rows.is_empty() {
            return Ok(());
        }
        if self.decoding {
            let retained: Vec<_> = self
                .active
                .iter()
                .enumerate()
                .filter_map(|(row, id)| (!rows.contains(id)).then_some(row))
                .collect();
            self.model.base_model.select_sequences(&retained)?;
            self.active.retain(|id| !rows.contains(id));
        } else {
            for id in rows {
                self.pending.remove(id);
            }
        }
        Ok(())
    }
}
