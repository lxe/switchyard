//! A bounded handoff for merging independently-prefilled decode rows.
//! State is owned by the model adapter, not by this scheduling plan.
use crate::{Error, Result};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateLease {
    pub slot: usize,
    pub generation: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeRow {
    pub request_id: u64,
    pub state: StateLease,
    pub token: u32,
    /// Absolute position of the next input token, after that row's own prefill.
    pub position: usize,
    pub remaining_outputs: usize,
}
#[derive(Debug)]
pub struct DecodePlan {
    rows: Vec<DecodeRow>,
    max_context: usize,
}
impl DecodePlan {
    pub fn new(rows: Vec<DecodeRow>, max_rows: usize, max_context: usize) -> Result<Self> {
        if rows.is_empty() || rows.len() > max_rows {
            return Err(Error::Limit("decode row capacity".into()));
        }
        let mut ids = BTreeSet::new();
        let mut slots = BTreeSet::new();
        for row in &rows {
            if !ids.insert(row.request_id) || !slots.insert(row.state.slot) {
                return Err(Error::Invalid(
                    "decode request IDs and state slots must be unique".into(),
                ));
            }
            if row.remaining_outputs == 0
                || row
                    .position
                    .checked_add(row.remaining_outputs)
                    .is_none_or(|end| end > max_context)
            {
                return Err(Error::Limit("per-row decode context/output bounds".into()));
            }
        }
        Ok(Self { rows, max_context })
    }
    pub fn rows(&self) -> &[DecodeRow] {
        &self.rows
    }
    /// Applies outputs in plan order, retaining caller IDs and independent
    /// positions. Completed leases are returned for event-protected release.
    /// A lease is not permission to reuse the adapter's still-in-flight state.
    pub fn advance(&mut self, tokens: &[u32], stop: &[bool]) -> Result<Vec<StateLease>> {
        if tokens.len() != self.rows.len() || stop.len() != self.rows.len() {
            return Err(Error::Invalid("decode result row count".into()));
        }
        let mut released = Vec::new();
        for ((row, &token), &stop) in self.rows.iter_mut().zip(tokens).zip(stop) {
            row.token = token;
            row.position += 1;
            row.remaining_outputs -= 1;
            if stop || row.remaining_outputs == 0 || row.position >= self.max_context {
                released.push(row.state);
                row.remaining_outputs = 0;
            }
        }
        self.rows.retain(|row| row.remaining_outputs > 0);
        Ok(released)
    }
}
