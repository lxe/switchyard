//! Opaque per-row state and packed mixed-length decode. Weights stay shared.
use super::{Model, Positions, TokenMixer};
use candle_core::{Result, Tensor};
use std::{collections::BTreeSet, sync::Arc};

#[derive(Debug, Clone)]
enum LayerState {
    Attention { rows: Vec<super::paged::PagedKv> },
    Gdn { conv: Tensor, recurrent: Tensor },
}
/// State belongs to one immutable model owner. Callers cannot forge its tensors,
/// position or owner. Cloning retains tensor ownership until the final reader drops.
#[derive(Debug, Clone)]
pub struct SequenceState {
    owner: Arc<()>,
    position: usize,
    layers: Vec<LayerState>,
}
impl SequenceState {
    pub fn position(&self) -> usize {
        self.position
    }
    pub(crate) fn belongs_to(&self, model: &Model) -> bool {
        Arc::ptr_eq(&self.owner, &model.state_owner)
            && self.position > 0
            && self.position <= model.max_positions
            && self.layers.len() == model.layers.len()
    }
    /// Conservative retained bytes: shared pages are counted in every state
    /// referencing them, and hybrid recurrent/convolution snapshots are included.
    pub fn resident_bytes(&self) -> usize {
        self.layers.iter().fold(0usize, |bytes, layer| {
            let retained = match layer {
                LayerState::Attention { rows } => rows.iter().map(|row| row.resident_bytes()).sum(),
                LayerState::Gdn { conv, recurrent } => conv
                    .elem_count()
                    .saturating_mul(conv.dtype().size_in_bytes())
                    .saturating_add(
                        recurrent
                            .elem_count()
                            .saturating_mul(recurrent.dtype().size_in_bytes()),
                    ),
            };
            bytes.saturating_add(retained)
        })
    }
}

impl Model {
    pub fn context_limit(&self) -> usize {
        self.max_positions
    }
    pub fn state_kv_width(&self) -> Result<Option<usize>> {
        Ok(self
            .layers
            .iter()
            .find_map(|layer| match &layer.token_mixer {
                TokenMixer::FullAttention(attention) => attention
                    .kv_cache
                    .as_ref()
                    .and_then(|rows| rows.iter().map(|row| row.len).max()),
                TokenMixer::LinearAttention(_) => None,
            }))
    }
    pub fn state_positions(&self) -> &[usize] {
        &self.state_positions
    }

    pub fn sequence_states(&self) -> Result<Vec<SequenceState>> {
        if self.state_positions.is_empty() {
            candle_core::bail!("Cannot export uninitialized sequence state");
        }
        self.state_positions
            .iter()
            .enumerate()
            .map(|(row, &position)| {
                let mut layers = Vec::with_capacity(self.layers.len());
                for layer in &self.layers {
                    layers.push(match &layer.token_mixer {
                        TokenMixer::FullAttention(attention) => {
                            let rows = attention.kv_cache.as_ref().ok_or_else(|| {
                                candle_core::Error::Msg("Missing attention state".into())
                            })?;
                            let cache = rows.get(row).ok_or_else(|| {
                                candle_core::Error::Msg("Missing attention state row".into())
                            })?;
                            if cache.len != position {
                                candle_core::bail!("KV state differs from sequence position");
                            }
                            LayerState::Attention {
                                rows: vec![cache.clone()],
                            }
                        }
                        TokenMixer::LinearAttention(gdn) => LayerState::Gdn {
                            conv: gdn
                                .conv_state
                                .as_ref()
                                .ok_or_else(|| {
                                    candle_core::Error::Msg("Missing convolution state".into())
                                })?
                                .narrow(0, row, 1)?
                                .force_contiguous()?,
                            recurrent: gdn
                                .recurrent_state
                                .as_ref()
                                .ok_or_else(|| {
                                    candle_core::Error::Msg("Missing recurrent state".into())
                                })?
                                .narrow(0, row, 1)?
                                .force_contiguous()?,
                        },
                    });
                }
                Ok(SequenceState {
                    owner: self.state_owner.clone(),
                    position,
                    layers,
                })
            })
            .collect()
    }

    /// Restore shared page tables and compact recurrent state without allocating
    /// leading KV padding or copying the immutable attention prefix.
    pub fn restore_sequences(&mut self, states: &[&SequenceState]) -> Result<()> {
        if states.is_empty() {
            candle_core::bail!("Cannot restore an empty sequence cohort");
        }
        for state in states {
            if !state.belongs_to(self) {
                candle_core::bail!("Foreign model state or incompatible sequence position");
            }
        }
        let mut merged = Vec::with_capacity(self.layers.len());
        for layer in 0..self.layers.len() {
            let attention = matches!(self.layers[layer].token_mixer, TokenMixer::FullAttention(_));
            let mut rows = Vec::new();
            let mut conv = Vec::new();
            let mut recurrent = Vec::new();
            for state in states {
                match (&state.layers[layer], attention) {
                    (LayerState::Attention { rows: cached }, true)
                        if cached.len() == 1 && cached[0].len == state.position =>
                    {
                        rows.push(cached[0].clone())
                    }
                    (
                        LayerState::Gdn {
                            conv: c,
                            recurrent: r,
                        },
                        false,
                    ) => {
                        conv.push(c.clone());
                        recurrent.push(r.clone());
                    }
                    _ => candle_core::bail!("Sequence state layer type mismatch"),
                }
            }
            merged.push(if attention {
                LayerState::Attention { rows }
            } else {
                LayerState::Gdn {
                    conv: Tensor::cat(&conv, 0)?,
                    recurrent: Tensor::cat(&recurrent, 0)?,
                }
            });
        }
        self.replace_layers(merged);
        self.state_positions = states.iter().map(|s| s.position).collect();
        Ok(())
    }

    /// Select surviving page tables and recurrent rows. Selection is validated
    /// before state changes; dropping a row releases its page references.
    pub fn select_sequences(&mut self, rows: &[usize]) -> Result<()> {
        let mut unique = BTreeSet::new();
        if rows
            .iter()
            .any(|&r| r >= self.state_positions.len() || !unique.insert(r))
        {
            candle_core::bail!("Invalid or duplicate sequence row selection");
        }
        if rows.is_empty() {
            self.clear_kv_cache();
            return Ok(());
        }
        let positions: Vec<_> = rows.iter().map(|&r| self.state_positions[r]).collect();
        let ids = rows
            .iter()
            .map(|&r| u32::try_from(r).map_err(candle_core::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        let ids = Tensor::from_vec(ids, rows.len(), &self.device)?;
        let mut selected = Vec::new();
        for layer in &self.layers {
            selected.push(match &layer.token_mixer {
                TokenMixer::FullAttention(attention) => {
                    let cached = attention
                        .kv_cache
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("Missing attention state".into()))?;
                    LayerState::Attention {
                        rows: rows.iter().map(|&row| cached[row].clone()).collect(),
                    }
                }
                TokenMixer::LinearAttention(gdn) => LayerState::Gdn {
                    conv: gdn
                        .conv_state
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("Missing convolution state".into()))?
                        .contiguous()?
                        .index_select(&ids, 0)?,
                    recurrent: gdn
                        .recurrent_state
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("Missing recurrent state".into()))?
                        .contiguous()?
                        .index_select(&ids, 0)?,
                },
            });
        }
        self.replace_layers(selected);
        self.state_positions = positions;
        Ok(())
    }

    fn replace_layers(&mut self, states: Vec<LayerState>) {
        for (layer, state) in self.layers.iter_mut().zip(states) {
            match (&mut layer.token_mixer, state) {
                (TokenMixer::FullAttention(attention), LayerState::Attention { rows }) => {
                    attention.kv_cache = Some(rows)
                }
                (TokenMixer::LinearAttention(gdn), LayerState::Gdn { conv, recurrent }) => {
                    gdn.conv_state = Some(conv);
                    gdn.recurrent_state = Some(recurrent);
                }
                _ => unreachable!("Layer state types were checked before replacement"),
            }
        }
    }

    pub fn decode_rows(&mut self, input_ids: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (rows, length) = input_ids.dims2()?;
        if length != 1
            || rows == 0
            || rows != positions.len()
            || positions != self.state_positions
            || positions.iter().any(|&p| p == 0 || p >= self.max_positions)
        {
            candle_core::bail!(
                "Mixed decode requires one token per owned row at its exact next position"
            );
        }
        let ids = positions
            .iter()
            .map(|&p| u32::try_from(p).map_err(candle_core::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        let ids = Tensor::from_vec(ids, rows, &self.device)?;
        let output =
            self.forward_with_positions(input_ids, None, Positions::Rows(&ids), None, false)?;
        self.state_positions
            .iter_mut()
            .for_each(|position| *position += 1);
        Ok(output)
    }
}
