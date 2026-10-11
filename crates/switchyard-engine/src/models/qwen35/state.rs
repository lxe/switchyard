//! Opaque per-row state and packed mixed-length decode. Weights stay shared.
use super::{Model, Positions, TokenMixer};
use candle_core::{Result, Tensor};
use std::{collections::BTreeSet, sync::Arc};

#[derive(Debug, Clone)]
enum LayerState {
    Attention { key: Tensor, value: Tensor },
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
}

impl Model {
    pub fn context_limit(&self) -> usize {
        self.max_positions
    }
    pub fn state_kv_width(&self) -> Result<Option<usize>> {
        self.layers
            .iter()
            .find_map(|layer| match &layer.token_mixer {
                TokenMixer::FullAttention(attention) => Some(
                    attention
                        .kv_cache
                        .as_ref()
                        .map(|(k, _)| k.dim(2))
                        .transpose(),
                ),
                TokenMixer::LinearAttention(_) => None,
            })
            .unwrap_or(Ok(None))
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
                            let (key, value) = attention.kv_cache.as_ref().ok_or_else(|| {
                                candle_core::Error::Msg("Missing attention state".into())
                            })?;
                            let total = key.dim(2)?;
                            if total < position {
                                candle_core::bail!("KV state shorter than sequence position");
                            }
                            LayerState::Attention {
                                key: key
                                    .narrow(0, row, 1)?
                                    .narrow(2, total - position, position)?
                                    .contiguous()?,
                                value: value
                                    .narrow(0, row, 1)?
                                    .narrow(2, total - position, position)?
                                    .contiguous()?,
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
                                .contiguous()?,
                            recurrent: gdn
                                .recurrent_state
                                .as_ref()
                                .ok_or_else(|| {
                                    candle_core::Error::Msg("Missing recurrent state".into())
                                })?
                                .narrow(0, row, 1)?
                                .contiguous()?,
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

    /// Gather normalized per-row state once after independent prefills. Leading
    /// KV padding is private to this packed cohort and masked during attention.
    pub fn restore_sequences(&mut self, states: &[&SequenceState]) -> Result<()> {
        if states.is_empty() {
            candle_core::bail!("Cannot restore an empty sequence cohort");
        }
        for state in states {
            if !Arc::ptr_eq(&self.state_owner, &state.owner)
                || state.position == 0
                || state.position > self.max_positions
                || state.layers.len() != self.layers.len()
            {
                candle_core::bail!("Foreign model state or incompatible sequence position");
            }
        }
        let longest = states.iter().map(|s| s.position).max().unwrap();
        // Construct all replacement tensors before modifying the model.
        let mut merged = Vec::with_capacity(self.layers.len());
        for layer in 0..self.layers.len() {
            let mut first = Vec::new();
            let mut second = Vec::new();
            let attention = matches!(self.layers[layer].token_mixer, TokenMixer::FullAttention(_));
            for state in states {
                match (&state.layers[layer], attention) {
                    (LayerState::Attention { key, value }, true) => {
                        let padding = longest - state.position;
                        let pad = |tensor: &Tensor| -> Result<Tensor> {
                            if padding == 0 {
                                return Ok(tensor.clone());
                            }
                            let zeros = Tensor::zeros(
                                (1, tensor.dim(1)?, padding, tensor.dim(3)?),
                                tensor.dtype(),
                                tensor.device(),
                            )?;
                            Tensor::cat(&[&zeros, tensor], 2)
                        };
                        first.push(pad(key)?);
                        second.push(pad(value)?);
                    }
                    (LayerState::Gdn { conv, recurrent }, false) => {
                        first.push(conv.clone());
                        second.push(recurrent.clone());
                    }
                    _ => candle_core::bail!("Sequence state layer type mismatch"),
                }
            }
            let first = Tensor::cat(&first, 0)?;
            let second = Tensor::cat(&second, 0)?;
            merged.push(if attention {
                LayerState::Attention {
                    key: first,
                    value: second,
                }
            } else {
                LayerState::Gdn {
                    conv: first,
                    recurrent: second,
                }
            });
        }
        self.replace_layers(merged);
        self.state_positions = states.iter().map(|s| s.position).collect();
        Ok(())
    }

    /// Gather surviving rows and crop leading KV storage when the longest row
    /// leaves. Selection is validated before any state tensor changes.
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
        let longest = *positions.iter().max().unwrap();
        let ids = rows
            .iter()
            .map(|&r| u32::try_from(r).map_err(candle_core::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        let ids = Tensor::from_vec(ids, rows.len(), &self.device)?;
        let mut selected = Vec::new();
        for layer in &self.layers {
            selected.push(match &layer.token_mixer {
                TokenMixer::FullAttention(attention) => {
                    let (key, value) = attention
                        .kv_cache
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("Missing attention state".into()))?;
                    let total = key.dim(2)?;
                    LayerState::Attention {
                        key: key
                            .index_select(&ids, 0)?
                            .narrow(2, total - longest, longest)?
                            .contiguous()?,
                        value: value
                            .index_select(&ids, 0)?
                            .narrow(2, total - longest, longest)?
                            .contiguous()?,
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
                (TokenMixer::FullAttention(attention), LayerState::Attention { key, value }) => {
                    attention.kv_cache = Some((key, value))
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
        let longest = *positions.iter().max().unwrap();
        let mut mask = Vec::with_capacity(rows * (longest + 1));
        for &position in positions {
            mask.extend((0..=longest).map(|column| {
                if column < longest - position {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            }));
        }
        let mask = Tensor::from_vec(mask, (rows, 1, 1, longest + 1), &self.device)?;
        let ids = positions
            .iter()
            .map(|&p| u32::try_from(p).map_err(candle_core::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        let ids = Tensor::from_vec(ids, rows, &self.device)?;
        let output =
            self.forward_with_positions(input_ids, Some(&mask), Positions::Rows(&ids), None)?;
        self.state_positions
            .iter_mut()
            .for_each(|position| *position += 1);
        Ok(output)
    }
}
