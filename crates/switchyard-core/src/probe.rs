//! Bounded, non-overlapping bandwidth samples from matching received-byte counters.
use crate::ProbePolicy;
use serde::Serialize;
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TransferCounter {
    pub source: String,
    pub total_bytes: u64,
    pub received_bytes: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct BandwidthSample {
    pub started_s: f64,
    pub at_s: f64,
    pub received_bytes: u64,
    pub bytes_per_s: f64,
}
#[derive(Debug, Default)]
pub struct BandwidthWindow {
    samples: VecDeque<(f64, Vec<TransferCounter>)>,
}
impl BandwidthWindow {
    /// Return true when an existing counter domain or timeline was reset.
    pub fn record(
        &mut self,
        at_s: f64,
        counters: Vec<TransferCounter>,
        policy: &ProbePolicy,
    ) -> bool {
        if !at_s.is_finite() || at_s < 0.0 || counters.is_empty() {
            self.samples.clear();
            return true;
        }
        let mut reset = false;
        if let Some((last_at, last)) = self.samples.back() {
            if at_s < *last_at || !matching(last, &counters) {
                self.samples.clear();
                reset = true;
            } else if at_s == *last_at {
                return false;
            }
        }
        while self
            .samples
            .front()
            .is_some_and(|(at, _)| at_s - at > policy.maximum_window_s)
        {
            self.samples.pop_front();
        }
        // Bound memory even if a caller polls much faster than its probe policy.
        if self.samples.len() == 256 {
            self.samples.pop_front();
        }
        self.samples.push_back((at_s, counters));
        reset
    }
    pub fn qualify(&mut self, policy: &ProbePolicy) -> Option<BandwidthSample> {
        let (end, current) = self.samples.back()?;
        let result = self
            .samples
            .iter()
            .rev()
            .skip(1)
            .find_map(|(start, previous)| {
                let duration = end - start;
                if duration < policy.minimum_window_s || duration > policy.maximum_window_s {
                    return None;
                }
                let bytes = current
                    .iter()
                    .zip(previous)
                    .try_fold(0u64, |sum, (new, old)| {
                        sum.checked_add(new.received_bytes.checked_sub(old.received_bytes)?)
                    })?;
                let rate = bytes as f64 / duration;
                (bytes >= policy.minimum_received_bytes && rate.is_finite() && rate > 0.0)
                    .then_some(BandwidthSample {
                        started_s: *start,
                        at_s: *end,
                        received_bytes: bytes,
                        bytes_per_s: rate,
                    })
            })?;
        // A later sample must measure new payload, rather than relearning overlap.
        let latest = self.samples.pop_back().expect("sampled endpoint");
        self.samples.clear();
        self.samples.push_back(latest);
        Some(result)
    }
}
fn matching(old: &[TransferCounter], new: &[TransferCounter]) -> bool {
    old.len() == new.len()
        && old.iter().zip(new).all(|(old, new)| {
            old.source == new.source
                && old.total_bytes == new.total_bytes
                && new.received_bytes >= old.received_bytes
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn counters(model: u64, artifact: u64) -> Vec<TransferCounter> {
        vec![
            TransferCounter {
                source: "model".into(),
                total_bytes: 1_000,
                received_bytes: model,
            },
            TransferCounter {
                source: "runtime".into(),
                total_bytes: 10_000,
                received_bytes: artifact,
            },
        ]
    }
    fn policy() -> ProbePolicy {
        ProbePolicy {
            minimum_received_bytes: 100,
            minimum_window_s: 2.0,
            maximum_window_s: 5.0,
            ..ProbePolicy::default()
        }
    }
    #[test]
    fn metadata_does_not_qualify_and_new_payload_replaces_early_rate() {
        let mut w = BandwidthWindow::default();
        let p = policy();
        w.record(0.0, counters(0, 0), &p);
        w.record(0.1, counters(10, 0), &p);
        assert!(w.qualify(&p).is_none());
        w.record(2.0, counters(10, 0), &p);
        assert!(w.qualify(&p).is_none());
        w.record(3.0, counters(10, 300), &p);
        let first = w.qualify(&p).unwrap();
        assert_eq!(first.received_bytes, 300);
        assert_eq!(first.bytes_per_s, 300.0 / 2.9);
        assert!(w.qualify(&p).is_none());
        w.record(5.0, counters(10, 500), &p);
        assert_eq!(w.qualify(&p).unwrap().bytes_per_s, 100.0);
    }
    #[test]
    fn rollback_identity_and_total_changes_reset_each_channel() {
        let p = policy();
        for changed in [
            counters(0, 500),
            {
                let mut c = counters(100, 500);
                c[0].source = "different".into();
                c
            },
            {
                let mut c = counters(100, 500);
                c[1].total_bytes += 1;
                c
            },
        ] {
            let mut w = BandwidthWindow::default();
            w.record(0.0, counters(50, 0), &p);
            assert!(w.record(2.0, changed, &p));
            assert!(w.qualify(&p).is_none());
        }
        let mut w = BandwidthWindow::default();
        w.record(2.0, counters(50, 0), &p);
        assert!(w.record(1.0, counters(100, 500), &p));
        assert!(w.qualify(&p).is_none());
    }
    #[test]
    fn expired_anchors_and_insufficient_payload_never_qualify() {
        let p = policy();
        let mut w = BandwidthWindow::default();
        w.record(0.0, counters(0, 0), &p);
        w.record(6.0, counters(500, 0), &p);
        assert!(w.qualify(&p).is_none());
        w.record(8.0, counters(500, 0), &p);
        assert!(w.qualify(&p).is_none());
    }
}
