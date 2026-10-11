//! Hydration timing observes the generic progress callback without altering hydration.
use serde::{Deserialize, Serialize};
use std::{sync::Mutex, time::Instant};
use switchyard_artifacts::{HydrationMode, HydrationOptions, HydrationReport, Progress};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydrationReceipt {
    pub id: String,
    pub prefix: String,
    pub mode: HydrationMode,
    pub configured_retries: usize,
    pub declared_bytes: u64,
    pub downloaded_bytes: u64,
    pub verified_bytes: u64,
    pub network_verified_bytes: u64,
    pub cache_hits: usize,
    pub already_present: bool,
    pub controller_elapsed_s: f64,
    /// Controller resolution plus hydration through successful directory publication.
    pub total_elapsed_s: f64,
    pub hydration_elapsed_s: f64,
    /// First progress callback observing positive wire bytes, not the exact first byte.
    pub first_network_progress_elapsed_s: Option<f64>,
    /// Callback observing the declared total; only trusted streaming with retries disabled.
    /// The observation may follow the actual last byte because callbacks are throttled.
    pub all_network_bytes_observed_elapsed_s: Option<f64>,
    /// Publication minus the final transfer observation: a lower bound on the actual tail.
    pub post_transfer_observation_s: Option<f64>,
    pub timing_qualification: String,
}

#[derive(Default)]
struct Observations {
    first: Option<f64>,
    complete: Option<f64>,
}
pub(crate) struct HydrationTiming {
    started: Instant,
    observations: Mutex<Observations>,
    id: String,
    prefix: String,
    mode: HydrationMode,
    retries: usize,
    declared_bytes: u64,
    controller_elapsed_s: f64,
}
impl HydrationTiming {
    pub(crate) fn new(
        id: String,
        prefix: String,
        declared_bytes: u64,
        controller_elapsed_s: f64,
        options: &HydrationOptions,
    ) -> Self {
        Self {
            started: Instant::now(),
            observations: Mutex::default(),
            id,
            prefix,
            mode: options.mode,
            retries: options.retries,
            declared_bytes,
            controller_elapsed_s,
        }
    }
    pub(crate) fn observe(&self, progress: Progress) {
        let mut observations = self
            .observations
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let elapsed = self.started.elapsed().as_secs_f64();
        if progress.downloaded_bytes > 0 {
            observations.first.get_or_insert(elapsed);
        }
        if progress.downloaded_bytes == progress.total_bytes && progress.total_bytes > 0 {
            observations.complete.get_or_insert(elapsed);
        }
    }
    pub(crate) fn receipt(&self, report: HydrationReport) -> HydrationReceipt {
        let hydration_elapsed_s = self.started.elapsed().as_secs_f64();
        let observations = self
            .observations
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let valid = self.mode == HydrationMode::TrustedStreaming
            && self.retries == 0
            && !report.already_present
            && report.downloaded_bytes == self.declared_bytes;
        let complete = valid.then_some(observations.complete).flatten();
        let timing_qualification = if report.already_present {
            "existing_receipt_no_transfer"
        } else if self.mode != HydrationMode::TrustedStreaming {
            "verified_or_cached_transfer_ambiguous"
        } else if self.retries != 0 {
            "retries_enabled_transfer_completion_ambiguous"
        } else {
            "throttled_callback_last_byte_upper_bound_tail_lower_bound"
        };
        HydrationReceipt {
            id: self.id.clone(),
            prefix: self.prefix.clone(),
            mode: self.mode,
            configured_retries: self.retries,
            declared_bytes: self.declared_bytes,
            downloaded_bytes: report.downloaded_bytes,
            verified_bytes: report.verified_bytes,
            network_verified_bytes: report.network_verified_bytes,
            cache_hits: report.cache_hits,
            already_present: report.already_present,
            controller_elapsed_s: self.controller_elapsed_s,
            total_elapsed_s: self.controller_elapsed_s + hydration_elapsed_s,
            hydration_elapsed_s,
            first_network_progress_elapsed_s: observations.first,
            all_network_bytes_observed_elapsed_s: complete,
            post_transfer_observation_s: complete.map(|time| (hydration_elapsed_s - time).max(0.0)),
            timing_qualification: timing_qualification.into(),
        }
    }
}
