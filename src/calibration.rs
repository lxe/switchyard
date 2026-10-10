//! Small, portable calibration constants from Bullet Train's October 2026 studies.
//!
//! Startup observations were selected sequential priming runs on October 10.
//! Their build fraction is not a production build probability. Text throughput
//! observations and startup observations are separate experiments: attaching a
//! startup sample does not make an assumed throughput curve empirical.

use crate::config::{CapacityPoint, Profile, Workload};

const QWEN_08_RESTORE: &[f64] = &[17.7, 14.6, 11.6, 11.6, 15.3, 41.3];
const QWEN_08_BUILD: &[f64] = &[197.7, 161.1, 132.6];
const QWEN_2_RESTORE: &[f64] = &[
    13.5, 11.7, 15.5, 13.0, 9.0, 12.9, 11.6, 13.8, 12.2, 11.4, 15.0, 21.5,
];
const QWEN_2_BUILD: &[f64] = &[246.4, 150.4, 230.1, 166.9, 234.7];
const QWEN_4_RESTORE: &[f64] = &[
    24.4, 18.1, 19.7, 12.2, 12.9, 13.2, 13.0, 12.5, 41.5, 10.7, 15.6, 38.6, 13.8, 12.8, 17.7, 13.2,
];
const QWEN_4_BUILD: &[f64] = &[191.7, 196.3, 166.6, 296.2];
const GEMMA_E4_RESTORE: &[f64] = &[33.9, 20.7, 19.7, 18.0];
const GEMMA_E4_BUILD: &[f64] = &[213.3];
const GEMMA_31_RESTORE: &[f64] = &[73.2, 65.2, 68.6, 25.1, 77.2];
const GEMMA_31_BUILD: &[f64] = &[383.6, 401.2];

fn canonical_model(model: &str) -> &str {
    match model {
        "gemma4-e4b" => "gemma-4-e4b",
        "gemma4-31b" => "gemma-4-31b",
        other => other,
    }
}

fn annotate(value: &mut String, note: &str) {
    if !value.contains(note) {
        if !value.is_empty() {
            value.push_str("; ");
        }
        value.push_str(note);
    }
}

fn attach_startup(profile: &mut Profile, restore: &[f64], build: &[f64]) {
    profile.startup.restore_samples_s = restore.to_vec();
    profile.startup.build_samples_s = build.to_vec();
    annotate(
        &mut profile.evidence,
        "measured startup only: exact selected model/GPU end-to-end restore and build samples; throughput and startup-component breakdown remain separate assumptions",
    );
    annotate(
        &mut profile.source,
        "Bullet Train sequential model-app priming, 2026-10-10: total start time includes provider queue and snapshot work; small selected samples, no inferred production build probability",
    );
}

/// Enrich matching generated defaults without transferring measurements to a
/// different GPU. Repeated application does not append duplicate profiles.
pub fn apply_calibration(profiles: &mut Vec<Profile>) {
    for profile in profiles.iter_mut() {
        let model = canonical_model(&profile.model);
        let samples = if profile.gpu == "H100" {
            match model {
                "qwen3.5-0.8b" if profile.id.starts_with("fast-") => {
                    Some((QWEN_08_RESTORE, QWEN_08_BUILD))
                }
                "qwen3.5-2b" if profile.id.starts_with("fast-") => {
                    Some((QWEN_2_RESTORE, QWEN_2_BUILD))
                }
                "qwen3.5-4b" if profile.id.starts_with("fast-") => {
                    Some((QWEN_4_RESTORE, QWEN_4_BUILD))
                }
                "gemma-4-e4b"
                    if profile.id.starts_with("elastic-") || profile.id.starts_with("fast-") =>
                {
                    Some((GEMMA_E4_RESTORE, GEMMA_E4_BUILD))
                }
                _ => None,
            }
        } else if profile.gpu == "RTX-PRO-6000"
            && model == "gemma-4-31b"
            && profile.id.starts_with("elastic-")
        {
            Some((GEMMA_31_RESTORE, GEMMA_31_BUILD))
        } else {
            None
        };
        if let Some((restore, build)) = samples {
            attach_startup(profile, restore, build);
        }

        // Both sides use the same 64-client, 60-second, 200-output-token
        // workload. Subtract 64 unfinished requests from each old count before
        // taking the ratio. This is conservative boundary adjustment, not a
        // repaired raw trace, and belongs only to the measured GPU/workload.
        if profile.workload == Workload::Text && profile.id.starts_with("elastic-") {
            let ratio = match (canonical_model(&profile.model), profile.gpu.as_str()) {
                ("qwen3.5-0.8b", "A10") => Some((640.0 - 64.0) / (921.0 - 64.0)),
                ("qwen3.5-2b", "A10") => Some((448.0 - 64.0) / (608.0 - 64.0)),
                ("qwen3.5-4b", "L40S") => Some((401.0 - 64.0) / (480.0 - 64.0)),
                _ => None,
            };
            if let Some(ratio) = ratio {
                profile.multi_adapter_factor = ratio;
                annotate(
                    &mut profile.evidence,
                    "boundary_adjusted adapter factor: four interleaved adapters/base ratio from matching 64-client text observations; other adapter counts are unmeasured",
                );
                annotate(
                    &mut profile.source,
                    "Bullet Train bench-2026-10-08.jsonl: original counts predate window-cutoff fix; ratio subtracts 64 requests from both base and adapters4 counts (60 s, 200 output tokens, about 500 input tokens)",
                );
            }
        }
    }

    // These are end-to-end aggregate observations at 32 clients on one GPU.
    // Median stream decode rates are not multiplied by client count. A single
    // calibration point deliberately leaves the lower-concurrency curve as the
    // engine's assumed linear interpolation, rather than fabricating samples.
    for (model, aggregate_output_tps) in [("qwen3.5-0.8b", 3_373.0), ("qwen3.5-2b", 2_449.0)] {
        let id = format!("documented-rtx4090-{model}");
        if profiles.iter().any(|profile| profile.id == id) {
            continue;
        }
        let template = profiles
            .iter()
            .find(|profile| {
                profile.model == model
                    && profile.workload == Workload::Text
                    && profile.id.starts_with("baseline-")
            })
            .or_else(|| {
                profiles
                    .iter()
                    .find(|profile| profile.model == model && profile.workload == Workload::Text)
            });
        let Some(template) = template else {
            continue;
        };
        let mut profile = template.clone();
        profile.id = id;
        profile.gpu = "RTX4090".into();
        profile.usd_per_hour = 0.89;
        profile.max_in_flight = 32;
        profile.startup.restore_samples_s.clear();
        profile.startup.build_samples_s.clear();
        // For the measured 500-in/200-out workload, the macro capacity curve
        // also pays the explicitly assumed input-work coefficient. This keeps
        // the 32-request observation in output-token units after prefill work.
        let macro_work_factor = 1.0 + 500.0 * profile.input_work_per_token / 200.0;
        profile.curve = vec![CapacityPoint {
            in_flight: 32,
            work_units_per_s: aggregate_output_tps * macro_work_factor,
        }];
        profile.evidence = format!(
            "documented: one RTX4090, 32-client text aggregate {aggregate_output_tps:.0} output tokens/s; lower-concurrency curve is estimated linear interpolation; input-work conversion, memory, adapters, host, network and startup are assumed; no matching startup CDF"
        );
        profile.source = "Bullet Train serving-concurrency.md, 2026-10-08: streamed synthetic text, about 500 input and exactly 200 output tokens; raw sample count and duration unavailable. Advertised RunPod Secure RTX4090 rate $0.89/hour checked 2026-10-10, availability dependent: https://www.runpod.io/gpu-models/rtx-4090".into();
        profiles.push(profile);
    }
}
