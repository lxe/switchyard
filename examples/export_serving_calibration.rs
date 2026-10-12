//! Bind public cloud serving profiles to an exact deployment configuration offline.
//! cargo run --features live --example export_serving_calibration -- DATASET CONFIG OUTPUT
#[cfg(feature = "live")]
use std::{fs, path::Path};
#[cfg(feature = "live")]
use switchyard_core::calibration::ServingCalibration;
#[cfg(feature = "live")]
use switchyard_gauntlet::empirical::Dataset;
#[cfg(feature = "live")]
use switchyard_live::{calibration::binding, config::DeploymentConfig};
#[cfg(feature = "live")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("expected DATASET CONFIG OUTPUT".into());
    }
    let dataset: Dataset = serde_json::from_slice(&fs::read(&args[0])?)?;
    dataset.validate()?;
    let mut config: DeploymentConfig = serde_json::from_slice(&fs::read(&args[1])?)?;
    let execution = config.execution();
    for model in &mut config.models {
        model.prior.measured.clear();
        for profile in dataset.profiles.iter().filter(|p| p.model == model.id) {
            let mut measured = profile.clone();
            let mut samples: Vec<f64> = dataset
                .cases
                .iter()
                .filter_map(|case| case.observations.profiles.get(&profile.id))
                .flat_map(|o| o.startups.iter().map(|s| s.ready_s))
                .collect();
            // Cases can refer to the same observed allocation; repeated receipts
            // are not independent startup samples.
            samples.sort_by(f64::total_cmp);
            samples.dedup_by(|a, b| a.to_bits() == b.to_bits());
            measured.startup.restore_samples_s = samples;
            measured.source = format!("{}; deployment-bound calibration schema 1", measured.source);
            model.prior.measured.push(ServingCalibration {
                schema_version: 1,
                binding_sha256: binding(&execution, model, &profile.gpu)?,
                profile: measured,
            });
        }
    }
    config.validate()?;
    if let Some(parent) = Path::new(&args[2]).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args[2], serde_json::to_vec_pretty(&config)?)?;
    Ok(())
}

#[cfg(not(feature = "live"))]
fn main() {
    eprintln!("enable --features live");
}
