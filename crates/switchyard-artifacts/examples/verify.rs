//! Run complete verification with a chosen reader count, without touching the tree.
use std::{path::PathBuf, time::Instant};
use switchyard_artifacts::{Error, Manifest, VerificationOptions, verify_with_options};

fn main() -> switchyard_artifacts::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let usage = || Error::Invalid("usage: verify MANIFEST DIRECTORY [CONCURRENCY]".into());
    let manifest_path = PathBuf::from(args.next().ok_or_else(usage)?);
    let directory = PathBuf::from(args.next().ok_or_else(usage)?);
    let options = match args.next() {
        Some(value) => VerificationOptions {
            concurrency: value
                .to_str()
                .ok_or_else(usage)?
                .parse()
                .map_err(|_| usage())?,
        },
        None => VerificationOptions::default(),
    };
    if args.next().is_some() {
        return Err(usage());
    }
    let started = Instant::now();
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    let parsed_s = started.elapsed().as_secs_f64();
    verify_with_options(&directory, &manifest, options)?;
    println!(
        "{}",
        serde_json::json!({
            "id": manifest.id, "entries": manifest.entries.len(), "concurrency": options.concurrency,
            "file_bytes": manifest.entries.iter().map(|entry| match entry {
                switchyard_artifacts::Entry::File { size, .. } => *size,
                _ => 0,
            }).sum::<u64>(),
            "manifest_parse_s": parsed_s, "verify_s": started.elapsed().as_secs_f64() - parsed_s,
            "total_s": started.elapsed().as_secs_f64(),
        })
    );
    Ok(())
}
