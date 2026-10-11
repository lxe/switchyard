//! Build-time runtime derivation and component layout; never worker startup logic.
pub mod native_closure;
pub mod partition;
mod patches;

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub use patches::{Receipt as PatchReceipt, apply as apply_source_patches};

#[cfg(test)]
mod fixtures;

pub fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn package_version<'a>(packages: &'a Value, name: &str) -> Result<&'a str> {
    packages
        .as_array()
        .and_then(|p| p.iter().find(|p| p["name"] == name))
        .and_then(|p| p["version"].as_str())
        .ok_or_else(|| format!("installed package has no recorded version: {name}").into())
}
