use crate::{BUILD, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub(super) fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(super) async fn download(
    client: &reqwest::Client,
    source: &Value,
    filename: &str,
) -> Result<PathBuf> {
    let target = Path::new(BUILD).join("downloads").join(filename);
    let expected = source["sha256"].as_str().ok_or("missing download hash")?;
    if target.exists() && sha256(&target)? == expected {
        return Ok(target);
    }
    let temporary = target.with_extension("partial");
    let mut response = client
        .get(source["url"].as_str().ok_or("missing source URL")?)
        .send()
        .await?
        .error_for_status()?;
    let mut file = File::create(&temporary)?;
    while let Some(bytes) = response.chunk().await? {
        file.write_all(&bytes)?;
    }
    drop(file);
    if sha256(&temporary)? != expected {
        return Err(format!("download hash mismatch: {filename}").into());
    }
    fs::rename(&temporary, &target)?;
    Ok(target)
}
