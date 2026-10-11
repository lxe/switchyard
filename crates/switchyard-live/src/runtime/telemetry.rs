//! Incremental evidence survives an interrupted workload before final summaries.
use crate::Result;
use serde::Serialize;
use std::{fs::OpenOptions, io::Write, path::Path};

pub(super) fn append<T: Serialize>(out: &Path, name: &str, event: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec(event)?;
    bytes.push(b'\n');
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join(name))?
        .write_all(&bytes)?;
    Ok(())
}
