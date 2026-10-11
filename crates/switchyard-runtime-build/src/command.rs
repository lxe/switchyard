use crate::Result;
use std::ffi::OsStr;
use std::process::{Command, Output};

pub(super) fn run(program: impl AsRef<OsStr>, args: &[impl AsRef<OsStr>]) -> Result<()> {
    let status = Command::new(program.as_ref()).args(args).status()?;
    if !status.success() {
        return Err(format!("{} failed: {status}", program.as_ref().to_string_lossy()).into());
    }
    Ok(())
}

pub(super) fn capture(program: impl AsRef<OsStr>, args: &[impl AsRef<OsStr>]) -> Result<Output> {
    let output = Command::new(program.as_ref()).args(args).output()?;
    if !output.status.success() {
        return Err(format!(
            "{} failed: {}: {}",
            program.as_ref().to_string_lossy(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output)
}
