//! Precompile immutable Python trees without tying caches to extraction timestamps.

use crate::Result;
use crate::command::run;
use std::path::Path;

pub(super) const INVALIDATION_MODE: &str = "unchecked-hash";

pub(super) fn compile(root: &Path) -> Result<()> {
    let libraries = [
        root.join("python/lib/python3.12"),
        root.join("venv/lib/python3.12/site-packages"),
    ];
    for library in &libraries {
        run(
            root.join("venv/bin/python"),
            &[
                "-m",
                "compileall",
                "-q",
                "-f",
                "--invalidation-mode",
                INVALIDATION_MODE,
                library.to_str().ok_or("invalid Python library path")?,
            ],
        )?;
    }
    Ok(())
}
