//! Docker is an isolated preparation tool; its filesystem export never becomes a worker image.
use crate::{Error, Result};
use std::{
    fs,
    path::{Component, Path},
};

struct ExportScratch<'a> {
    root: &'a Path,
}
impl Drop for ExportScratch<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.root.join(".runtime-export.tar"));
        let _ = fs::remove_dir_all(self.root.join(".runtime-export-staging"));
    }
}

pub(crate) async fn build(context: &Path, dockerfile: &Path, out: &Path) -> Result<()> {
    if out.exists() && fs::read_dir(out)?.next().is_some() {
        return Err(Error::Config(
            "runtime export directory must be empty".into(),
        ));
    }
    fs::create_dir_all(out)?;
    let destination = fs::canonicalize(out)?;
    let dockerfile = fs::canonicalize(dockerfile)?;
    let context = fs::canonicalize(context)?;
    if destination.to_string_lossy().contains(',') {
        return Err(Error::Config(
            "Docker export path cannot contain commas".into(),
        ));
    }
    let _scratch = ExportScratch { root: &destination };
    let archive = destination.join(".runtime-export.tar");
    // A tar export avoids BuildKit's per-file directory-transfer protocol for large runtimes.
    let status = tokio::process::Command::new("docker")
        .args([
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--target",
            "export",
            "--output",
        ])
        .arg(format!("type=tar,dest={}", archive.display()))
        .arg("--file")
        .arg(dockerfile)
        .arg(context)
        .kill_on_drop(true)
        .status()
        .await?;
    if !status.success() {
        return Err(Error::Config(
            "runtime Docker build failed; partial export is not usable".into(),
        ));
    }
    // This is the trusted local builder's export, not an artifact transfer. Published
    // artifacts use the stronger complete manifest verification in switchyard-artifacts.
    unpack_export(&archive, &destination.join(".runtime-export-staging"))?;
    let runtime = destination.join(".runtime-export-staging/runtime");
    if !runtime.join("artifact-metadata.json").is_file() {
        return Err(Error::Config(
            "runtime export is missing build metadata".into(),
        ));
    }
    fs::rename(runtime, destination.join("runtime"))?;
    Ok(())
}

fn unpack_export(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir(destination)?;
    let mut archive = tar::Archive::new(fs::File::open(source)?);
    archive.set_preserve_permissions(true);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let mut components = path
            .components()
            .filter(|c| !matches!(c, Component::CurDir));
        if components.next() != Some(Component::Normal("runtime".as_ref()))
            || components.any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(Error::Config(
                "Docker export contains a path outside runtime".into(),
            ));
        }
        let kind = entry.header().entry_type();
        if !(kind.is_file() || kind.is_dir() || kind.is_symlink()) {
            return Err(Error::Config(
                "Docker export contains an unsupported entry".into(),
            ));
        }
        if kind.is_symlink() && entry.link_name()?.is_none_or(|target| target.is_absolute()) {
            return Err(Error::Config(
                "Docker export contains an invalid runtime symlink".into(),
            ));
        }
        if !entry.unpack_in(destination)? {
            return Err(Error::Config(
                "Docker export path could not be extracted safely".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn export_requires_runtime_root_and_rejects_external_links() {
        let root = std::env::temp_dir().join(format!("switchyard-export-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        for (index, path, target, accepted) in [
            (0, "runtime/bin/python", None, true),
            (1, "unexpected/file", None, false),
            (2, "runtime/bin/python", Some("/usr/bin/python"), false),
        ] {
            let archive_path = root.join(format!("{index}.tar"));
            let mut archive = tar::Builder::new(fs::File::create(&archive_path).unwrap());
            let mut header = tar::Header::new_gnu();
            header.set_mode(0o755);
            if let Some(target) = target {
                header.set_entry_type(tar::EntryType::Symlink);
                header.set_size(0);
                archive.append_link(&mut header, path, target).unwrap();
            } else {
                header.set_size(2);
                header.set_cksum();
                archive
                    .append_data(&mut header, path, Cursor::new(b"ok"))
                    .unwrap();
            }
            archive.finish().unwrap();
            assert_eq!(
                unpack_export(&archive_path, &root.join(index.to_string())).is_ok(),
                accepted
            );
        }
        fs::remove_dir_all(root).unwrap();
    }
}
