use crate::{Error, Result};
use std::{
    fs,
    path::{Component, Path, PathBuf},
};

pub(crate) fn relative(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 4096
        || value.contains(['\0', '\\', ':'])
        || value.split('/').count() > 128
        || value
            .split('/')
            .any(|c| c.is_empty() || c == "." || c == "..")
        || Path::new(value)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(Error::Invalid("unsafe relative path".into()));
    }
    Ok(())
}

pub(crate) fn link_target(path: &str, target: &str) -> Result<String> {
    if target.is_empty()
        || target.len() > 4096
        || target.contains(['\0', '\\', ':'])
        || target.starts_with('/')
    {
        return Err(Error::Invalid("unsafe symlink target".into()));
    }
    let mut components = path.split('/').collect::<Vec<_>>();
    components.pop();
    for part in target.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if components.pop().is_none() {
                    return Err(Error::Invalid("symlink escapes artifact root".into()));
                }
            }
            value => components.push(value),
        }
    }
    Ok(components.join("/"))
}

/// Inspect each existing ancestor without resolving or trusting a symlink.
pub(crate) fn ancestors(path: &Path) -> Result<()> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut current = PathBuf::new();
    for part in absolute.components() {
        match part {
            Component::ParentDir => {
                return Err(Error::Invalid("parent components in destination".into()));
            }
            Component::CurDir => continue,
            _ => current.push(part.as_os_str()),
        }
        match fs::symlink_metadata(&current) {
            Ok(meta) if !meta.is_dir() => {
                return Err(Error::Invalid(
                    "destination ancestor is not a real directory".into(),
                ));
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

pub(crate) fn directory(path: &Path) -> Result<()> {
    ancestors(path)?;
    fs::create_dir_all(path)?;
    ancestors(path)
}

pub(crate) fn nonce() -> Result<String> {
    let mut bytes = [0_u8; 12];
    getrandom::fill(&mut bytes).map_err(|_| Error::Worker)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
