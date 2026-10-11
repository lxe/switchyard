use crate::Result;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

pub(super) fn copy_file(source: &Path, target: &Path) -> Result<()> {
    // Dereference source links; exported links must stay within the runtime root.
    fs::create_dir_all(target.parent().ok_or("missing destination parent")?)?;
    fs::copy(source, target)?;
    Ok(())
}

pub(super) fn copy_tree(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let destination = target.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &destination)?;
        } else {
            copy_file(&entry.path(), &destination)?;
        }
    }
    Ok(())
}

pub(super) fn files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        paths.push(path.clone());
        if entry.file_type()?.is_dir() {
            paths.extend(files(&path)?);
        }
    }
    paths.sort();
    Ok(paths)
}

pub(super) fn internalize_symlinks(root: &Path) -> Result<()> {
    for path in files(root)? {
        if !path.is_symlink() {
            continue;
        }
        let destination = fs::canonicalize(&path)?;
        let suffix = destination
            .strip_prefix(root)
            .map_err(|_| format!("external runtime symlink: {}", path.display()))?;
        let parent = path
            .parent()
            .ok_or("missing link parent")?
            .strip_prefix(root)?;
        let mut relative = PathBuf::new();
        for _ in parent.components() {
            relative.push("..");
        }
        relative.push(suffix);
        fs::remove_file(&path)?;
        symlink(relative, path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("switchyard-builder-{}-{stamp}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn absolute_internal_links_become_relative_and_remain_resolvable() {
        let scratch = Scratch::new();
        let root = scratch.0.join("runtime");
        fs::create_dir_all(root.join("venv/bin")).unwrap();
        fs::create_dir_all(root.join("python/bin")).unwrap();
        fs::write(root.join("python/bin/python"), b"interpreter").unwrap();
        let interpreter = fs::canonicalize(root.join("python/bin/python")).unwrap();
        symlink(&interpreter, root.join("venv/bin/python")).unwrap();
        internalize_symlinks(&root).unwrap();
        assert_eq!(
            fs::read_link(root.join("venv/bin/python")).unwrap(),
            Path::new("../../python/bin/python")
        );
        assert_eq!(
            fs::canonicalize(root.join("venv/bin/python")).unwrap(),
            interpreter
        );
    }

    #[test]
    fn external_links_are_rejected_without_rewriting_them() {
        let scratch = Scratch::new();
        let root = scratch.0.join("runtime");
        fs::create_dir_all(&root).unwrap();
        fs::write(scratch.0.join("outside"), b"outside").unwrap();
        symlink("../outside", root.join("escape")).unwrap();
        let error = internalize_symlinks(&root).unwrap_err();
        assert!(error.to_string().contains("external runtime symlink"));
        assert_eq!(
            fs::read_link(root.join("escape")).unwrap(),
            Path::new("../outside")
        );
    }

    #[test]
    fn copying_native_files_dereferences_links_and_preserves_executable_permissions() {
        let scratch = Scratch::new();
        fs::write(scratch.0.join("native"), b"native executable").unwrap();
        fs::set_permissions(scratch.0.join("native"), fs::Permissions::from_mode(0o755)).unwrap();
        symlink("native", scratch.0.join("alias")).unwrap();
        copy_file(
            &scratch.0.join("alias"),
            &scratch.0.join("export/bin/native"),
        )
        .unwrap();
        let exported = scratch.0.join("export/bin/native");
        assert!(!exported.is_symlink());
        assert_eq!(fs::read(&exported).unwrap(), b"native executable");
        assert_eq!(
            fs::metadata(&exported).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }
}
