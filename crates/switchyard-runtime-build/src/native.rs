use crate::download::sha256;
use crate::filesystem::{copy_file, copy_tree, files};
use crate::{BUILD, Result};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) fn assemble_toolchain(root: &Path) -> Result<()> {
    let toolchain = root.join("toolchain");
    copy_tree(
        Path::new("/usr/include"),
        &toolchain.join("sysroot/usr/include"),
    )?;
    copy_tree(
        Path::new("/usr/lib/gcc/x86_64-linux-gnu/13"),
        &toolchain.join("lib/gcc/x86_64-linux-gnu/13"),
    )?;
    // Ubuntu separates cc1, cc1plus, collect2, and LTO helpers into libexec.
    // GCC_EXEC_PREFIX points at the combined relocated directory on workers.
    copy_tree(
        Path::new("/usr/libexec/gcc/x86_64-linux-gnu/13"),
        &toolchain.join("lib/gcc/x86_64-linux-gnu/13"),
    )?;
    for (name, source) in [
        ("gcc.real", "/usr/bin/x86_64-linux-gnu-gcc-13"),
        ("g++.real", "/usr/bin/x86_64-linux-gnu-g++-13"),
        ("as", "/usr/bin/x86_64-linux-gnu-as"),
        ("ld", "/usr/bin/x86_64-linux-gnu-ld.bfd"),
    ] {
        copy_file(Path::new(source), &toolchain.join("bin").join(name))?;
    }
    for name in ["gcc", "g++"] {
        let target = toolchain.join("bin").join(name);
        copy_file(&Path::new(BUILD).join("compiler-wrapper.sh"), &target)?;
        fs::set_permissions(target, fs::Permissions::from_mode(0o755))?;
    }
    let libraries = Path::new("/usr/lib/x86_64-linux-gnu");
    for entry in fs::read_dir(libraries)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".a")
            || name.ends_with(".o")
            || ["libc.so", "libm.so", "libmvec.so"].contains(&name.as_str())
        {
            copy_file(
                &entry.path(),
                &toolchain
                    .join("sysroot/usr/lib/x86_64-linux-gnu")
                    .join(name),
            )?;
        }
    }
    for name in [
        "libc.so.6",
        "libm.so.6",
        "libmvec.so.1",
        "ld-linux-x86-64.so.2",
    ] {
        copy_file(
            &libraries.join(name),
            &toolchain.join("sysroot/lib/x86_64-linux-gnu").join(name),
        )?;
    }
    // Ubuntu's libc linker script names the loader through /lib64.
    copy_file(
        &libraries.join("ld-linux-x86-64.so.2"),
        &toolchain.join("sysroot/lib64/ld-linux-x86-64.so.2"),
    )?;
    Ok(())
}

fn elf_dependencies(paths: &[PathBuf], root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut dependencies = Vec::new();
    for path in paths {
        if !path.is_file() || path.is_symlink() {
            continue;
        }
        let mut magic = [0; 4];
        if File::open(path)?.read(&mut magic)? != 4 || magic != *b"\x7fELF" {
            continue;
        }
        let output = Command::new("ldd")
            .arg(path)
            .env("LD_LIBRARY_PATH", root.join("lib"))
            .output()?;
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let columns: Vec<_> = line.split_whitespace().collect();
            if columns.len() >= 3 && columns[1] == "=>" && columns[2].starts_with('/') {
                dependencies.push((columns[0].to_owned(), PathBuf::from(columns[2])));
            }
        }
    }
    Ok(dependencies)
}

pub(super) fn assemble_native_libraries(root: &Path) -> Result<BTreeMap<String, String>> {
    let excluded = [
        "libc.so.6",
        "libm.so.6",
        "libpthread.so.0",
        "libdl.so.2",
        "librt.so.1",
        "libresolv.so.2",
        "libutil.so.1",
        "ld-linux-x86-64.so.2",
    ];
    let paths = files(root)?;
    let groups: Vec<_> = paths.chunks(paths.len().div_ceil(12).max(1)).collect();
    let results: Result<Vec<_>> = std::thread::scope(|scope| {
        let handles: Vec<_> = groups
            .into_iter()
            .map(|group| scope.spawn(move || elf_dependencies(group, root)))
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| "ELF inspection thread panicked")?
            })
            .collect()
    });
    let mut dependencies: BTreeMap<String, String> = BTreeMap::new();
    for (name, source) in results?.into_iter().flatten() {
        if excluded.contains(&name.as_str()) || source.starts_with(root) {
            continue;
        }
        if let Some(previous) = dependencies.get(&name)
            && sha256(Path::new(previous))? != sha256(&source)?
        {
            return Err(format!("conflicting native library: {name}").into());
        }
        dependencies.insert(name, source.to_string_lossy().into_owned());
    }
    for (name, source) in &dependencies {
        copy_file(Path::new(source), &root.join("lib").join(name))?;
    }
    for name in ["libstdc++.so.6", "libgcc_s.so.1"] {
        let source = root.join("lib").join(name);
        if source.exists() {
            copy_file(
                &source,
                &root
                    .join("toolchain/sysroot/usr/lib/x86_64-linux-gnu")
                    .join(name),
            )?;
        }
    }
    Ok(dependencies)
}
