//! Assemble NVIDIA's pinned compiler components without installing a system CUDA package.

use crate::command::run;
use crate::download::download;
use crate::filesystem::{copy_file, copy_tree};
use crate::{BUILD, Result};
use serde_json::Value;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;

pub(super) async fn assemble(root: &Path, lock: &Value, client: &reqwest::Client) -> Result<()> {
    let toolkit = root.join("cuda");
    fs::create_dir_all(&toolkit)?;
    let components = lock["cuda_toolkit"]["components"]
        .as_array()
        .ok_or("missing CUDA compiler components")?;
    for component in components {
        let name = component["name"]
            .as_str()
            .ok_or("missing CUDA component name")?;
        if !matches!(
            name,
            "cuda_nvcc" | "cuda_cudart" | "cuda_cccl" | "cuda_crt" | "libnvvm" | "libcurand"
        ) {
            return Err(format!("unsupported CUDA compiler component: {name}").into());
        }
        let archive = download(client, component, &format!("{name}.tar.xz")).await?;
        let stage = Path::new(BUILD).join("cuda-components").join(name);
        fs::create_dir_all(&stage)?;
        run(
            "tar",
            &[
                "-xJf",
                archive.to_str().ok_or("invalid CUDA archive path")?,
                "--strip-components=1",
                "-C",
                stage.to_str().ok_or("invalid CUDA stage path")?,
            ],
        )?;
        copy_tree(&stage, &toolkit)?;
        copy_file(
            &stage.join("LICENSE"),
            &toolkit.join("licenses").join(format!("{name}.txt")),
        )?;
    }
    // NVIDIA redistributions use lib; CUDA extension builders commonly expect lib64.
    symlink("lib", toolkit.join("lib64"))?;
    for path in [
        "bin/nvcc",
        "include/cuda_runtime.h",
        "include/crt/host_config.h",
        "include/curand.h",
        "nvvm/bin/cicc",
        "nvvm/libdevice/libdevice.10.bc",
        "lib/libcudart.so",
    ] {
        if !toolkit.join(path).is_file() {
            return Err(format!("CUDA compiler component is incomplete: {path}").into());
        }
    }
    Ok(())
}
