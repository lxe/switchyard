use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/paged_decode.cu");
    for name in ["CUDA_ROOT", "CUDA_COMPUTE_CAP", "NVCC"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let nvcc = env::var_os("NVCC").map(PathBuf::from).unwrap_or_else(|| {
        env::var_os("CUDA_ROOT")
            .map(|root| PathBuf::from(root).join("bin/nvcc"))
            .unwrap_or_else(|| PathBuf::from("nvcc"))
    });
    // PTX for Ampere is valid on newer devices. Platform bake jobs override this
    // explicitly, using the same architecture selection as Candle's kernels.
    let arch = env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "80".into());
    assert!(
        {
            let base = arch.strip_suffix(['a', 'f']).unwrap_or(&arch);
            (2..=3).contains(&base.len()) && base.chars().all(|c| c.is_ascii_digit())
        },
        "invalid CUDA_COMPUTE_CAP"
    );
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("paged_decode.ptx");
    let status = Command::new(nvcc)
        .args([
            "--ptx",
            "-std=c++17",
            "-O3",
            "--prec-div=true",
            "--prec-sqrt=true",
        ])
        .arg(format!("--gpu-architecture=compute_{arch}"))
        .args(["src/paged_decode.cu", "-o"])
        .arg(output)
        .status()
        .expect("nvcc is required to build the CUDA feature");
    assert!(status.success(), "paged decode CUDA compilation failed");
}
