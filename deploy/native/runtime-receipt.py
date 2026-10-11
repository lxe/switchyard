"""Inventory the native Ubuntu runtime before publication, outside serving."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--builder", required=True)
    parser.add_argument("--compute-cap", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{2,3}", args.compute_cap):
        raise ValueError(
            "numeric CUDA compiler target required; suffixes are unsupported"
        )
    target = "compute_" + args.compute_cap
    if target not in (args.evidence / "nvcc-gpu-arch.txt").read_text().splitlines():
        raise ValueError("compiler evidence does not support the selected target")
    toolkit = (args.evidence / "cuda-environment-version.txt").read_text().strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", toolkit):
        raise ValueError("builder toolkit release must have major.minor.patch version")
    packages = dict(
        line.split("\t", 1)
        for line in (args.evidence / "cuda-packages.txt").read_text().splitlines()
        if "\t" in line
    )
    nvcc = re.search(
        r"release (\d+\.\d+), V([0-9.]+)", (args.evidence / "nvcc.txt").read_text()
    )
    if not nvcc or toolkit.split(".")[:2] != nvcc[1].split("."):
        raise ValueError("CUDA toolkit and compiler evidence disagree")
    compiler_package = "cuda-nvcc-" + nvcc[1].replace(".", "-")
    if not packages.get(compiler_package, "").startswith(nvcc[2] + "-"):
        raise ValueError("installed CUDA compiler package differs from nvcc evidence")
    operating_system = dict(
        line.split("=", 1)
        for line in (args.evidence / "os-release.txt").read_text().splitlines()
        if "=" in line
    )
    os_id = operating_system["ID"].strip('"')
    os_version = operating_system["VERSION_ID"].strip('"')
    architecture = (args.evidence / "architecture.txt").read_text().strip()
    if (os_id, os_version, architecture) != ("ubuntu", "24.04", "x86_64"):
        raise ValueError("this runtime helper requires Ubuntu 24.04 x86_64")
    files = []
    for path in sorted(args.tree.rglob("*")):
        if path.is_file():
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            files.append(
                {
                    "path": str(path.relative_to(args.tree)),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    receipt = {
        "format": "switchyard-native-runtime-build-v1",
        "source_commit": args.source_commit,
        "builder_image": args.builder,
        "os": "ubuntu24.04",
        "architecture": "x86_64",
        "cuda_build": toolkit,
        "cuda_build_version_origin": "builder CUDA_VERSION with installed compiler validation",
        "cuda_compiler_version": nvcc[2],
        "cuda_compiler_target": target,
        "cuda_package_versions": packages,
        "gpu_compute_capability": str(int(args.compute_cap[:-1]))
        + "."
        + args.compute_cap[-1],
        "files": files,
        "runtime_bytes": sum(item["bytes"] for item in files),
        "numerical_accuracy": "see source commit accuracy receipt; not measured by build",
        "serving_python": False,
        "torch": False,
        "vllm": False,
        "criu": False,
        "gpu_process_snapshot": False,
    }
    args.build.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runtime_bytes": receipt["runtime_bytes"],
                "files": len(files),
                "source_commit": args.source_commit,
            }
        )
    )


if __name__ == "__main__":
    main()
