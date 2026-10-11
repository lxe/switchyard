"""Inventory the native Ubuntu runtime before publication, outside serving."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--builder", required=True)
    parser.add_argument("--compute-cap", required=True)
    args = parser.parse_args()
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
        "cuda_build": "13.0.0",
        "gpu_compute_capability": args.compute_cap[:-1] + "." + args.compute_cap[-1],
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
