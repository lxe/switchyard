"""Publish one engine/library artifact and a common native launch configuration."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from control import Control, write
from prepare_artifacts import pack_publish, runtime_compatibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "runtime", "tool", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--cache-policy", choices=["strict", "opportunistic"], default="strict"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    build = json.loads((args.runtime / "build.json").read_text())
    if build["source_commit"] != config["engine_source_commit"]:
        raise ValueError("shared build differs from accepted source configuration")
    source = args.runtime / "tree"
    # Complete build inventory validation stays outside all paid serving runs.
    for entry in build["files"]:
        path = source / entry["path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError("exported runtime differs from build inventory")
    args.output.mkdir(parents=True, exist_ok=False)
    tree = args.output / "runtime-tree"
    (tree / "bin").mkdir(parents=True)
    shutil.copy2(source / "bin/switchyard-engine", tree / "bin/switchyard-engine")
    shutil.copytree(source / "lib", tree / "lib")
    shutil.copy2(
        Path(__file__).with_name("launch-engine-family.sh"),
        tree / "bin/launch-engine.sh",
    )
    (tree / "bin/launch-engine.sh").chmod(0o755)
    compatibility = runtime_compatibility(build)
    control = Control(config, args.output)
    runtime = pack_publish(
        args.tool.resolve(),
        control,
        tree,
        args.output / "runtime-artifact",
        {
            "kind": "native_shared_runtime",
            "compatibility": compatibility,
            "attributes": {
                "native_driver_cache_layout": "native-cuda-driver-family-v1",
                "native_driver_cache_policy": "strict default; explicit opportunistic cold JIT",
                "model_configuration": "inline --config-json",
                "serving_python": "false",
                "torch": "false",
                "vllm": "false",
                "criu": "false",
            },
        },
        "runtime",
    )
    launch = {
        "hydration": {"retries": 0},
        "artifact": {key: runtime[key] for key in ["id", "prefix"]},
        "compatibility": compatibility,
        "program": "bin/launch-engine.sh",
        "args": [],
        "environment": {
            "LD_LIBRARY_PATH": "{runtime_root}/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
            "NATIVE_ENGINE_CACHE_ROOT": "{cache_root}/gpu-kernels",
            "NATIVE_ENGINE_EXPECTED_GPU": config["provider"]["gpu_id"],
            "NATIVE_ENGINE_EXPECTED_COMPUTE_CAP": build["gpu_compute_capability"],
            "NATIVE_ENGINE_CACHE_POLICY": args.cache_policy,
            "CUDA_CACHE_DISABLE": "0",
            "CUDA_CACHE_MAXSIZE": "536870912",
        },
    }
    write(
        args.output / "shared-runtime.json",
        {
            "runtime": runtime,
            "compatibility": compatibility,
            "launch_configuration": launch,
            "serving_configuration_in_runtime": False,
        },
    )
    control.event(
        "shared_runtime_published",
        runtime_id=runtime["id"],
        engine_sha256=compatibility["engine_sha256"],
    )


if __name__ == "__main__":
    main()
