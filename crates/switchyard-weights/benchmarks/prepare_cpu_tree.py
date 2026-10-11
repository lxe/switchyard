"""Bake a native API/engine CPU process tree for one immutable serving profile.

Uses the selected runtime and warmed cache on the same paths used by the agent.
No tensor transfer or GPU allocation occurs during this CPU checkpoint bake.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("runtime_set", type=Path)
    parser.add_argument("bake", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image", default="switchyard-local-criu")
    parser.add_argument("--loader-config", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[3]
    prepared = json.loads((args.prepared / "prepared.json").read_text())
    runtime = json.loads((args.prepared / "runtime.json").read_text())["bootstrap"][
        "runtime"
    ]
    loader_config = json.loads(args.loader_config.read_text())
    loader = subprocess.check_output(
        [
            str(project / "target/release/switchyard-weights"),
            "loader-identity",
            "--config",
            str(args.loader_config.resolve()),
        ],
        text=True,
    ).strip()
    identity = hashlib.sha256(
        f"switchyard-loader-path-v1:{prepared['model_content_id']}:{loader}".encode()
    ).hexdigest()
    sidecar = "/opt/switchyard/stream-models/.model-loader-" + identity[:24]
    runtime_root = "/opt/switchyard/runtime"
    cache_root = "/opt/switchyard/cache/" + runtime["artifact"]["id"]
    components_root = "/opt/switchyard/components"
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output / "cache"
    shutil.copytree(args.bake / "cache/compiled", cache)
    # Match the agent's bounded receiver: safetensors are transported to GPU
    # destinations, never exposed in the metadata directory. Their presence
    # changes native vLLM's checkpoint-based word-embedding tying decision.
    metadata = args.output / "metadata"
    manifest = json.loads(
        (Path(prepared["model_objects"]) / "manifest.json").read_text()
    )
    for entry in manifest["entries"]:
        target = metadata / entry["path"]
        if entry["kind"] == "directory":
            target.mkdir(parents=True, exist_ok=True)
        elif entry["kind"] == "file" and not entry["path"].endswith(".safetensors"):
            target.parent.mkdir(parents=True, exist_ok=True)
            source = args.bake / "model" / entry["path"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(
                    "CPU bake metadata differs from the selected weight manifest"
                )
            shutil.copyfile(source, target)
    plugin = args.output / "plugin"
    plugin.mkdir()
    for source in (project / "crates/switchyard-weights/python").glob("*.py"):
        shutil.copyfile(source, plugin / source.name)
    dist = plugin / "switchyard_extent_loader-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: switchyard-extent-loader\nVersion: 0.0.0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[vllm.general_plugins]\nswitchyard_extents = switchyard_extent_loader:register\n"
    )
    (plugin / "sitecustomize.py").write_text(
        "import os, sys\nif os.environ.get('SWITCHYARD_CPU_TREE_BAKE') == '1':\n"
        "    from switchyard_cpu_tree import guard_cuda, detach_bake_stdio\n"
        "    if '-c' not in sys.orig_argv or any('spawn_main' in arg for arg in sys.orig_argv):\n"
        "        guard_cuda()\n"
        "    else:\n        detach_bake_stdio()\n"
    )
    environment = {
        key: value.replace("{runtime_root}", runtime_root)
        .replace("{cache_root}", cache_root)
        .replace("{components_root}", components_root)
        for key, value in runtime["environment"].items()
        if not key.startswith("VLLM_CPU_SNAPSHOT")
    }
    environment.update(
        {
            "PYTHONPATH": sidecar
            + "/plugin:"
            + runtime_root
            + "/venv/lib/python3.12/site-packages:"
            + environment.get("PYTHONPATH", ""),
            "PYTHONHOME": runtime_root + "/python",
            "SWITCHYARD_CPU_TREE_BAKE": "1",
            "SWITCHYARD_RUNTIME_ROOT": runtime_root,
            "SWITCHYARD_RUNTIME_ARTIFACT_ID": runtime["artifact"]["id"],
            "SWITCHYARD_CACHE_ROOT": cache_root,
            "SWITCHYARD_COMPONENTS_ROOT": components_root,
            "SWITCHYARD_MODEL_CONTENT_ID": prepared["model_content_id"],
            "VLLM_PLUGINS": "switchyard_extents",
            "OMP_NUM_THREADS": "1",
        }
    )
    engine_args = [
        value.replace("{model_root}", sidecar + "/metadata")
        for value in json.loads((args.prepared / "engine-args.json").read_text())
    ]
    engine_args += [
        "--load-format",
        "switchyard_extents",
        "--model-loader-extra-config",
        json.dumps(
            {
                "socket_path": sidecar + "/weights.sock",
                "timeout_seconds": 600,
                "cuda_stage": loader_config.get("cuda_stage", False),
                "pinned_bytes": loader_config.get("pinned_bytes", 16 * 1024 * 1024),
                "chunk_streaming": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]
    config = {"args": engine_args, "environment": environment}
    (args.output / "serving.json").write_text(json.dumps(config, indent=2) + "\n")
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        "switchyard-cpu-tree-bake",
        "--privileged",
        "--pid",
        "host",
        "--network",
        "none",
        "--gpus",
        "device=0",
        "--memory",
        "32g",
        "--memory-swap",
        "32g",
        "--shm-size",
        "8g",
    ]
    mounts = {
        args.runtime_set / "boot/tree": runtime_root,
        args.runtime_set / "boot/artifact/manifest.json": "/runtime-manifest.json",
        metadata: sidecar + "/metadata",
        plugin: sidecar + "/plugin",
        project
        / "target/musl/x86_64-unknown-linux-musl/release/switchyard-cpu-snapshot": "/cpu-restore",
    }
    for source, target in mounts.items():
        command += ["-v", str(source.resolve()) + ":" + target + ":ro"]
    command += ["-v", str(cache.resolve()) + ":" + cache_root + "/compiled"]
    command += [
        "-v",
        str(args.output.resolve()) + ":/snapshots",
        "--entrypoint",
        "/cpu-restore",
        args.image,
        "prepare",
        "--runtime-root",
        runtime_root,
        "--runtime-manifest",
        "/runtime-manifest.json",
        "--serving-config",
        "/snapshots/serving.json",
        "--out",
        "/snapshots/snapshot",
    ]
    print(
        "[CPU bake] native API configuration and spawned worker; pause before CUDA",
        flush=True,
    )
    subprocess.run(command, check=True)
    subprocess.run(
        [
            "sudo",
            "-n",
            "chown",
            "-R",
            f"{os.getuid()}:{os.getgid()}",
            str(args.output.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
