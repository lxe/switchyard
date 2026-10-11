"""Package and publish native runtime and model independently, before allocation."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from control import Control, write


def pack_publish(tool, control, source, output, metadata, namespace):
    write(output.with_suffix(".metadata.json"), metadata)
    subprocess.run(
        [
            str(tool),
            "artifacts",
            "pack",
            "--source",
            str(source),
            "--out",
            str(output),
            "--metadata",
            str(output.with_suffix(".metadata.json")),
        ],
        check=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    store = control.config["storage"]
    publish_packed(control, output, namespace)
    return {
        "id": manifest["id"],
        "prefix": store["prefix"] + "/" + namespace + "/" + manifest["id"],
        "directory": str(output.resolve()),
        "compressed_bytes": sum(c["size"] for c in manifest["chunks"]),
    }


def publish_packed(control, output, namespace):
    """Publish a packed artifact through bounded multipart-aware AWS transport."""
    manifest = json.loads((output / "manifest.json").read_text())
    prefix = (
        control.config["storage"]["prefix"] + "/" + namespace + "/" + manifest["id"]
    )
    for chunk in manifest["chunks"]:
        path = output / chunk["name"]
        if path.stat().st_size != chunk["size"]:
            raise ValueError("packed object size differs from manifest")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != chunk["sha256"]:
                raise ValueError("packed object hash differs from manifest")
        # AWS CLI uses bounded multipart uploads for large objects, including the
        # 4B native archive that exceeds R2's single-request 5 GiB ceiling.
        control.upload_immutable(path, prefix + "/" + chunk["name"], chunk["sha256"])
    manifest_path = output / "manifest.json"
    with manifest_path.open("rb") as source:
        manifest_sha = hashlib.file_digest(source, "sha256").hexdigest()
    control.upload_immutable(manifest_path, prefix + "/manifest.json", manifest_sha)
    write(
        output.with_suffix(".published.json"),
        {
            "artifact_id": manifest["id"],
            "prefix": prefix,
            "chunks": len(manifest["chunks"]),
            "manifest_published_last": True,
            "transport": "AWS CLI automatic multipart",
        },
    )


def legacy_manifest(manifest):
    chunks = [
        {
            "name": c["name"],
            "size": c["size"],
            "raw": c["unpacked_size"],
            "shard": i,
            "seq": 0,
            "sha256": c["sha256"],
        }
        for i, c in enumerate(manifest["chunks"])
    ]
    return {
        "format": "shards-v1",
        "compress": "zstd",
        "chunks": chunks,
        "total_size": sum(c["size"] for c in chunks),
        "raw_size": sum(c["raw"] for c in chunks),
        "sha256": hashlib.sha256(
            "".join(c["sha256"] for c in chunks).encode()
        ).hexdigest(),
        "files": {
            e["path"]: {"size": e["size"]}
            for e in manifest["entries"]
            if e["kind"] == "file"
        },
    }


def runtime_compatibility(build):
    engine_sha = next(
        f["sha256"] for f in build["files"] if f["path"] == "bin/switchyard-engine"
    )
    labels = {
        "architecture": build["architecture"],
        "os": build["os"],
        "cuda_build": build["cuda_build"],
        "gpu_compute_capability": build["gpu_compute_capability"],
        "engine_sha256": engine_sha,
        "engine_source_commit": build["source_commit"],
    }
    for key in ["cuda_compiler_target", "cuda_compiler_version"]:
        if key in build:
            labels[key] = build[key]
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-prepared", type=Path)
    parser.add_argument("--shared-runtime", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    control = Control(config, args.output)
    build = json.loads((args.runtime / "build.json").read_text())
    if build["source_commit"] != config["engine_source_commit"]:
        raise ValueError("runtime build differs from selected engine source commit")
    built_tree = args.runtime / "tree"
    shared = (
        json.loads(args.shared_runtime.read_text()) if args.shared_runtime else None
    )
    base_compatibility = runtime_compatibility(build)
    if shared and shared["compatibility"] != base_compatibility:
        raise ValueError("shared runtime differs from selected build")
    tree = args.output / "runtime-tree"
    if not shared:
        (tree / "bin").mkdir(parents=True)
        shutil.copy2(
            built_tree / "bin/switchyard-engine", tree / "bin/switchyard-engine"
        )
        shutil.copytree(built_tree / "lib", tree / "lib")
        shutil.copy2(
            Path(__file__).with_name("launch-engine.sh"), tree / "bin/launch-engine.sh"
        )
        (tree / "bin/launch-engine.sh").chmod(0o755)
        write(tree / "native-config.json", config["engine"])
    requests = json.loads(args.requests.read_text())
    # One maximum-size equal-length cohort plus varied prompt lengths prepares
    # actual generation without an automatic memory sizing forward.
    requests = requests[:8] + [requests[0]] * config["engine"]["max_batch_size"]
    preparation_tree = args.output / "preparation-tree"
    (preparation_tree / "bin").mkdir(parents=True)
    shutil.copy2(
        built_tree / "bin/switchyard-engine-bake",
        preparation_tree / "bin/switchyard-engine-bake",
    )
    write(preparation_tree / "bake-requests.json", requests)
    native_config = (preparation_tree if shared else tree) / "native-config.json"
    if shared:
        write(native_config, config["engine"])
    shutil.copy2(args.runtime / "build.json", preparation_tree / "native-build.json")
    config_sha = hashlib.sha256(native_config.read_bytes()).hexdigest()
    compatibility = {
        **base_compatibility,
        "native_config_sha256": config_sha,
    }
    runtime = (
        shared["runtime"]
        if shared
        else pack_publish(
            args.tool.resolve(),
            control,
            tree,
            args.output / "runtime-artifact",
            {
                "kind": "native_runtime",
                "compatibility": compatibility,
                "attributes": {
                    "accuracy": config["numerical_accuracy"],
                    "serving_python": "false",
                    "torch": "false",
                    "vllm": "false",
                    "criu": "false",
                },
            },
            "runtime",
        )
    )
    preparation = pack_publish(
        args.tool.resolve(),
        control,
        preparation_tree,
        args.output / "preparation-artifact",
        {
            "kind": "native_preparation",
            "compatibility": compatibility,
            "attributes": {"serving_dependency": "false", "offline_bake": "true"},
        },
        "preparation",
    )
    previous = (
        json.loads(args.reuse_prepared.read_text()) if args.reuse_prepared else None
    )
    if previous:
        model_manifest = json.loads(
            (Path(previous["model"]["directory"]) / "manifest.json").read_text()
        )
        if (
            model_manifest["metadata"]["compatibility"].get("model")
            != config["engine"]["model"]
        ):
            raise ValueError("reused native weights belong to another model")
    model = (
        previous["model"]
        if previous
        else pack_publish(
            args.tool.resolve(),
            control,
            args.model,
            args.output / "model-artifact",
            {
                "kind": "native_weights",
                "compatibility": {"model": config["engine"]["model"]},
                "attributes": {
                    "format": "switchyard-native-text-v1",
                    "vision": "excluded",
                },
            },
            "models",
        )
    )
    legacy = legacy_manifest(
        json.loads((Path(model["directory"]) / "manifest.json").read_text())
    )
    write(args.output / "model-legacy-manifest.json", legacy)
    agent_sha = hashlib.sha256(args.agent.read_bytes()).hexdigest()
    agent_key = (
        config["storage"]["prefix"] + "/agents/" + agent_sha + "/switchyard-agent"
    )
    control.upload(args.agent, agent_key)
    write(
        args.output / "prepared.json",
        {
            "runtime": runtime,
            "preparation": preparation,
            "model": model,
            "legacy_manifest": str(
                (args.output / "model-legacy-manifest.json").resolve()
            ),
            "agent": {"key": agent_key, "sha256": agent_sha},
            "compatibility": compatibility,
            "accuracy": config["numerical_accuracy"],
            "gpu_bake_completed": False,
            **(
                {
                    "shared_runtime": shared["launch_configuration"],
                    "native_config_location": "preparation",
                }
                if shared
                else {}
            ),
        },
    )
    if (
        previous
        and previous.get("gpu_bake_completed")
        and previous["compatibility"] == compatibility
        and previous["model"]["id"] == model["id"]
        and previous["cache_compatibility"].get("runtime_id") == runtime["id"]
        and previous["cache_compatibility"].get("model_revision")
        == config["model_revision"]
    ):
        selected = json.loads((args.output / "prepared.json").read_text())
        for key in ["cache", "cache_compatibility", "bake_receipt"]:
            selected[key] = previous[key]
        selected["gpu_bake_completed"] = True
        write(args.output / "prepared.json", selected)
    control.event(
        "artifacts_published",
        runtime_id=runtime["id"],
        model_id=model["id"],
        runtime_bytes=runtime["compressed_bytes"],
        model_bytes=model["compressed_bytes"],
    )


if __name__ == "__main__":
    main()
