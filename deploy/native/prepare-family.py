"""Verify and pack exact driver-cache components for the generic live runtime."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from control import Control, write
from prepare_artifacts import publish_packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, action="append", required=True)
    parser.add_argument("--fleet-config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish-config", type=Path)
    args = parser.parse_args()
    fleet = json.loads(args.fleet_config.read_text())
    selected_model = next(
        model for model in fleet["models"] if model["id"] == args.model
    )
    engine = selected_model["engine"]
    if engine["kind"] != "native":
        raise ValueError("driver-cache families require a native fleet model")
    native = engine["native"]
    runtime_compatibility = fleet["bootstrap"]["runtime"]["compatibility"]
    expected_configuration = {
        "model": args.model,
        "device": {"kind": "cuda", "ordinal": native["cuda_ordinal"]},
        "dtype": engine["dtype"],
        "max_context": engine["max_model_len"],
        "max_weight_bytes": native["max_weight_bytes"],
        "max_batch_size": engine["max_num_seqs"],
        "queue_capacity": native["queue_capacity"],
        "batch_wait_ms": native["batch_wait_ms"],
    }
    if "cache" in native:
        expected_configuration["cache"] = native["cache"]
    args.output.mkdir(parents=True, exist_ok=False)
    # Rust validates the typed fleet and derives every generic fingerprint.
    labels_path = args.output / "cache-bindings.json"
    gpu = None
    for path in args.prepared:
        candidate = json.loads(path.read_text())["cache_compatibility"]["gpu"]
        if gpu is not None and gpu != candidate:
            raise ValueError("driver family components must use one GPU name")
        gpu = candidate
    subprocess.run(
        [
            str(args.tool.resolve()),
            "live",
            "cache-bindings",
            "--config",
            str(args.fleet_config),
            "--gpu",
            gpu,
            "--model",
            args.model,
            "--out",
            str(labels_path),
        ],
        check=True,
    )
    bindings = json.loads(labels_path.read_text())[args.model]
    required = {
        "runtime_id",
        "model_revision",
        "model_repository",
        "model_archive_sha256",
        "engine_configuration",
        "runtime_launch_configuration",
        "gpu",
    }
    if not required.issubset(bindings) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v
        for k, v in bindings.items()
    ):
        raise ValueError("complete generic Rust live cache bindings are required")
    tree = args.output / "tree"
    tree.mkdir()
    components = {}
    prepared_components = {}
    common = None
    configuration = None
    for path in args.prepared:
        prepared = json.loads(path.read_text())
        if not prepared["gpu_bake_completed"]:
            raise ValueError("every family component requires a completed native bake")
        cache = prepared["cache"]
        manifest_path = Path(cache["directory"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        labels = prepared["cache_compatibility"]
        for key in ["engine_sha256", "gpu_compute_capability"]:
            if labels[key] != runtime_compatibility[key]:
                raise ValueError("component differs from shared-runtime " + key)
        if (
            "cuda_compiler_target" in runtime_compatibility
            and labels.get("cuda_compiler_target")
            != runtime_compatibility["cuda_compiler_target"]
        ):
            raise ValueError("component compiler target differs from shared runtime")
        if (
            manifest["id"] != cache["id"]
            or manifest["metadata"]["compatibility"] != labels
        ):
            raise ValueError("component artifact or compatibility identity changed")
        driver = labels["driver_version"]
        if not re.fullmatch(r"\d+\.\d+\.\d+", driver) or driver in components:
            raise ValueError("each family component requires a unique exact driver")
        for key in ["runtime_id", "model_revision", "gpu"]:
            if labels[key] != bindings[key]:
                raise ValueError("component differs from generic live " + key)
        legacy = json.loads(Path(prepared["legacy_manifest"]).read_text())
        if legacy["sha256"] != bindings["model_archive_sha256"]:
            raise ValueError("component model archive differs from live binding")
        selected_common = {
            key: value for key, value in labels.items() if key != "driver_version"
        }
        if common is not None and common != selected_common:
            raise ValueError(
                "family components must have identical runtime/model/configuration"
            )
        common = selected_common
        source = Path(prepared["bake_receipt"]).parent
        cleanup = json.loads((source.parent / "cleanup.json").read_text())
        if not cleanup["confirmed_absent"]:
            raise ValueError("component bake pod cleanup must be confirmed")
        receipt = json.loads((source / "bake.json").read_text())
        hardware = [part.strip() for part in receipt["hardware"].split(",")]
        if (
            receipt["engine_sha256"] != labels["engine_sha256"]
            or receipt["config"]["model"] != args.model
            or hardware[0] != labels["gpu"]
            or hardware[2] != driver
            or hardware[3] != labels["gpu_compute_capability"]
            or int(hardware[4].split()[0]) * 1024 * 1024
            < native["minimum_gpu_memory_bytes"]
            or (configuration is not None and receipt["config"] != configuration)
        ):
            raise ValueError("component bake receipt differs from family identity")
        configuration = receipt["config"]
        if {
            key: value for key, value in configuration.items() if key != "model_root"
        } != expected_configuration:
            raise ValueError("baked native bounds differ from the validated live fleet")
        if any(item.is_symlink() for item in source.rglob("*")):
            raise ValueError("native cache family components cannot contain symlinks")
        subprocess.run(
            [
                str(args.tool.resolve()),
                "artifacts",
                "verify",
                "--manifest",
                str(manifest_path),
                "--directory",
                str(source),
            ],
            check=True,
        )
        destination = tree / "drivers" / driver
        shutil.copytree(source, destination)
        shutil.copy2(manifest_path, destination / "component-manifest.json")
        (destination / "identity.tsv").write_text(
            "\t".join([labels["gpu"], driver, labels["gpu_compute_capability"]]) + "\n"
        )
        components[driver] = {
            "artifact": {k: cache[k] for k in ["id", "prefix"]},
            "bake_receipt": "drivers/" + driver + "/bake.json",
            "component_manifest": "drivers/" + driver + "/component-manifest.json",
            "cache_bytes": receipt["cache_bytes"],
            "cache_files": receipt["cache_files"],
            "compatibility": labels,
        }
        prepared_components[driver] = str(path.resolve())
    compatibility = {**bindings}
    for key, value in common.items():
        if key in compatibility and compatibility[key] != value:
            raise ValueError("native family label overrides generated live identity")
        compatibility[key] = value
    compatibility["native_driver_cache_layout"] = "native-cuda-driver-family-v1"
    write(
        tree / "driver-family.json",
        {
            "format": "native-cuda-driver-family-v1",
            "model": args.model,
            "configuration": configuration,
            "compatibility": compatibility,
            "components": components,
            "component_contents_preserved": True,
            "serve_time_content_hashes": False,
        },
    )
    metadata = args.output / "family.metadata.json"
    write(metadata, {"kind": "native_gpu_cache_family", "compatibility": compatibility})
    artifact = args.output / "cache-artifact"
    subprocess.run(
        [
            str(args.tool.resolve()),
            "artifacts",
            "pack",
            "--source",
            str(tree),
            "--out",
            str(artifact),
            "--metadata",
            str(metadata),
        ],
        check=True,
    )
    packed = json.loads((artifact / "manifest.json").read_text())
    write(
        args.output / "family.json",
        {
            "artifact_id": packed["id"],
            "compatibility": compatibility,
            "drivers": sorted(components),
            "published": False,
            "prepared_components": prepared_components,
        },
    )
    if args.publish_config:
        control = Control(json.loads(args.publish_config.read_text()), args.output)
        publish_packed(control, artifact, "cache-families")
        family = json.loads((args.output / "family.json").read_text())
        family.update(
            published=True,
            prefix=control.config["storage"]["prefix"]
            + "/cache-families/"
            + packed["id"],
        )
        family["cache"] = {
            "id": packed["id"],
            "prefix": family["prefix"],
            "directory": str(artifact.resolve()),
            "compressed_bytes": sum(chunk["size"] for chunk in packed["chunks"]),
        }
        write(args.output / "family.json", family)
    print(
        json.dumps(
            {
                "artifact_id": packed["id"],
                "drivers": sorted(components),
                "published": bool(args.publish_config),
            }
        )
    )


if __name__ == "__main__":
    main()
