"""Restore verified native archives and execute the exact offline F32 bake requests."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from control import write
from generation_parity import required
from prepare_artifacts import bake_requests


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "prepared", "requests", "engine", "tool", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not required(config):
        raise ValueError("reference execution requires cache-enabled F32 inputs")
    if sha(args.engine) != args.engine_sha256:
        raise ValueError("local engine differs from selected reference-tested binary")
    prepared = json.loads(args.prepared.read_text())
    archive = Path(prepared["model"]["directory"])
    manifest = json.loads((archive / "manifest.json").read_text())
    if (
        manifest["id"] != prepared["model"]["id"]
        or manifest["metadata"]["compatibility"]["model"] != config["engine"]["model"]
    ):
        raise ValueError("native archive differs from pinned model artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    weights = args.output / "weights"
    weights.mkdir()
    for chunk in manifest["chunks"]:
        source = archive / chunk["name"]
        if source.stat().st_size != chunk["size"] or sha(source) != chunk["sha256"]:
            raise ValueError("local archive chunk differs from manifest")
        subprocess.run(
            ["tar", "--zstd", "-xf", str(source), "-C", str(weights)], check=True
        )
    subprocess.run(
        [
            str(args.tool.resolve()),
            "artifacts",
            "verify",
            "--manifest",
            str(archive / "manifest.json"),
            "--directory",
            str(weights),
        ],
        check=True,
    )
    requests = bake_requests(config, json.loads(args.requests.read_text()))
    if any(
        len(row["prompt"]) + row["max_tokens"] > config["engine"]["max_context"]
        for row in requests
    ):
        raise ValueError("fixed bake requests exceed the declared context")
    write(args.output / "requests.json", requests)
    write(args.output / "native-config.json", config["engine"])
    cache = args.output / "cuda"
    cache.mkdir()
    with (args.output / "native-generation.log").open("w") as log:
        subprocess.run(
            [
                str(args.engine.resolve()),
                "--config",
                str(args.output / "native-config.json"),
                "--model-root",
                str(weights.resolve()),
                "generate",
                "--input",
                str(args.output / "requests.json"),
                "--output",
                str(args.output / "generated.json"),
            ],
            env={
                **os.environ,
                "CUDA_CACHE_PATH": str(cache.resolve()),
                "CUDA_CACHE_DISABLE": "0",
                "CUDA_CACHE_MAXSIZE": "536870912",
            },
            stdout=log,
            stderr=log,
            check=True,
            timeout=300,
        )
    if sha(args.engine) != args.engine_sha256:
        raise ValueError("local reference engine changed during execution")
    generated = json.loads((args.output / "generated.json").read_text())
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
            "-i",
            str(config["engine"]["device"]["ordinal"]),
        ],
        text=True,
    ).strip()
    reference = {
        "format": "switchyard-native-generation-reference-v1",
        "configuration": config["engine"],
        "expected_cloud_build_commit": config["engine_source_commit"],
        "model_artifact_id": prepared["model"]["id"],
        "model_revision": config["model_revision"],
        "weights_verified_against_artifact_manifest": True,
        "local_engine_sha256": args.engine_sha256,
        "hardware": hardware,
        "requests": requests,
        "generated": generated,
        "weight_path_override": "local restored artifact directory; all native configuration fields unchanged",
        "scope": "offline F32 generation of identical future bake requests; separate from published-model mathematical gates and serving readiness",
    }
    from generation_parity import validate_reference

    validate_reference(reference, config, prepared)
    write(args.output / "reference.json", reference)
    print(
        json.dumps(
            {
                "model": config["engine"]["model"],
                "artifact": prepared["model"]["id"],
                "reference_rows": len(generated),
                "reference": str(args.output / "reference.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
