"""Publish a legacy manifest for native archives using bounded R2 server copies."""

import argparse
import hashlib
import json
from pathlib import Path

from control import Control, write
from prepare_artifacts import legacy_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "prepared", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    prepared = json.loads(args.prepared.read_text())
    model = prepared["model"]
    directory = Path(model["directory"])
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["id"] != model["id"]
        or manifest["metadata"]["compatibility"]["model"] != config["engine"]["model"]
    ):
        raise ValueError("published native model differs from selected profile")
    legacy = legacy_manifest(manifest)
    prefix = config["storage"]["prefix"] + "/live-models/" + legacy["sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    control = Control(config, args.output)
    chunks = []
    for chunk in manifest["chunks"]:
        path = directory / chunk["name"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != chunk["size"] or digest != chunk["sha256"]:
            raise ValueError("native model archive differs from immutable manifest")
        control.copy_immutable(
            model["prefix"] + "/" + chunk["name"],
            path,
            prefix + "/" + chunk["name"],
            chunk["sha256"],
        )
        chunks.append(
            {
                "name": chunk["name"],
                "bytes": chunk["size"],
                "expected_sha256": chunk["sha256"],
                "source_key": model["prefix"] + "/" + chunk["name"],
                "target_length_verified": True,
                "target_full_hash_readback": False,
            }
        )
    target = args.output / "manifest.json"
    write(target, legacy)
    control.upload_immutable(
        target,
        prefix + "/manifest.json",
        hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    write(
        args.output / "live-model.json",
        {
            "model": config["engine"]["model"],
            "model_revision": config["model_revision"],
            "native_artifact_id": model["id"],
            "archive_prefix": prefix,
            "archive_sha256": legacy["sha256"],
            "published": True,
            "manifest_published_last": True,
            "transport": "AWS CLI bounded multipart server-side R2 copy",
            "chunks": chunks,
        },
    )
    control.event(
        "live_model_published",
        model=config["engine"]["model"],
        archive_prefix=prefix,
        archive_sha256=legacy["sha256"],
    )


if __name__ == "__main__":
    main()
