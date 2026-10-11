"""Bind final tensor weights, warmed caches and a CPU snapshot for one rollout.

The model must already contain the bounded transport's immutable tensor index.
The caller supplies the loader identity from the final integrated agent source.
This helper performs only local JSON preparation and artifact packing.
"""

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bake", type=Path, required=True)
    parser.add_argument("--model-objects", type=Path, required=True)
    parser.add_argument("--cpu-artifact", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--weight-loader-identity", required=True)
    parser.add_argument("--artifact-packer", type=Path, required=True)
    parser.add_argument("--cache-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch("[0-9a-f]{64}", args.weight_loader_identity) is None:
        parser.error("weight loader identity must be a SHA-256 digest")

    model = read(args.model_objects / "manifest.json")
    index = json.loads(model["metadata"]["attributes"]["tensor_destinations_v1"])
    receipt = read(args.bake / "model/switchyard-final-state.json")
    if index["format"] != "switchyard-tensor-destinations-v1":
        raise ValueError("bounded final tensor index required")
    if {item["name"] for item in index["tensors"]} != set(receipt["layout"]):
        raise ValueError("packed tensor index differs from native final tensor state")
    if any(item["layout"] != "dense_c" for item in index["tensors"]):
        raise ValueError("initial final tensor proof requires dense destinations")
    if index["worker"] != {
        "rank": 0,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }:
        raise ValueError("initial final tensor proof requires one worker")
    dtypes = {"BF16": "torch.bfloat16", "F32": "torch.float32"}
    for item in index["tensors"]:
        expected = receipt["layout"][item["name"]]
        if {
            "shape": item["shape"],
            "stride": item["stride"],
            "dtype": dtypes.get(item["dtype"]),
        } != expected:
            raise ValueError(
                "packed tensor index differs from final shape/dtype/stride"
            )
    if (
        len(index["tensors"]) != receipt["tensor_count"]
        or sum(item["bytes"] for item in index["tensors"]) != receipt["tensor_bytes"]
    ):
        raise ValueError("packed tensor index count or byte total mismatch")
    if max(extent["length"] for extent in model["extents"]) > 4 * 1024 * 1024:
        raise ValueError("model needs chunks at most 4 MiB")

    profile = read(args.bake / "serving-profile.unbound.json")
    if profile["final_layout_sha256"] != receipt["layout_sha256"]:
        raise ValueError("profile differs from baked final tensor layout")
    profile["model_content_id"] = model["content_id"]
    write(args.bake / "cache/compiled/serving-profile.json", profile)

    engine_args = [
        "serve",
        "{model_root}",
        "--served-model-name",
        "qwen-local",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--language-model-only",
        "--max-model-len",
        "64",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "64",
        "--gpu-memory-utilization",
        "0.25",
        "--kv-cache-memory-bytes",
        "1073741824",
        "--enforce-eager",
        "--attention-backend",
        "TRITON_ATTN",
        "--seed",
        "123",
    ]
    arguments_identity = hashlib.sha256(
        json.dumps(
            ["switchyard-serving-arguments-v1", engine_args],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    configuration = copy.deepcopy(read(args.runtime_config))
    runtime = configuration["bootstrap"]["runtime"]
    runtime["environment"]["VLLM_ENGINE_STARTUP_POLICY"] = json.dumps(
        read(args.policy), separators=(",", ":")
    )
    runtime["environment"]["VLLM_CPU_SNAPSHOT_ROOT"] = "{components_root}/cpu-process"
    cpu = read(args.cpu_artifact / "manifest.json")
    runtime["components"] = [
        item for item in runtime["components"] if item["name"] != "cpu-process"
    ] + [
        {
            "name": "cpu-process",
            "artifact": {"id": cpu["id"], "prefix": "local-cpu-process/" + cpu["id"]},
            "compatibility": cpu["metadata"]["compatibility"],
        }
    ]
    labels = {
        **runtime["compatibility"],
        "runtime_id": runtime["artifact"]["id"],
        "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "model_content_id": model["content_id"],
        "weight_loader": args.weight_loader_identity,
        "engine_arguments": arguments_identity,
        "gpu": "RTX5090",
        "compute_capability": "12.0",
    }
    metadata = {
        "kind": "cache",
        "compatibility": labels,
        "attributes": {
            "scope": "Native .8B RTX5090 fixed-profile bake; final direct tensor rollout pending",
            "native_warmup_completed": "true",
            "representative_decode_prepared": "true",
            "gpu_snapshot": "false",
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "engine-args.json", engine_args)
    write(args.output / "cache-metadata.json", metadata)
    subprocess.run(
        [
            str(args.artifact_packer.resolve()),
            "artifacts",
            "pack",
            "--source",
            str((args.bake / "cache/compiled").resolve()),
            "--out",
            str(args.cache_artifact.resolve()),
            "--metadata",
            str((args.output / "cache-metadata.json").resolve()),
        ],
        check=True,
    )
    cache = read(args.cache_artifact / "manifest.json")
    configuration["models"][0]["caches"] = [
        {
            "name": "compiled",
            "gpu": "RTX5090",
            "artifact": {
                "id": cache["id"],
                "prefix": "local-final-cache/" + cache["id"],
            },
            "compatibility": labels,
        }
    ]
    write(args.output / "runtime.json", configuration)
    write(
        args.output / "prepared.json",
        {
            "model_manifest_id": model["id"],
            "model_content_id": model["content_id"],
            "cache_id": cache["id"],
            "cpu_process_id": cpu["id"],
            "weight_loader": args.weight_loader_identity,
            "engine_arguments": arguments_identity,
            "final_layout_sha256": receipt["layout_sha256"],
            "tensor_count": len(index["tensors"]),
            "model_objects": str(args.model_objects.resolve()),
            "cache_objects": str(args.cache_artifact.resolve()),
            "cpu_objects": str(args.cpu_artifact.resolve()),
            "cuda_rollout_completed": False,
        },
    )
    print(json.dumps(read(args.output / "prepared.json")), flush=True)


if __name__ == "__main__":
    main()
