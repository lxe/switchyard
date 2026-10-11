"""Create a portable, nonsecret receipt from a completed native cloud launch."""

import argparse
import datetime
import json
from pathlib import Path

from control import write


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "build", "bake", "launch", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    launch = read(args.launch / "receipt.json")
    prepared = (
        read(args.launch / "selected-prepared.json")
        if (args.launch / "selected-prepared.json").exists()
        else read(args.bake / "prepared-baked.json")
    )
    bake_dir = Path(prepared["bake_receipt"]).parent.parent
    bake = read(bake_dir / "tree/bake.json")
    cleanup = read(args.launch / "cleanup.json")
    bake_cleanup = read(bake_dir / "cleanup.json")
    if not cleanup["confirmed_absent"] or not bake_cleanup["confirmed_absent"]:
        raise ValueError("owned pod cleanup must be confirmed before receipt")
    state = read(args.launch / "agent-state-final.json")
    if state["phase"] != "ready" or state["inference_verified"]:
        raise ValueError(
            "launch must use native health availability without inference probes"
        )
    logs = events(args.launch / "container-events.ndjson")
    native = []
    container = []
    gpu = None
    for record in logs:
        line = record["line"]
        if line.startswith("native_host_gpu="):
            name, gpu_uuid, driver, capability, memory = [
                part.strip() for part in line.split("=", 1)[1].split(",")
            ]
            gpu = {
                "name": name,
                "uuid": gpu_uuid,
                "driver": driver,
                "compute_capability": capability,
                "total_memory_mib": int(memory.split()[0]),
            }
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if "event" in item:
            native.append({"provider_ts": record["ts"], **item})
        if "stage" in item and "wall_s" in item:
            container.append(
                {
                    "stage": item["stage"],
                    "allocation_offset_s": item["wall_s"]
                    - launch["allocation_started_wall_s"],
                    "provider_ts": record["ts"],
                }
            )
    required = {
        "process_started",
        "device_initialized",
        "weights_read_finished",
        "model_loaded",
        "http_listening",
        "batch_finished",
    }
    if not required.issubset({item["event"] for item in native}):
        raise ValueError("native phase telemetry is incomplete")
    if (
        gpu is None
        or gpu["driver"] != prepared["cache_compatibility"]["driver_version"]
    ):
        raise ValueError("serving hardware does not match native cache")
    artifacts = []
    for key in ["runtime", "model", "cache", "preparation"]:
        ref = prepared[key]
        manifest = read(Path(ref["directory"]) / "manifest.json")
        artifacts.append(
            {
                "role": key,
                "id": ref["id"],
                "prefix": ref["prefix"],
                "compressed_bytes": ref["compressed_bytes"],
                "shards": len(manifest["chunks"]),
                "files": [
                    {"path": e["path"], "bytes": e["size"]}
                    for e in manifest["entries"]
                    if e["kind"] == "file"
                ],
            }
        )
    launched = datetime.datetime.fromtimestamp(
        launch["allocation_started_wall_s"], datetime.timezone.utc
    ).isoformat()
    receipt = {
        "format": "switchyard-native-cloud-evidence-v1",
        "model": config["engine"]["model"],
        "model_revision": config["model_revision"],
        "allocation_started_utc": launched,
        "engine_source_commit": launch["source_commit"],
        "engine_sha256": launch["engine_sha256"],
        "accuracy": config["numerical_accuracy"],
        "numerical_parity_measured_in_cloud": False,
        "image": config["image"],
        "hardware": gpu,
        "pod": launch["pod"],
        "runtime_build": read(args.build),
        "artifacts": artifacts,
        "cache_compatibility": prepared["cache_compatibility"],
        "resource_bounds": {
            "engine": config["engine"],
            "platform": config["limits"],
            "requested_host": {
                "minimum_ram_gb": config["provider"]["min_ram_gb"],
                "minimum_vcpus": config["provider"]["min_vcpus"],
                "container_disk_gb": config["provider"]["disk_gb"],
            },
        },
        "timing_s": {
            key: launch[key]
            for key in [
                "agent_listener_allocation_s",
                "ready_allocation_s",
                "first_content_allocation_s",
                "request_to_first_content_s",
            ]
        },
        "container_stages": container,
        "native_process_events": native,
        "provider_lifecycle_events": [r for r in logs if r["source"] == "system"],
        "agent_state": state,
        "request": launch["request"],
        "frames": launch["frames"],
        "workloads": launch.get("workloads"),
        "offline_bake": {
            "engine_sha256": bake["engine_sha256"],
            "hardware": bake["hardware"],
            "native_generation_s": bake["elapsed_s"],
            "cache_bytes": bake["cache_bytes"],
            "cache_files": bake["cache_files"],
            "controller_events": events(bake_dir / "controller-events.ndjson"),
            "cleanup": bake_cleanup,
        },
        "launch_controller_events": events(args.launch / "controller-events.ndjson"),
        "cleanup": cleanup,
        "limitations": {
            "hard_gpu_allocator_cap_enforced": False,
            "direct_gpu_weight_chunk_consumer": False,
            "native_weight_reader": "aggregate host buffer bounded by max_weight_bytes",
            "model_transport": "bounded Rust archive streaming into native files",
            "image_cache_residency": "unknown",
            "serving_python": False,
            "torch": False,
            "vllm": False,
            "criu": False,
            "serving_content_hashes": False,
            "inference_readiness_probe": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, receipt)
    print(
        json.dumps(
            {
                "model": receipt["model"],
                "timing_s": receipt["timing_s"],
                "cleanup": True,
            }
        )
    )


if __name__ == "__main__":
    main()
