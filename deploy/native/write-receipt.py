"""Create a portable, nonsecret receipt from a completed native cloud launch."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from control import write
from generation_parity import compare as compare_generation


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "build", "launch", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bake", type=Path)
    args = parser.parse_args()
    config = read(args.config)
    launch = read(args.launch / "receipt.json")
    selected_path = args.launch / "selected-prepared.json"
    if selected_path.exists():
        prepared = read(selected_path)
    elif args.bake:
        prepared = read(args.bake / "prepared-baked.json")
    else:
        raise ValueError("selected prepared identity or --bake is required")
    opportunistic = prepared.get("cache_policy") == "opportunistic"
    if opportunistic and (
        prepared.get("cache") is not None
        or prepared.get("cache_compatibility") is not None
        or prepared.get("bake_receipt")
        or prepared.get("gpu_bake_completed")
    ):
        raise ValueError("cacheless launch must not retain baked cache claims")
    bake_dir = None if opportunistic else Path(prepared["bake_receipt"]).parent.parent
    bake = read(bake_dir / "tree/bake.json") if bake_dir else None
    parity_path = bake_dir / "tree/generation-parity.json" if bake_dir else None
    parity = None
    if parity_path and parity_path.exists():
        parity = read(parity_path)
        if not parity["passed"] or parity != compare_generation(
            prepared["generation_reference"],
            bake,
            read(bake_dir / "tree/generated.json"),
        ):
            raise ValueError("retained cloud generation comparison changed")
    cleanup = read(args.launch / "cleanup.json")
    bake_cleanup = read(bake_dir / "cleanup.json") if bake_dir else None
    if not cleanup["confirmed_absent"] or (
        bake_cleanup and not bake_cleanup["confirmed_absent"]
    ):
        raise ValueError("owned pod cleanup must be confirmed before receipt")
    state = read(args.launch / "agent-state-final.json")
    if state["phase"] != "ready" or state["inference_verified"]:
        raise ValueError(
            "launch must use native health availability without inference probes"
        )
    logs = events(args.launch / "container-events.ndjson")
    native = []
    container = []
    memory_samples = []
    gpu = None
    for record in logs:
        line = record["line"]
        if line.startswith("native_host_memory="):
            memory_samples.append(json.loads(line.split("=", 1)[1]))
            continue
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
    selected_driver = (
        prepared["observed_driver"]
        if opportunistic
        else (
            prepared.get("selected_driver")
            or prepared["cache_compatibility"].get("driver_version")
        )
    )
    if gpu is None or gpu["driver"] != selected_driver:
        raise ValueError("serving hardware does not match native cache")
    artifacts = []
    for key in ["runtime", "model", "cache", "preparation"]:
        ref = prepared[key]
        if ref is None:
            continue
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
        "numerical_parity_measured_in_cloud": parity is not None,
        "numerical_parity_scope": (
            "complete offline bake token IDs against the identical local F32 execution; teacher-forced mathematical gates remain separate"
            if parity
            else "no cloud generation comparison recorded"
        ),
        "image": config["image"],
        "hardware": gpu,
        "pod": launch["pod"],
        "runtime_build": read(args.build),
        "artifacts": artifacts,
        **(
            {"live_model_archive": launch["live_model_archive"]}
            if "live_model_archive" in launch
            else {}
        ),
        "cache_compatibility": prepared["cache_compatibility"],
        "cache_policy": prepared.get("cache_policy", "strict"),
        "cache_family_status": prepared.get("cache_family_status", "hit"),
        **(
            {
                "cache_family_selection": {
                    "driver": prepared["selected_driver"],
                    "component_cache_id": prepared["component_cache"]["id"],
                    "layout": prepared["cache_compatibility"][
                        "native_driver_cache_layout"
                    ],
                }
            }
            if "selected_driver" in prepared
            else {}
        ),
        "resource_bounds": {
            "engine": config["engine"],
            "platform": config["limits"],
            "requested_host": {
                "minimum_ram_gb": config["provider"]["min_ram_gb"],
                "minimum_vcpus": config["provider"]["min_vcpus"],
                "container_disk_gb": config["provider"]["disk_gb"],
            },
        },
        "host_weight_buffers": {
            "aggregate_cap_bytes": config["engine"]["max_weight_bytes"],
            "observed_source_bytes": next(
                item["details"]["bytes"]
                for item in native
                if item["event"] == "weights_read_finished"
            ),
            "scope": "aggregate native safetensors file buffers; excludes other process memory and GPU allocations",
        },
        "host_memory_samples": memory_samples,
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
            **(
                {
                    "generation_comparison": {
                        "passed": True,
                        "rows": parity["rows"],
                        "exact_rows": parity["exact_rows"],
                        "exact_output_token_ids": sum(
                            len(ids) for ids in parity["actual_token_ids"]
                        ),
                        "local_engine_sha256": parity["local_engine_sha256"],
                        "cloud_engine_sha256": parity["cloud_engine_sha256"],
                        "model_artifact_id": parity["model_artifact_id"],
                        "retained_parity_sha256": hashlib.sha256(
                            parity_path.read_bytes()
                        ).hexdigest(),
                    }
                }
                if parity
                else {}
            ),
            "controller_events": events(bake_dir / "controller-events.ndjson"),
            "cleanup": bake_cleanup,
        }
        if bake
        else None,
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
