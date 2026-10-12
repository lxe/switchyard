"""Bake multiple fixed native configurations sequentially on one owned driver host."""

import argparse
import importlib.util
import json
import re
import shutil
import time
import uuid
from pathlib import Path

from control import Control, write
from generation_parity import required, validate_reference, verify
from prepare_artifacts import pack_publish


def publish_model(config, prepared, output, tree, tool, control, identity):
    gpu, _gpu_uuid, driver, _capability, _memory = [
        part.strip() for part in identity.split(",")
    ]
    receipt = json.loads((tree / "bake.json").read_text())
    if (
        receipt["engine_sha256"] != prepared["compatibility"]["engine_sha256"]
        or receipt["config"] != config["engine"]
        or receipt["hardware"] != identity
    ):
        raise RuntimeError(
            "native bake binary, model configuration, or hardware changed"
        )
    if required(config):
        reference = prepared["generation_reference"]
        validate_reference(reference, config, prepared)
        verify(
            reference,
            receipt,
            json.loads((tree / "generated.json").read_text()),
            tree / "generation-parity.json",
        )
    labels = {
        **prepared["compatibility"],
        "runtime_id": prepared["runtime"]["id"],
        "model_revision": config["model_revision"],
        "gpu": gpu,
        "driver_version": driver,
        "model_artifact_id": prepared["model"]["id"],
    }
    cache = pack_publish(
        tool.resolve(),
        control,
        tree,
        output / "cache-artifact",
        {
            "kind": "native_gpu_cache",
            "compatibility": labels,
            "attributes": {
                "accuracy": config["numerical_accuracy"],
                "gpu_snapshot": "false",
                "offline_bake": "true",
            },
        },
        "caches",
    )
    prepared.update(
        cache=cache,
        cache_compatibility=labels,
        gpu_bake_completed=True,
        bake_receipt=str((tree / "bake.json").resolve()),
    )
    write(output / "prepared-baked.json", prepared)
    control.event(
        "cache_published",
        model=config["engine"]["model"],
        cache_id=cache["id"],
        driver=driver,
        cache_bytes=receipt["cache_bytes"],
        cache_files=receipt["cache_files"],
    )


def bake_and_publish(control, children, operation):
    completed = []
    try:
        operation(lambda *record: completed.append(record))
    finally:
        control.cleanup()
        for child in children:
            for name in ["owned-pod.json", "cleanup.json", "controller-events.ndjson"]:
                source = control.output / name
                if source.exists():
                    shutil.copy2(source, child / name)
    # Local packing and object-store publication must not extend GPU ownership.
    for record in completed:
        publish_model(*record)


def run_ssh(args, configs, preparations, base, control, single, children, record_model):
    pod = control.allocate(
        "switchyard-native-bake-" + uuid.uuid4().hex[:12],
        ["/bin/bash", "-c"],
        [Path(__file__).with_name("bake-pod.sh").read_text()],
        {
            "SWITCHYARD_SSH_PUBLIC_KEY": (Path.home() / ".ssh/id_ed25519.pub")
            .read_text()
            .strip()
        },
        ssh=True,
    )
    deadline = time.monotonic() + base["limits"]["pod_timeout_s"]
    hosts = args.output / "known_hosts"
    while time.monotonic() < deadline:
        pod = control.api("/pods/" + pod["id"])
        endpoint = pod.get("ssh", {}).get("direct")
        if (
            endpoint
            and single.remote(endpoint, hosts, "true", timeout=8).returncode == 0
        ):
            break
        if pod["status"] in ["ERROR", "EXITED", "TERMINATED"]:
            raise RuntimeError("owned bake host stopped before control listener")
        time.sleep(2)
    else:
        raise RuntimeError("owned bake control deadline exceeded")
    hardware = single.remote(
        endpoint,
        hosts,
        "nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total --format=csv,noheader -i 0",
        timeout=10,
    )
    if hardware.returncode:
        raise RuntimeError("owned bake hardware observation failed")
    identity = hardware.stdout.decode().strip()
    gpu, gpu_uuid, driver, capability, memory = [
        part.strip() for part in identity.split(",")
    ]
    if (
        gpu != base["provider"]["gpu_id"]
        or capability != preparations[0]["compatibility"]["gpu_compute_capability"]
        or not re.fullmatch(r"\d+\.\d+\.\d+", driver)
        or int(memory.split()[0]) * 1024 * 1024
        < base["limits"]["initial_gpu_admission_bytes"]
    ):
        raise RuntimeError("owned bake hardware does not satisfy fixed admission")
    write(
        args.output / "hardware.json",
        {
            "gpu": gpu,
            "uuid": gpu_uuid,
            "driver": driver,
            "capability": capability,
            "memory": memory,
        },
    )
    control.event(
        "bake_hardware_observed",
        pod_id=pod["id"],
        gpu=gpu,
        driver=driver,
        capability=capability,
    )
    if driver in base["excluded_drivers"]:
        raise RuntimeError("observed driver is excluded by bake platform policy")
    if base["driver_target"] is not None and driver != base["driver_target"]:
        raise RuntimeError("observed driver differs from explicit bake target")
    runtime_script = "set -euo pipefail\n" + single.hydrate_script(
        control, preparations[0]["runtime"], "/opt/switchyard/runtime"
    )
    result = single.remote(
        endpoint,
        hosts,
        "bash -s",
        input=runtime_script.encode(),
        timeout=max(1, int(deadline - time.monotonic())),
    )
    if result.returncode:
        raise RuntimeError("shared runtime hydration failed")
    for config, prepared in zip(configs, preparations):
        output = args.output / config["engine"]["model"]
        output.mkdir()
        children.append(output)
        script = "set -euo pipefail\nrm -rf /work/switchyard/model /opt/switchyard/preparation /work/native-bake\n"
        script += single.hydrate_script(
            control, prepared["model"], "/work/switchyard/model"
        )
        script += single.hydrate_script(
            control, prepared["preparation"], "/opt/switchyard/preparation"
        )
        script += (
            "export LD_LIBRARY_PATH=/opt/switchyard/runtime/lib:${LD_LIBRARY_PATH:-}\n"
        )
        script += "/opt/switchyard/preparation/bin/switchyard-engine-bake --engine /opt/switchyard/runtime/bin/switchyard-engine --config /opt/switchyard/preparation/native-config.json --requests /opt/switchyard/preparation/bake-requests.json --output /work/native-bake\n"
        control.event(
            "offline_bake_started", model=config["engine"]["model"], driver=driver
        )
        result = single.remote(
            endpoint,
            hosts,
            "bash -s",
            input=script.encode(),
            timeout=max(1, int(deadline - time.monotonic())),
        )
        (output / "native-bake.log").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("native model bake failed; inspect owned log")
        fetch = single.remote(
            endpoint,
            hosts,
            "tar -C /work/native-bake -czf - .",
            timeout=max(1, min(120, int(deadline - time.monotonic()))),
        )
        if fetch.returncode:
            raise RuntimeError("native cache download failed")
        archive = output / "bake.tar.gz"
        archive.write_bytes(fetch.stdout)
        tree = output / "tree"
        tree.mkdir()
        with single.tarfile.open(archive) as source:
            source.extractall(tree, filter="data")
        record_model(config, prepared, output, tree, args.tool, control, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--prepared", type=Path, action="append", required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-config", type=Path, required=True)
    parser.add_argument("--generation-reference", type=Path, action="append")
    args = parser.parse_args()
    if len(args.config) != len(args.prepared):
        raise ValueError("each configuration requires one matching preparation")
    configs = [json.loads(path.read_text()) for path in args.config]
    preparations = [json.loads(path.read_text()) for path in args.prepared]
    references = args.generation_reference or []
    if references and len(references) != len(configs):
        raise ValueError("each model requires one paired local generation reference")
    base = json.loads(args.platform_config.read_text())
    if (
        base["purpose"] != "native_offline_driver_cache_bake"
        or base["limits"]["max_active_owned_pods"] != 1
        or base["limits"]["pod_timeout_s"] != 600
        or base["control"] not in ["http", "ssh"]
    ):
        raise ValueError("explicit bounded offline bake platform policy required")
    names = set()
    for index, (config, prepared) in enumerate(zip(configs, preparations)):
        name = config["engine"]["model"]
        if (
            name in names
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name)
            or prepared.get("native_config_location") != "preparation"
        ):
            raise ValueError("unique prepared shared-runtime native models required")
        names.add(name)
        if (
            config["engine"]["model_root"] != "/work/switchyard/model"
            or config["engine_source_commit"]
            != prepared["compatibility"]["engine_source_commit"]
            or config["image"] != base["image"]
            or config["storage"] != base["storage"]
            or config["limits"]["max_active_owned_pods"] != 1
            or config["limits"]["pod_timeout_s"] != 600
            or config["limits"]["max_hourly_usd"] != base["limits"]["max_hourly_usd"]
            or prepared["runtime"]["id"] != preparations[0]["runtime"]["id"]
            or prepared["shared_runtime"] != preparations[0]["shared_runtime"]
            or name not in base["models"]
            or config["provider"]["min_ram_gb"] > base["provider"]["min_ram_gb"]
            or config["provider"]["min_vcpus"] > base["provider"]["min_vcpus"]
            or config["provider"]["disk_gb"] > base["provider"]["disk_gb"]
            or config["limits"]["initial_gpu_admission_bytes"]
            > base["limits"]["initial_gpu_admission_bytes"]
            or prepared["compatibility"]["gpu_compute_capability"]
            != base["gpu_compute_capability"]
        ):
            raise ValueError(
                "all model bakes require the same fixed provider/runtime policy"
            )
        if required(config):
            if not references:
                raise ValueError(
                    "cache-enabled F32 bakes require local generation references before allocation"
                )
            reference = json.loads(references[index].read_text())
            validate_reference(reference, config, prepared)
            prepared["generation_reference"] = reference
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "allocation-policy.json", base)
    spec = importlib.util.spec_from_file_location(
        "native_single_bake", Path(__file__).with_name("run-bake.py")
    )
    single = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(single)
    control = Control(base, args.output)
    children = []

    def operation(record_model):
        if base["control"] == "http":
            from http_model_bakes import run

            run(
                args,
                configs,
                preparations,
                base,
                control,
                single,
                children,
                record_model,
            )
            return
        run_ssh(
            args, configs, preparations, base, control, single, children, record_model
        )

    bake_and_publish(control, children, operation)


if __name__ == "__main__":
    main()
