"""Bake native CUDA cache on one owned cloud GPU, publish it, then terminate."""

import argparse
import json
import shlex
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

from control import Control, write
from prepare_artifacts import pack_publish


def remote(endpoint, known_hosts, command, *, input=None, timeout=600):
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=" + str(known_hosts),
            "-p",
            str(endpoint["port"]),
            endpoint["username"] + "@" + endpoint["host"],
            command,
        ],
        input=input,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def hydrate_script(control, reference, destination):
    manifest = json.loads((Path(reference["directory"]) / "manifest.json").read_text())
    script = "mkdir -p " + shlex.quote(destination) + "\n"
    for chunk in manifest["chunks"]:
        url = control.url(reference["prefix"] + "/" + chunk["name"])
        script += (
            "curl -fsSL --max-time 300 "
            + shlex.quote(url)
            + " | zstd -dc | tar -xf - -C "
            + shlex.quote(destination)
            + "\n"
        )
    return script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    prepared = json.loads(args.prepared.read_text())
    control = Control(config, args.output)
    name = "switchyard-native-bake-" + uuid.uuid4().hex[:12]
    try:
        pod = control.allocate(
            name,
            ["/bin/bash", "-c"],
            [Path(__file__).with_name("bake-pod.sh").read_text()],
            {
                "SWITCHYARD_SSH_PUBLIC_KEY": (Path.home() / ".ssh/id_ed25519.pub")
                .read_text()
                .strip()
            },
            ssh=True,
        )
        deadline = time.monotonic() + config["limits"]["pod_timeout_s"]
        endpoint = None
        while time.monotonic() < deadline:
            pod = control.api("/pods/" + pod["id"])
            endpoint = pod.get("ssh", {}).get("direct")
            if endpoint:
                result = remote(
                    endpoint, args.output / "known_hosts", "true", timeout=8
                )
                if result.returncode == 0:
                    break
            if pod["status"] in ["ERROR", "EXITED", "TERMINATED"]:
                raise RuntimeError(
                    "owned bake container stopped before control listener"
                )
            time.sleep(2)
        else:
            raise RuntimeError("owned bake control deadline exceeded")
        control.event("bake_container_ready", pod_id=pod["id"])
        script = "set -euo pipefail\n"
        script += hydrate_script(
            control, prepared["runtime"], "/opt/switchyard/runtime"
        )
        script += hydrate_script(control, prepared["model"], "/work/switchyard/model")
        script += hydrate_script(
            control, prepared["preparation"], "/opt/switchyard/preparation"
        )
        script += (
            "export LD_LIBRARY_PATH=/opt/switchyard/runtime/lib:${LD_LIBRARY_PATH:-}\n"
        )
        script += "nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total --format=csv,noheader -i 0\n"
        script += "/opt/switchyard/preparation/bin/switchyard-engine-bake --engine /opt/switchyard/runtime/bin/switchyard-engine --config /opt/switchyard/runtime/native-config.json --requests /opt/switchyard/preparation/bake-requests.json --output /work/native-bake\n"
        control.event("offline_bake_started")
        result = remote(
            endpoint,
            args.output / "known_hosts",
            "bash -s",
            input=script.encode(),
            timeout=max(1, int(deadline - time.monotonic())),
        )
        (args.output / "native-bake.log").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("native cloud bake failed; inspect sanitized owned log")
        control.event("offline_bake_finished")
        fetch = remote(
            endpoint,
            args.output / "known_hosts",
            "tar -C /work/native-bake -czf - .",
            timeout=120,
        )
        if fetch.returncode:
            raise RuntimeError("bake output download failed")
        archive = args.output / "bake.tar.gz"
        archive.write_bytes(fetch.stdout)
        tree = args.output / "tree"
        tree.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(tree, filter="data")
        receipt = json.loads((tree / "bake.json").read_text())
        if (
            receipt["engine_sha256"] != prepared["compatibility"]["engine_sha256"]
            or receipt["config"] != config["engine"]
        ):
            raise RuntimeError("native bake identity or configuration mismatch")
        gpu, _gpu_uuid, driver, capability, memory = [
            item.strip() for item in receipt["hardware"].split(",")
        ]
        if capability != prepared["compatibility"]["gpu_compute_capability"]:
            raise RuntimeError("native bake GPU architecture mismatch")
        if (
            int(memory.split()[0]) * 1024 * 1024
            < config["limits"]["initial_gpu_admission_bytes"]
        ):
            raise RuntimeError("GPU is below declared fixed admission budget")
        labels = {
            **prepared["compatibility"],
            "runtime_id": prepared["runtime"]["id"],
            "model_revision": config["model_revision"],
            "gpu": gpu,
            "driver_version": driver,
            "model_artifact_id": prepared["model"]["id"],
        }
        cache = pack_publish(
            args.tool.resolve(),
            control,
            tree,
            args.output / "cache-artifact",
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
        write(args.output / "prepared-baked.json", prepared)
        control.event(
            "cache_published",
            cache_id=cache["id"],
            driver=driver,
            gpu=gpu,
            cache_bytes=receipt["cache_bytes"],
            cache_files=receipt["cache_files"],
        )
    finally:
        control.cleanup()


if __name__ == "__main__":
    main()
