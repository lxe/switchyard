"""Use HTTP-only allocations for native offline bakes and bounded cache downloads."""

import base64
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from control import write
from provider_logs import collect_logs


def run(args, configs, preparations, base, control, single, children, record_model):
    script = "set -euo pipefail\n"
    script += single.hydrate_script(
        control, preparations[0]["runtime"], "/opt/switchyard/runtime"
    )
    script += (
        "export LD_LIBRARY_PATH=/opt/switchyard/runtime/lib:${LD_LIBRARY_PATH:-}\n"
    )
    for config, prepared in zip(configs, preparations):
        name = config["engine"]["model"]
        script += "rm -rf /work/switchyard/model /opt/switchyard/preparation\n"
        script += single.hydrate_script(
            control, prepared["model"], "/work/switchyard/model"
        )
        script += single.hydrate_script(
            control, prepared["preparation"], "/opt/switchyard/preparation"
        )
        destination = "/work/native-model-bakes/" + name
        script += "mkdir -p " + destination + "\n"
        script += (
            "/opt/switchyard/preparation/bin/switchyard-engine-bake --engine /opt/switchyard/runtime/bin/switchyard-engine --config /opt/switchyard/preparation/native-config.json --requests /opt/switchyard/preparation/bake-requests.json --output "
            + destination
            + "/tree\n"
        )
        # Rename after the archive is complete; status never exposes a partial bundle.
        script += (
            "tar -C "
            + destination
            + "/tree -czf "
            + destination
            + "/bake.tar.gz.tmp .\n"
        )
        script += (
            "mv " + destination + "/bake.tar.gz.tmp " + destination + "/bake.tar.gz\n"
        )
    server = base64.b64encode(
        Path(__file__).with_name("bake_http_server.py").read_bytes()
    ).decode()
    bootstrap = (
        "set -euo pipefail\nexport DEBIAN_FRONTEND=noninteractive\napt-get update -qq\napt-get install --no-install-recommends -y python3 zstd\nmkdir -p /opt/switchyard\nprintf '%s' '"
        + server
        + "' | base64 -d > /opt/switchyard/bake-http-server.py\nexec python3 /opt/switchyard/bake-http-server.py\n"
    )
    token = secrets.token_urlsafe(32)
    pod = control.allocate(
        "switchyard-native-bake-" + uuid.uuid4().hex[:12],
        ["/bin/bash", "-c"],
        [bootstrap],
        {
            "SWITCHYARD_BAKE_TOKEN": token,
            "SWITCHYARD_BAKE_SCRIPT": base64.b64encode(script.encode()).decode(),
            "SWITCHYARD_BAKE_SETTINGS": json.dumps(
                {
                    "models": [c["engine"]["model"] for c in configs],
                    "gpu": base["provider"]["gpu_id"],
                    "capability": preparations[0]["compatibility"][
                        "gpu_compute_capability"
                    ],
                    "minimum_gpu_bytes": base["limits"]["initial_gpu_admission_bytes"],
                    "excluded_drivers": base["excluded_drivers"],
                    "driver_target": base["driver_target"],
                }
            ),
        },
        ssh=False,
    )
    deadline = time.monotonic() + base["limits"]["pod_timeout_s"]
    stopped = threading.Event()
    logs = threading.Thread(
        target=collect_logs, args=(control, pod["id"], stopped, (token,)), daemon=True
    )
    logs.start()
    try:
        consume(
            args,
            configs,
            preparations,
            base,
            control,
            single,
            children,
            record_model,
            pod,
            token,
            deadline,
        )
    finally:
        capture_log(
            args.output, "https://" + pod["id"] + "-8080.proxy.runpod.net", token
        )
        stopped.set()
        logs.join(timeout=5)


def consume(
    args,
    configs,
    preparations,
    base,
    control,
    single,
    children,
    record_model,
    pod,
    token,
    deadline,
):
    origin = "https://" + pod["id"] + "-8080.proxy.runpod.net"
    downloaded = set()
    identity = None
    started = False
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            origin + "/status",
            headers={
                "Authorization": "Bearer " + token,
                "User-Agent": "Switchyard/0.5",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=min(10, max(1, deadline - time.monotonic()))
            ) as response:
                state = json.load(response)
        except urllib.error.HTTPError as error:
            error.close()
            time.sleep(1)
            continue
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(1)
            continue
        write(args.output / "http-state.json", state)
        if state.get("hardware") and identity is None:
            identity = state["hardware"]
            gpu, gpu_uuid, driver, cap, memory = [
                part.strip() for part in identity.split(",")
            ]
            write(
                args.output / "hardware.json",
                {
                    "gpu": gpu,
                    "uuid": gpu_uuid,
                    "driver": driver,
                    "capability": cap,
                    "memory": memory,
                },
            )
            control.event(
                "bake_hardware_observed",
                pod_id=pod["id"],
                gpu=gpu,
                driver=driver,
                capability=cap,
            )
        if state["phase"] == "failed":
            raise RuntimeError("owned HTTP bake failed: " + str(state.get("error")))
        if state["phase"] == "waiting" and not started:
            request = urllib.request.Request(
                origin + "/start",
                data=b"",
                method="POST",
                headers={
                    "Authorization": "Bearer " + token,
                    "User-Agent": "Switchyard/0.5",
                },
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status != 204:
                    raise RuntimeError("fixed native bake gate was not released")
            started = True
            control.event(
                "offline_bakes_started",
                driver=driver,
                models=[c["engine"]["model"] for c in configs],
            )
        for config, prepared in zip(configs, preparations):
            name = config["engine"]["model"]
            if name not in state["completed"] or name in downloaded:
                continue
            output = args.output / name
            output.mkdir()
            children.append(output)
            request = urllib.request.Request(
                origin + "/bundle/" + name,
                headers={
                    "Authorization": "Bearer " + token,
                    "User-Agent": "Switchyard/0.5",
                },
            )
            with urllib.request.urlopen(
                request, timeout=min(60, max(1, deadline - time.monotonic()))
            ) as response:
                length = int(response.headers["Content-Length"])
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("native cache bundle exceeds bounded download")
                data = response.read(length + 1)
                if len(data) != length:
                    raise ValueError(
                        "native cache bundle length differs from declaration"
                    )
            archive = output / "bake.tar.gz"
            archive.write_bytes(data)
            tree = output / "tree"
            tree.mkdir()
            with single.tarfile.open(archive) as source:
                source.extractall(tree, filter="data")
            record_model(config, prepared, output, tree, args.tool, control, identity)
            downloaded.add(name)
        if state["phase"] == "complete" and len(downloaded) == len(configs):
            return
        time.sleep(1)
    raise RuntimeError("owned HTTP native model bake deadline exceeded")


def capture_log(output, origin, token):
    request = urllib.request.Request(
        origin + "/log",
        headers={"Authorization": "Bearer " + token, "User-Agent": "Switchyard/0.5"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            data = response.read(65537)
            if len(data) > 65536:
                raise ValueError("bake diagnostic log exceeds bounded tail")
            (output / "native-bake.log").write_bytes(data)
    except (OSError, ValueError):
        # A failed diagnostic download must not prevent owned-pod cleanup.
        pass
