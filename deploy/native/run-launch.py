"""Measure one fresh RunPod allocation through first native streamed content."""

import argparse
import copy
import hashlib
import json
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from control import Control, write
from workloads import run_workloads


def ticket_artifact(control, reference):
    manifest = json.loads((Path(reference["directory"]) / "manifest.json").read_text())
    return {
        "manifest": manifest,
        "urls": {
            c["name"]: control.url(reference["prefix"] + "/" + c["name"])
            for c in manifest["chunks"]
        },
    }


def collect_logs(control, pod_id, stopped, bootstrap_errors):
    seen = set()
    reported_errors = set()
    while not stopped.is_set():
        # Every allocation has a fresh container; retain both provider and native
        # history without filtering on differences between controller/host clocks.
        params = urllib.parse.urlencode({"tail": 5000})
        request = urllib.request.Request(
            control.config["provider"]["api_url"]
            + "/pods/"
            + pod_id
            + "/logs?"
            + params,
            headers={
                "Authorization": "Bearer " + control.key,
                "User-Agent": "Switchyard/0.5",
                "Accept": "text/event-stream",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                until = time.monotonic() + 3
                while not stopped.is_set() and time.monotonic() < until:
                    line = (
                        response.readline(1048576)
                        .decode("utf8", errors="replace")
                        .strip()
                    )
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    item = json.loads(line[5:])
                    identity = (item.get("ts"), item.get("line"))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    item["line"] = re.sub(
                        r"https?://[^\s\"'<>]+", "[url]", item.get("line", "")
                    ).replace(control.key, "[redacted]")
                    if (
                        item["line"].startswith("native_bootstrap_error=")
                        and not bootstrap_errors
                    ):
                        bootstrap_errors.append(item["line"])
                    with (control.output / "container-events.ndjson").open(
                        "a"
                    ) as target:
                        target.write(json.dumps(item) + "\n")
        except (OSError, ValueError, TimeoutError) as error:
            kind = type(error).__name__
            status = getattr(error, "code", None)
            if (kind, status) not in reported_errors:
                reported_errors.add((kind, status))
                control.event(
                    "provider_log_read_notice", error_type=kind, http_status=status
                )
        stopped.wait(1)


def validate_prepared(config, prepared):
    if not prepared.get("gpu_bake_completed"):
        raise ValueError("native driver cache bake must precede allocation")
    selected_config_sha = hashlib.sha256(
        (json.dumps(config["engine"], indent=2) + "\n").encode()
    ).hexdigest()
    if selected_config_sha != prepared["compatibility"]["native_config_sha256"]:
        raise ValueError("selected native configuration differs from baked runtime")
    for key, value in prepared["compatibility"].items():
        if prepared["cache_compatibility"].get(key) != value:
            raise ValueError("cache compatibility differs from selected native runtime")
    cache_manifest = json.loads(
        (Path(prepared["cache"]["directory"]) / "manifest.json").read_text()
    )
    labels = cache_manifest["metadata"]["compatibility"]
    if labels != prepared["cache_compatibility"]:
        raise ValueError("cache labels differ from published manifest")
    if labels.get("runtime_id") != prepared["runtime"]["id"]:
        raise ValueError("cache does not identify selected runtime")
    if labels.get("model_revision") != config["model_revision"]:
        raise ValueError("cache does not identify selected model revision")
    if labels.get("model_artifact_id") != prepared["model"]["id"]:
        raise ValueError("cache does not identify selected model artifact")


def cacheless_prepared(config, prepared, shared):
    """Bind the same native weights/configuration to an opportunistic wrapper."""
    config_sha = hashlib.sha256(
        (json.dumps(config["engine"], indent=2) + "\n").encode()
    ).hexdigest()
    if config_sha != prepared["compatibility"]["native_config_sha256"]:
        raise ValueError("selected native configuration differs from prepared model")
    if shared["compatibility"] != prepared.get("shared_runtime", {}).get(
        "compatibility"
    ) or any(
        prepared["compatibility"].get(key) != value
        for key, value in shared["compatibility"].items()
    ):
        raise ValueError("opportunistic runtime differs from prepared engine build")
    launch = shared["launch_configuration"]
    environment = launch["environment"]
    if (
        launch["artifact"]["id"] != shared["runtime"]["id"]
        or launch["program"] != "bin/launch-engine.sh"
        or environment.get("NATIVE_ENGINE_CACHE_POLICY") != "opportunistic"
        or environment.get("NATIVE_ENGINE_EXPECTED_GPU") != config["provider"]["gpu_id"]
        or environment.get("NATIVE_ENGINE_EXPECTED_COMPUTE_CAP")
        != shared["compatibility"]["gpu_compute_capability"]
    ):
        raise ValueError("opportunistic runtime launch identity is invalid")
    selected = copy.deepcopy(prepared)
    for key in [
        "cache",
        "cache_compatibility",
        "bake_receipt",
        "component_cache",
        "selected_driver",
    ]:
        selected.pop(key, None)
    selected.update(
        runtime=shared["runtime"],
        shared_runtime=launch,
        gpu_bake_completed=False,
        cache=None,
        cache_compatibility=None,
        cache_policy="opportunistic",
    )
    return selected


def driver_selector(by_driver, family=False, opportunistic=False):
    if opportunistic:
        return 'export SWITCHYARD_TICKET_URL="$NATIVE_CACHE_TICKET_URL_0"\n'
    selector = 'case "$native_driver" in\n'
    for index, driver in enumerate(by_driver):
        ticket_index = 0 if family else index
        selector += f'  {driver}) export SWITCHYARD_TICKET_URL="$NATIVE_CACHE_TICKET_URL_{ticket_index}";;\n'
    return selector + (
        '  *) printf "native_bootstrap_error=driver_mismatch\\n" >&2; exit 78;;\nesac\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache-alternative", type=Path, action="append", default=[])
    parser.add_argument("--family", type=Path)
    parser.add_argument("--shared-runtime", type=Path)
    parser.add_argument(
        "--cache-policy", choices=["strict", "opportunistic"], default="strict"
    )
    parser.add_argument("--live-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    prepared = json.loads(args.prepared.read_text())
    opportunistic = args.cache_policy == "opportunistic"
    if opportunistic:
        if not args.shared_runtime or args.family or args.cache_alternative:
            raise ValueError(
                "opportunistic direct launches require --shared-runtime and no baked seeds"
            )
        prepared = cacheless_prepared(
            config, prepared, json.loads(args.shared_runtime.read_text())
        )
    elif args.shared_runtime:
        raise ValueError(
            "--shared-runtime requires explicit opportunistic cache policy"
        )
    variants = [prepared] + [json.loads(p.read_text()) for p in args.cache_alternative]
    family = json.loads(args.family.read_text()) if args.family else None
    if family:
        if (
            not family["published"]
            or args.cache_alternative
            or "shared_runtime" not in prepared
        ):
            raise ValueError(
                "published families require the shared runtime and their exact components"
            )
        variants = []
        for driver, component in family["prepared_components"].items():
            variant = json.loads(Path(component).read_text())
            validate_prepared(config, variant)
            if variant["cache_compatibility"]["driver_version"] != driver:
                raise ValueError("family component driver mapping changed")
            variant["component_cache"] = variant["cache"]
            variant["cache"] = family["cache"]
            variant["cache_compatibility"] = family["compatibility"]
            variant["selected_driver"] = driver
            variants.append(variant)
        prepared = variants[0]
    by_driver = {}
    for variant in [] if opportunistic else variants:
        validate_prepared(config, variant)
        if (
            variant["runtime"]["id"] != prepared["runtime"]["id"]
            or variant["model"]["id"] != prepared["model"]["id"]
            or variant["compatibility"] != prepared["compatibility"]
        ):
            raise ValueError(
                "cache alternatives require identical runtime and model artifacts"
            )
        driver = (
            variant.get("selected_driver")
            or variant["cache_compatibility"]["driver_version"]
        )
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", driver) or driver in by_driver:
            raise ValueError("cache driver must be numeric and unique")
        by_driver[driver] = variant
    control = Control(config, args.output)
    token = secrets.token_hex(32)
    ticket_key = config["storage"]["prefix"] + "/tickets/" + uuid.uuid4().hex + ".json"
    legacy = json.loads(Path(prepared["legacy_manifest"]).read_text())
    live_model = json.loads(args.live_model.read_text()) if args.live_model else None
    if live_model:
        if (
            not live_model["published"]
            or live_model["model"] != config["engine"]["model"]
            or live_model["model_revision"] != config["model_revision"]
            or live_model["native_artifact_id"] != prepared["model"]["id"]
            or live_model["archive_sha256"] != legacy["sha256"]
        ):
            raise ValueError("live model archive differs from prepared native identity")
        with urllib.request.urlopen(
            control.url(live_model["archive_prefix"] + "/manifest.json"), timeout=30
        ) as source:
            if json.load(source) != legacy:
                raise ValueError(
                    "published live model manifest differs from prepared archive"
                )
    model_prefix = (
        live_model["archive_prefix"] if live_model else prepared["model"]["prefix"]
    )
    cache_labels = prepared["cache_compatibility"]
    runtime = (
        {}
        if opportunistic
        else {
            "artifact": ticket_artifact(control, prepared["runtime"]),
            "compatibility": prepared["compatibility"],
            "program": "bin/launch-engine.sh",
            "args": [],
            "hydration": {"retries": 0},
            "environment": {
                "LD_LIBRARY_PATH": "{runtime_root}/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
                "CUDA_CACHE_PATH": "{cache_root}/gpu-kernels/cuda",
                "CUDA_CACHE_DISABLE": "0",
                "CUDA_CACHE_MAXSIZE": "536870912",
                "CUDA_VISIBLE_DEVICES": "0",
                "NATIVE_ENGINE_EXPECTED_DRIVER": cache_labels.get("driver_version", ""),
                "NATIVE_ENGINE_EXPECTED_COMPUTE_CAP": cache_labels[
                    "gpu_compute_capability"
                ],
            },
            "caches": [
                {
                    "name": "gpu-kernels",
                    "artifact": ticket_artifact(control, prepared["cache"]),
                    "compatibility": cache_labels,
                }
            ],
        }
    )
    if family or opportunistic:
        runtime = {
            **copy.deepcopy(prepared["shared_runtime"]),
            "artifact": ticket_artifact(control, prepared["runtime"]),
            "caches": []
            if opportunistic
            else [
                {
                    "name": "gpu-kernels",
                    "artifact": ticket_artifact(control, prepared["cache"]),
                    "compatibility": cache_labels,
                }
            ],
        }
    ticket = {
        "model": config["engine"]["model"],
        "revision": config["model_revision"],
        "manifest": legacy,
        "urls": {
            c["name"]: control.url(model_prefix + "/" + c["name"])
            for c in legacy["chunks"]
        },
        "engine_args": [
            "--config",
            "/opt/switchyard/runtime/native-config.json",
            "--model-root",
            "/work/switchyard/model",
            "serve",
            "--bind",
            "127.0.0.1:8000",
        ],
        "runtime": runtime,
        "concurrency": config["limits"]["download_concurrency"],
        "stall_s": 120,
        "lifetime_s": config["limits"]["pod_timeout_s"],
        "startup": {
            "mode": "trusted_streaming",
            "model_root": "/work/switchyard",
            "require_model_ram": False,
            "early_launch": False,
        },
    }
    if family or opportunistic:
        ticket["engine_args"] = [
            "--config-json",
            json.dumps(config["engine"]),
            "serve",
            "--bind",
            "127.0.0.1:8000",
        ]
    control.put_json(ticket, ticket_key)
    ticket_keys = [ticket_key]
    selector_environment = {"NATIVE_CACHE_TICKET_URL_0": control.url(ticket_key)}
    for index, variant in enumerate([] if opportunistic else variants):
        driver = (
            variant.get("selected_driver")
            or variant["cache_compatibility"]["driver_version"]
        )
        if index and not family:
            variant_ticket = copy.deepcopy(ticket)
            variant_ticket["runtime"]["environment"][
                "NATIVE_ENGINE_EXPECTED_DRIVER"
            ] = driver
            variant_ticket["runtime"]["caches"][0] = {
                "name": "gpu-kernels",
                "artifact": ticket_artifact(control, variant["cache"]),
                "compatibility": variant["cache_compatibility"],
            }
            key = config["storage"]["prefix"] + "/tickets/" + uuid.uuid4().hex + ".json"
            ticket_keys.append(key)
            control.put_json(variant_ticket, key)
            selector_environment["NATIVE_CACHE_TICKET_URL_" + str(index)] = control.url(
                key
            )
    selector = driver_selector(by_driver, bool(family), opportunistic)
    memory_monitor = ""
    if config.get("measure_host_memory", False):
        memory_monitor = (
            Path(__file__).with_name("host-memory.sh").read_text()
            + "\nnative_host_memory_monitor >&2 &\n"
        )
    bootstrap = (
        r"""set -eu
printf '{"stage":"container_started","wall_s":%s}\n' "$(date +%s.%N)"
"""
        + memory_monitor
        + r"""
native_identity="$(nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total --format=csv,noheader -i 0)"
printf 'native_host_gpu=%s\n' "$native_identity"
IFS=',' read -r native_gpu native_uuid native_driver native_cap native_memory <<< "$native_identity"
native_driver="${native_driver//[[:space:]]/}"
"""
        + selector
        + r"""
printf 'native_selected_cache_driver=%s\n' "$native_driver"
mkdir -p /opt/switchyard
curl -fsSL --max-time 120 "$SWITCHYARD_AGENT_URL" -o /opt/switchyard/agent
chmod 700 /opt/switchyard/agent
printf '{"stage":"agent_downloaded","wall_s":%s}\n' "$(date +%s.%N)"
exec /opt/switchyard/agent
"""
    )
    stopped = threading.Event()
    logs = None
    bootstrap_errors = []
    allocation_began = None
    allocation_wall_s = None
    pod = None
    try:
        environment = {
            "SWITCHYARD_AGENT_URL": control.url(prepared["agent"]["key"]),
            "SWITCHYARD_AGENT_TOKEN": token,
            **selector_environment,
        }
        allocation_began = time.monotonic()
        allocation_wall_s = time.time()
        pod = control.allocate(
            "switchyard-native-launch-" + uuid.uuid4().hex[:12],
            ["/bin/bash", "-c"],
            [bootstrap],
            environment,
        )
        logs = threading.Thread(
            target=collect_logs,
            args=(control, pod["id"], stopped, bootstrap_errors),
            daemon=True,
        )
        logs.start()
        endpoint = "https://" + pod["id"] + "-8080.proxy.runpod.net"
        headers = {"Authorization": "Bearer " + token, "User-Agent": "Switchyard/0.5"}
        deadline = allocation_began + config["limits"]["pod_timeout_s"]
        first_agent = None
        while time.monotonic() < deadline:
            if bootstrap_errors:
                raise RuntimeError(bootstrap_errors[0])
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(endpoint + "/state", headers=headers),
                    timeout=4,
                ) as response:
                    state = json.load(response)
                if first_agent is None:
                    first_agent = time.monotonic() - allocation_began
                    control.event("agent_listener_observed", allocation_s=first_agent)
                write(args.output / "agent-state.json", state)
                with (args.output / "agent-states.ndjson").open("a") as target:
                    target.write(
                        json.dumps(
                            {
                                "allocation_s": time.monotonic() - allocation_began,
                                "state": state,
                            }
                        )
                        + "\n"
                    )
                if state["phase"] == "failed":
                    raise RuntimeError(
                        "native agent failed: " + str(state.get("error"))
                    )
                if state["phase"] == "ready":
                    break
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("fresh launch deadline exceeded")
        ready_s = time.monotonic() - allocation_began
        selected = None
        for record in (
            (args.output / "container-events.ndjson").read_text().splitlines()
        ):
            line = json.loads(record)["line"]
            if line.startswith("native_selected_cache_driver="):
                observed_driver = line.split("=", 1)[1]
                if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", observed_driver):
                    raise RuntimeError("observed driver telemetry is invalid")
                if opportunistic:
                    selected = copy.deepcopy(prepared)
                    selected["observed_driver"] = observed_driver
                    selected["cache_family_status"] = "miss"
                else:
                    selected = by_driver[observed_driver]
        if selected is None:
            raise RuntimeError(
                "selected cache identity is missing from startup telemetry"
            )
        prepared = selected
        cache_labels = prepared["cache_compatibility"]
        write(args.output / "selected-prepared.json", prepared)
        control.event("native_ready_observed", allocation_s=ready_s)
        body = {
            "model": config["engine"]["model"],
            "messages": [
                {"role": "user", "content": "Reply with exactly the word Paris."}
            ],
            "max_tokens": 4,
            "temperature": 0,
            "stream": True,
        }
        request_began = time.monotonic()
        first_content = None
        frames = []
        request = urllib.request.Request(
            endpoint + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={**headers, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                frame = json.loads(data)
                frames.append(frame)
                content = (
                    frame.get("choices", [{}])[0].get("delta", {}).get("content", "")
                )
                if content and first_content is None:
                    first_content = time.monotonic()
                    control.event(
                        "first_streamed_content",
                        allocation_s=first_content - allocation_began,
                        request_s=first_content - request_began,
                        content=content,
                    )
        if first_content is None:
            raise RuntimeError("native SSE returned no useful content")
        workloads = None
        if config.get("benchmark"):
            workloads = run_workloads(
                control,
                endpoint,
                headers,
                config["engine"]["model"],
                config["benchmark"],
            )
        final_state = json.load(
            urllib.request.urlopen(
                urllib.request.Request(endpoint + "/state", headers=headers), timeout=4
            )
        )
        write(args.output / "agent-state-final.json", final_state)
        write(
            args.output / "receipt.json",
            {
                "format": "switchyard-native-runpod-launch-v1",
                "source_commit": prepared["compatibility"]["engine_source_commit"],
                "engine_sha256": prepared["compatibility"]["engine_sha256"],
                "allocation_started_wall_s": allocation_wall_s,
                "accuracy": config["numerical_accuracy"],
                "pod": {
                    key: pod.get(key)
                    for key in ["id", "name", "cost", "dataCenterId", "cudaVersion"]
                },
                "runtime": prepared["runtime"],
                "model": prepared["model"],
                **(
                    {
                        "live_model_archive": {
                            key: live_model[key]
                            for key in [
                                "archive_prefix",
                                "archive_sha256",
                                "native_artifact_id",
                            ]
                        }
                    }
                    if live_model
                    else {}
                ),
                "cache": prepared["cache"],
                "cache_policy": args.cache_policy,
                "cache_family_status": "miss" if opportunistic else "hit",
                "observed_driver": prepared.get("observed_driver"),
                "numerical_parity_measured_in_this_serving_run": False,
                "compatibility": cache_labels,
                "bounds": config["engine"],
                "gpu_admission": config["limits"],
                "agent_listener_allocation_s": first_agent,
                "ready_allocation_s": ready_s,
                "first_content_allocation_s": first_content - allocation_began,
                "request_to_first_content_s": first_content - request_began,
                "request": body,
                "frames": frames,
                "workloads": workloads,
                "serving_python": False,
                "inference_readiness_probe": False,
                "model_transport": "bounded Rust archive streaming to published native files",
                "native_weight_loading": "native aggregate host buffer capped by max_weight_bytes; direct GPU chunk loading unimplemented",
            },
        )
        # Permit the log reader to receive final native cohort telemetry.
        stopped.wait(2)
    finally:
        stopped.set()
        if logs:
            logs.join(timeout=4)
        try:
            control.cleanup()
        finally:
            for key in ticket_keys:
                control.remove_object(key)
        if pod is not None:
            control.event(
                "rental_estimate",
                rental_duration_s=time.monotonic() - allocation_began,
                estimated_usd=(time.monotonic() - allocation_began)
                * pod["cost"]
                / 3600,
            )


if __name__ == "__main__":
    main()
