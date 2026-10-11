"""Cold endpoint benchmark using the actual agent and artifact-v1 runtime.

Runs only new owned Docker containers. The runtime source and object directories
are read-only inputs; all installations, caches, logs and receipts belong to output.
No provider allocation or credential is needed.
"""

import argparse
import concurrent.futures
import hashlib
import http.server
import json
import os
import secrets
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from benchmark_artifacts import artifacts, named_directories, runtime_ticket
from timeline import Timeline

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--objects", type=Path, required=True)
parser.add_argument("--legacy-ticket", type=Path, required=True)
parser.add_argument("--runtime-objects", type=Path, required=True)
parser.add_argument(
    "--runtime-config",
    type=Path,
    help="Selected deployment or ServingRuntimeConfig JSON",
)
parser.add_argument(
    "--component-objects", action="append", default=[], metavar="NAME=PATH"
)
parser.add_argument("--cache-objects", action="append", default=[], metavar="NAME=PATH")
parser.add_argument(
    "--component-delay",
    type=float,
    default=0,
    help="Per 64KiB supplemental response delay",
)
parser.add_argument(
    "--early-launch", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--engine-args",
    type=Path,
    help="Exact engine args JSON; {model_root} expands to owned model path",
)
parser.add_argument("--agent", type=Path, required=True)
parser.add_argument("--image", default="ghcr.io/lxe/switchyard-runtime:ubuntu24.04")
parser.add_argument("--model-delay", type=float, default=0)
parser.add_argument("--mode", choices=["legacy", "extents", "both"], default="both")
parser.add_argument(
    "--warm-cache-from",
    type=Path,
    help="Entire installation/cache tree, preserving boot-ID/compiled layout",
)
parser.add_argument("--profile-startup", action="store_true")
parser.add_argument("--live-timeline", action="store_true")
parser.add_argument(
    "--skip-tokenize-check",
    action="store_true",
    help="Submit the measured streaming request directly after readiness",
)
parser.add_argument("--env-override-source", type=Path)
parser.add_argument("--cli-main-source", type=Path)
parser.add_argument(
    "--module-source", action="append", default=[], metavar="MODULE=PATH"
)
parser.add_argument("--verify-concurrency", type=int, default=0, metavar="REQUESTS")
parser.add_argument("--concurrency", type=int, default=8)
parser.add_argument(
    "--decode-benchmark", type=int, default=0, metavar="REQUESTS_PER_PHASE"
)
parser.add_argument("--runtime-env", action="append", default=[], metavar="KEY=VALUE")
args = parser.parse_args()
if args.verify_concurrency < 0 or args.decode_benchmark < 0 or args.concurrency < 1:
    parser.error("request count must be nonnegative and concurrency must be positive")
repo = Path(__file__).resolve().parents[3]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
objects = args.objects.resolve()
runtime_objects = args.runtime_objects.resolve()
manifest = json.loads((objects / "manifest.json").read_text())
runtime_manifest = json.loads((runtime_objects / "manifest.json").read_text())
legacy = json.loads(args.legacy_ticket.read_text())
configuration = json.loads(
    (args.runtime_config or repo / "configs/runpod/qwen-small.json").read_text()
)
runtime_configuration = configuration.get("bootstrap", {}).get("runtime", configuration)
if (
    args.runtime_config
    and runtime_configuration["artifact"]["id"] != runtime_manifest["id"]
):
    raise ValueError("selected boot artifact identity mismatch")
component_directories = named_directories(args.component_objects)
cache_directories = named_directories(args.cache_objects)
components = artifacts(
    runtime_configuration.get("components", []), component_directories, "component"
)
cache_configurations = (
    configuration.get("models", [{}])[0].get("caches", []) if cache_directories else []
)
caches = artifacts(cache_configurations, cache_directories, "cache")
object_directories = {
    "runtime": runtime_objects,
    "model": objects,
    **{f"components-{name}": path for name, path in component_directories.items()},
    **{f"caches-{name}": path for name, path in cache_directories.items()},
}
gateway = subprocess.check_output(
    [
        "docker",
        "network",
        "inspect",
        "bridge",
        "--format",
        "{{(index .IPAM.Config 0).Gateway}}",
    ],
    text=True,
).strip()
control_token = secrets.token_hex(32)
lock = threading.Lock()
events = []
current_ticket = None
ticket_requests = []
active_timeline = None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/ticket":
            with lock:
                ticket_requests.append(time.monotonic())
            data = json.dumps(current_ticket).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        parts = self.path.strip("/").split("/")
        if (
            len(parts) != 2
            or parts[0] not in object_directories
            or parts[1] in (".", "..")
        ):
            self.send_error(404)
            return
        path = object_directories[parts[0]] / parts[1]
        if not path.is_file():
            self.send_error(404)
            return
        began = time.monotonic()
        received = 0
        if active_timeline is not None:
            active_timeline.emit(
                "local_http_object_started",
                at=began,
                console=False,
                artifact=parts[0],
                object=parts[1],
                expected_bytes=path.stat().st_size,
            )
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as source:
                while block := source.read(65536):
                    if parts[0] == "model" and args.model_delay:
                        time.sleep(args.model_delay)
                    if parts[0].startswith("components-") and args.component_delay:
                        time.sleep(args.component_delay)
                    self.wfile.write(block)
                    received += len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            ended = time.monotonic()
            if active_timeline is not None:
                active_timeline.emit(
                    "local_http_object_finished",
                    at=ended,
                    console=False,
                    artifact=parts[0],
                    object=parts[1],
                    bytes=received,
                    duration_s=ended - began,
                )
            with lock:
                events.append(
                    {
                        "kind": parts[0],
                        "object": parts[1],
                        "started_monotonic": began,
                        "ended_monotonic": ended,
                        "bytes": received,
                    }
                )

    def log_message(self, *unused):
        pass


server = http.server.ThreadingHTTPServer((gateway, 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://host.docker.internal:{server.server_port}"
headers = {"Authorization": "Bearer " + control_token}
results = {}
try:
    for mode in ["legacy", "extents"] if args.mode == "both" else [args.mode]:
        root = out / mode
        root.mkdir()
        installation = root / "installation"
        installation.mkdir()
        source_overlay = None
        cli_overlay = None
        module_overlays = {}
        if (
            args.profile_startup
            or args.env_override_source
            or args.cli_main_source
            or args.module_source
        ):
            profile = installation / "startup-profile"
            profile.mkdir()
            # The agent uses umask 077 as container root. Preserve host ownership
            # so the live follower can read every span without privileged access.
            if args.profile_startup:
                (profile / "events.jsonl").touch(mode=0o600)
            (profile / "sitecustomize.py").write_bytes(
                (Path(__file__).parent / "startup_profile.py").read_bytes()
            )
            for assignment in args.module_source:
                module, filename = assignment.split("=", 1)
                target = (
                    "venv/lib/python3.12/site-packages/"
                    + module.replace(".", "/")
                    + ".py"
                )
                original = next(
                    entry
                    for entry in runtime_manifest["entries"]
                    if entry["path"] == target
                )
                source = Path(filename).read_bytes()
                staged = module.replace(".", "_") + ".py"
                (profile / staged).write_bytes(source)
                module_overlays[module] = {
                    "original_sha256": original["sha256"],
                    "staged_sha256": hashlib.sha256(source).hexdigest(),
                    "staged_path": "/opt/switchyard/startup-profile/" + staged,
                    "scope": "Private diagnostic source overlay; original runtime/cache identities are retained solely as controls, not published compatibility claims.",
                }
            if args.env_override_source:
                original = next(
                    entry
                    for entry in runtime_manifest["entries"]
                    if entry["path"]
                    == "venv/lib/python3.12/site-packages/vllm/env_override.py"
                )
                source = args.env_override_source.read_bytes()
                (profile / "env_override.py").write_bytes(source)
                source_overlay = {
                    "original_sha256": original["sha256"],
                    "staged_sha256": hashlib.sha256(source).hexdigest(),
                    "staged_path": "/opt/switchyard/startup-profile/env_override.py",
                    "scope": "Python module source staged and hashed before the request epoch. Original source hash comes from the pinned runtime manifest. This is a private diagnostic overlay; the original artifact ID and caches[] do not identify a published patched runtime or cache.",
                }
            if args.cli_main_source:
                original = next(
                    entry
                    for entry in runtime_manifest["entries"]
                    if entry["path"]
                    == "venv/lib/python3.12/site-packages/vllm/entrypoints/cli/main.py"
                )
                source = args.cli_main_source.read_bytes()
                (profile / "cli_main.py").write_bytes(source)
                cli_overlay = {
                    "original_sha256": original["sha256"],
                    "staged_sha256": hashlib.sha256(source).hexdigest(),
                    "staged_path": "/opt/switchyard/startup-profile/cli_main.py",
                    "scope": "Private diagnostic CLI module source staged and hashed before the request epoch. The agent, loader identity, runtime artifact identity and cache labels remain those of the control; this does not identify a selected or published engine adapter.",
                }
        if args.warm_cache_from:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "-v",
                    str(args.warm_cache_from.resolve()) + ":/seed:ro",
                    "--entrypoint",
                    "/bin/test",
                    args.image,
                    "-d",
                    "/seed/" + runtime_manifest["id"] + "/compiled",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    "switchyard-extent-cache-seed-" + secrets.token_hex(6),
                    "-v",
                    str(args.warm_cache_from.resolve()) + ":/seed:ro",
                    "-v",
                    str(installation) + ":/destination",
                    "--entrypoint",
                    "/bin/cp",
                    args.image,
                    "--archive",
                    "/seed/.",
                    "/destination/cache",
                ],
                check=True,
            )
        epoch = time.monotonic()
        timeline = (
            Timeline(root / "timeline.jsonl", epoch) if args.live_timeline else None
        )
        active_timeline = timeline
        stop_followers = threading.Event()
        profile_follower = None
        log_follower = None
        log_process = None
        if timeline:
            timeline.emit(
                "rollout_started",
                at=epoch,
                utc=datetime.now(timezone.utc).isoformat(),
                hydration_source="local_http",
                fresh_installation=True,
                native_profile_enabled=args.profile_startup,
            )
            if args.profile_startup:
                profile_follower = threading.Thread(
                    target=timeline.follow_native_file,
                    args=(
                        installation / "startup-profile/events.jsonl",
                        stop_followers,
                    ),
                    daemon=True,
                )
                profile_follower.start()
        ticket_before = len(ticket_requests)
        before = len(events)
        runtime = runtime_ticket(
            runtime_configuration, runtime_manifest, components, caches, base
        )
        if (
            args.profile_startup
            or args.env_override_source
            or args.cli_main_source
            or args.module_source
        ):
            runtime["environment"]["PYTHONPATH"] = (
                "/opt/switchyard/startup-profile:"
                + runtime["environment"].get("PYTHONPATH", "")
            )
            if args.profile_startup:
                runtime["environment"]["NATIVE_STARTUP_PROFILE"] = (
                    "/opt/switchyard/startup-profile/events.jsonl"
                )
            if args.env_override_source:
                runtime["environment"]["NATIVE_ENV_OVERRIDE_SOURCE"] = source_overlay[
                    "staged_path"
                ]
            if args.cli_main_source:
                runtime["environment"]["NATIVE_CLI_MAIN_SOURCE"] = cli_overlay[
                    "staged_path"
                ]
        if module_overlays:
            runtime["environment"]["NATIVE_MODULE_SOURCES"] = json.dumps(
                {
                    module: item["staged_path"]
                    for module, item in module_overlays.items()
                }
            )
        for assignment in args.runtime_env:
            key, value = assignment.split("=", 1)
            runtime["environment"][key] = value
        current_ticket = {
            "model": "qwen-local",
            "revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "runtime": runtime,
            "startup": {
                "mode": "trusted_streaming",
                "model_root": "/dev/shm/switchyard",
                "require_model_ram": True,
                "early_launch": args.early_launch,
            },
            "engine_args": [
                "serve",
                "/dev/shm/switchyard/model",
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
                "--enforce-eager",
                "--attention-backend",
                "TRITON_ATTN",
                "--seed",
                "123",
            ],
            "concurrency": 20,
            "stall_s": 120,
            "lifetime_s": 600,
        }
        if args.engine_args:
            current_ticket["engine_args"] = [
                argument.replace("{model_root}", "/dev/shm/switchyard/model")
                for argument in json.loads(args.engine_args.read_text())
            ]
        if mode == "legacy":
            current_ticket.update(
                {
                    "manifest": legacy["manifest"],
                    "urls": {
                        c["name"]: base + "/model/" + c["name"]
                        for c in legacy["manifest"]["chunks"]
                    },
                }
            )
        else:
            current_ticket["model_source"] = {
                "kind": "extents",
                "artifact": {
                    "manifest": manifest,
                    "urls": {
                        e["object"]: base + "/model/" + e["object"]
                        for e in manifest["extents"]
                    },
                },
                "loader": {"cuda_stage": True, "pinned_bytes": 1048576},
            }
        name = "switchyard-extent-agent-" + secrets.token_hex(6)
        env = os.environ.copy()
        env["SWITCHYARD_TICKET_URL"] = base + "/ticket"
        env["SWITCHYARD_AGENT_TOKEN"] = control_token
        command = [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--gpus",
            "device=0",
            "--shm-size",
            "8g",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-p",
            "127.0.0.1::8080",
            "-v",
            str(args.agent.resolve()) + ":/agent:ro",
            "-v",
            str(installation) + ":/opt/switchyard",
            "-e",
            "SWITCHYARD_TICKET_URL",
            "-e",
            "SWITCHYARD_AGENT_TOKEN",
            "--entrypoint",
            "/bin/sh",
            args.image,
            "-c",
            "umask 077; read task_uptime unused < /proc/uptime; printf '%s\\n' \"$task_uptime\" > /opt/switchyard/agent-exec-uptime; exec /agent",
        ]
        if timeline:
            timeline.emit("container_launch_started")
        subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)
        if timeline:
            timeline.emit("container_launch_returned")
            log_process = subprocess.Popen(
                ["docker", "logs", "--follow", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            def follow_engine_records(
                root=root, log_process=log_process, timeline=timeline
            ):
                with (root / "engine-startup.jsonl").open("w") as raw:
                    for line in log_process.stdout:
                        marker = "SWITCHYARD_ENGINE_STARTUP "
                        if marker in line:
                            item = json.loads(line.split(marker, 1)[1])
                            raw.write(json.dumps(item, separators=(",", ":")) + "\n")
                            raw.flush()
                            timeline.native(item)
                        elif (
                            "Uvicorn running" in line
                            or "Application startup complete" in line
                        ):
                            timeline.emit("native_api_ready_log_observed")

            log_follower = threading.Thread(target=follow_engine_records, daemon=True)
            log_follower.start()
        last = None
        try:
            # /proc start ticks are rounded down and use boot time. Capture the
            # host boot/monotonic offset instead of assuming uptime equals monotonic.
            pid = int(
                subprocess.check_output(
                    ["docker", "inspect", "--format", "{{.State.Pid}}", name], text=True
                )
            )
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            tick_seconds = 1 / os.sysconf("SC_CLK_TCK")
            boot_start = int(stat[19]) * tick_seconds
            monotonic_before = time.monotonic()
            boot_now = time.clock_gettime(time.CLOCK_BOOTTIME)
            monotonic_after = time.monotonic()
            process_start_bounds = [
                boot_start + monotonic_before - boot_now,
                boot_start + tick_seconds + monotonic_after - boot_now,
            ]
            pre_exec_boot = float(
                subprocess.check_output(
                    [
                        "docker",
                        "exec",
                        name,
                        "cat",
                        "/opt/switchyard/agent-exec-uptime",
                    ],
                    text=True,
                )
            )
            pre_exec_lower = pre_exec_boot + monotonic_before - boot_now
            if timeline:
                timeline.emit(
                    "agent_exec_clock",
                    at=pre_exec_lower,
                    lower_monotonic=pre_exec_lower,
                    rounding_bound_s=0.01,
                )
            port = (
                subprocess.check_output(["docker", "port", name, "8080/tcp"], text=True)
                .strip()
                .rsplit(":", 1)[1]
            )
            endpoint = "http://127.0.0.1:" + port
            while time.monotonic() - epoch < 590:
                try:
                    response = requests.get(
                        endpoint + "/state", headers=headers, timeout=5
                    )
                    if response.status_code == 200:
                        last = response.json()
                        (root / "last-state.json").write_text(json.dumps(last))
                        if timeline:
                            timeline.state(last, pre_exec_lower + 0.005)
                            loading = last.get("weight_loading") or {}
                            if (
                                loading.get("assignment_complete_seconds") is not None
                                and "weight_bytes_delivered" not in timeline.seen
                            ):
                                timeline.seen.add("weight_bytes_delivered")
                                timeline.emit(
                                    "weight_bytes_delivered",
                                    at=pre_exec_lower
                                    + 0.005
                                    + loading["assignment_complete_seconds"],
                                    tensors=loading["tensors_delivered"],
                                    bytes=loading["bytes_delivered"],
                                    first_tensor_agent_s=loading.get(
                                        "first_tensor_seconds"
                                    ),
                                    gpu=loading.get("gpu"),
                                )
                        if last["phase"] == "failed":
                            raise RuntimeError(
                                "agent failed: " + str(last.get("error"))
                            )
                        if last["phase"] == "ready":
                            break
                except (requests.ConnectionError, requests.Timeout):
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("agent readiness deadline")
            ready_at = time.monotonic() - epoch
            if timeline:
                timeline.emit("agent_ready_observed")
            tokenized = None
            if not args.skip_tokenize_check:
                tokenized = requests.post(
                    endpoint + "/tokenize",
                    headers=headers,
                    json={
                        "model": "qwen-local",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Reply with exactly the word Paris.",
                            }
                        ],
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=30,
                )
                tokenized.raise_for_status()
                assert tokenized.json()["count"] > 0
            first = None
            fragments = []
            request_started = time.monotonic()
            if timeline:
                timeline.emit("stream_request_submitted", at=request_started)
            with requests.post(
                endpoint + "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "qwen-local",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply with exactly the word Paris.",
                        }
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "max_tokens": 4,
                    "temperature": 0,
                    "stream": True,
                },
                stream=True,
                timeout=60,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                        event = json.loads(line[6:])
                        choices = event.get("choices", [])
                        text = (
                            choices[0].get("delta", {}).get("content", "")
                            if choices
                            else ""
                        )
                        if text:
                            fragments.append(text)
                            if first is None and text.strip():
                                first = time.monotonic() - epoch
                                if timeline:
                                    timeline.emit(
                                        "first_streamed_content",
                                        at=epoch + first,
                                        direct_request_ttft_s=epoch
                                        + first
                                        - request_started,
                                        content=text,
                                    )
            assert first is not None
            assert "".join(fragments).strip() == "Paris"
            if timeline:
                timeline.emit(
                    "stream_request_completed",
                    completion="Paris",
                    correctness_passed=True,
                )
            concurrent_verification = None
            if args.verify_concurrency:
                verification_started = time.monotonic()

                def verify_request(unused, endpoint=endpoint):
                    started = time.monotonic()
                    response = requests.post(
                        endpoint + "/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": "qwen-local",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Reply with exactly the word Paris.",
                                }
                            ],
                            "chat_template_kwargs": {"enable_thinking": False},
                            "max_tokens": 4,
                            "temperature": 0,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    completion = response.json()["choices"][0]["message"]["content"]
                    assert completion.strip() == "Paris"
                    return time.monotonic() - started

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.concurrency
                ) as executor:
                    latencies = list(
                        executor.map(verify_request, range(args.verify_concurrency))
                    )
                concurrent_verification = {
                    "accepted_requests": len(latencies),
                    "maximum_inflight": args.concurrency,
                    "all_completions_equal": "Paris",
                    "elapsed_seconds": time.monotonic() - verification_started,
                    "request_seconds": latencies,
                    "scope": "Bounded correctness burst after the first useful token; identical short prompts with prefix caching. This is not a throughput benchmark.",
                }
            decode_benchmark = None
            if args.decode_benchmark:

                def decode_request(index, endpoint=endpoint):
                    began = time.monotonic()
                    response = requests.post(
                        endpoint + "/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": "qwen-local",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Reply with exactly the word Paris.",
                                }
                            ],
                            "chat_template_kwargs": {"enable_thinking": False},
                            "max_tokens": 32,
                            "ignore_eos": True,
                            "temperature": 0,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()
                    assert data["usage"]["completion_tokens"] == 32
                    return {
                        "index": index,
                        "elapsed_seconds": time.monotonic() - began,
                        "output_tokens": data["usage"]["completion_tokens"],
                        "content_sha256": hashlib.sha256(
                            data["choices"][0]["message"]["content"].encode()
                        ).hexdigest(),
                    }

                decode_benchmark = {}
                for phase, workers in (("serial", 1), ("queued", args.concurrency)):
                    began = time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=workers
                    ) as executor:
                        decoded = list(
                            executor.map(decode_request, range(args.decode_benchmark))
                        )
                    elapsed = time.monotonic() - began
                    decode_benchmark[phase] = {
                        "requests": decoded,
                        "maximum_inflight": workers,
                        "elapsed_seconds": elapsed,
                        "output_tokens_per_second": sum(
                            r["output_tokens"] for r in decoded
                        )
                        / elapsed,
                        "scope": "Direct agent HTTP calls; fixed 32-token decode with EOS ignored, identical prompts, same small eager-engine fixture. Not a saturation calibration.",
                    }
            result = {
                "request_to_ready_seconds": ready_at,
                "request_to_first_useful_token_seconds": first,
                "direct_request_to_first_token_seconds": epoch
                + first
                - request_started,
                "completion": "".join(fragments),
                "tokenized_count": tokenized.json()["count"] if tokenized else None,
                "concurrent_verification": concurrent_verification,
                "decode_benchmark": decode_benchmark,
                "agent_state": last,
                "http": [
                    {
                        **e,
                        "started_seconds": e["started_monotonic"] - epoch,
                        "ended_seconds": e["ended_monotonic"] - epoch,
                    }
                    for e in events[before:]
                ],
                "scope": "Cold private container, actual agent ticket/preflights/runtime reception and extraction, tokenizer and native vLLM serving. Compiler caches initially empty. Correctness hashing was performed only in the separate loader comparison.",
                "compiler_cache_seeded": args.warm_cache_from is not None,
                "compiler_cache_artifact_hydrated": bool(caches),
                "cache_artifact_provenance": {
                    item["name"]: item["manifest"]["metadata"].get("attributes", {})
                    for item in caches
                },
                "request_epoch_monotonic": epoch,
                "agent_sha256": hashlib.sha256(args.agent.read_bytes()).hexdigest(),
                "runtime_artifact_id": runtime_manifest["id"],
                "component_artifact_ids": {
                    item["name"]: item["manifest"]["id"] for item in components
                },
                "cache_artifact_ids": {
                    item["name"]: item["manifest"]["id"] for item in caches
                },
                "early_launch": args.early_launch,
                "engine_args": current_ticket["engine_args"],
                "runtime_environment_overrides": args.runtime_env,
                "native_profile_enabled": args.profile_startup,
                "experimental_env_override_source": source_overlay,
                "experimental_cli_main_source": cli_overlay,
                "experimental_module_sources": module_overlays,
                "agent_process_start_monotonic_bounds": process_start_bounds,
                "agent_epoch_monotonic_bounds": [
                    pre_exec_lower,
                    ticket_requests[ticket_before],
                ],
                "agent_epoch_bounds_explanation": "The owned benchmark shell records Linux /proc/uptime immediately before exec of the unchanged agent binary. This value is rounded down to 10 ms. The Rust epoch is created after that lower bound and before its first ticket HTTP request. Boot-time conversion is bracketed by host monotonic samples. Native Python and host request times use the same Linux CLOCK_MONOTONIC. Kernel process creation is recorded separately because GPU container hooks run before exec.",
            }
            if args.warm_cache_from:
                result["scope"] = (
                    "Fresh private container and runtime/model reception; compiler cache copied from a prior matching local run before the request epoch. This tests upstream cache reuse, without claiming a published artifact-v1 cache bake. Correctness hashing was performed only in the separate loader comparison."
                )
            elif caches:
                result["scope"] = (
                    "Fresh owned GPU container and empty installation; boot, build, model and diagnostic native-cache artifacts hydrated from local HTTP. Native startup profiling is enabled. The cache records its original diagnostic provenance and is not qualified for a new production deployment identity. Memory-plan acceptance remains native."
                )
            for e in result["http"]:
                e.pop("started_monotonic")
                e.pop("ended_monotonic")
            results[mode] = result
            (root / "receipt.json").write_text(json.dumps(result, indent=2))
        except Exception as error:
            (root / "failure.json").write_text(
                json.dumps(
                    {
                        "elapsed_seconds": time.monotonic() - epoch,
                        "error": str(error),
                        "agent_state": last,
                    },
                    indent=2,
                )
            )
            raise
        finally:
            if args.profile_startup:
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        name + ":/opt/switchyard/startup-profile/events.jsonl",
                        str(root / "startup-profile.jsonl"),
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            with (root / "agent.log").open("w") as log:
                subprocess.run(
                    ["docker", "logs", name],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            subprocess.run(
                ["docker", "rm", "--force", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            stop_followers.set()
            if log_process is not None:
                try:
                    log_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    log_process.terminate()
                    log_process.wait(timeout=3)
            for follower in (profile_follower, log_follower):
                if follower is not None:
                    follower.join(timeout=3)
            if timeline:
                timeline.emit("owned_container_removed")
                active_timeline = None
                timeline.close()
    if args.mode == "both":
        assert results["legacy"]["completion"] == results["extents"]["completion"]
    (out / "receipt.json").write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                mode: {
                    k: r[k]
                    for k in (
                        "request_to_ready_seconds",
                        "request_to_first_useful_token_seconds",
                        "direct_request_to_first_token_seconds",
                        "completion",
                    )
                }
                for mode, r in results.items()
            },
            indent=2,
        )
    )
finally:
    server.shutdown()
    server.server_close()
