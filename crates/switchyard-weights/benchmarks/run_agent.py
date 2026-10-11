"""Cold endpoint benchmark using the actual agent and artifact-v1 runtime.

Runs only new owned Docker containers. The runtime source and object directories
are read-only inputs; all installations, caches, logs and receipts belong to output.
No provider allocation or credential is needed.
"""

import argparse
import http.server
import json
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--objects", type=Path, required=True)
parser.add_argument("--legacy-ticket", type=Path, required=True)
parser.add_argument("--runtime-objects", type=Path, required=True)
parser.add_argument("--agent", type=Path, required=True)
parser.add_argument("--image", default="ghcr.io/lxe/switchyard-runtime:ubuntu24.04")
parser.add_argument("--model-delay", type=float, default=0)
parser.add_argument("--mode", choices=["legacy", "extents", "both"], default="both")
parser.add_argument("--warm-cache-from", type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
objects = args.objects.resolve()
runtime_objects = args.runtime_objects.resolve()
manifest = json.loads((objects / "manifest.json").read_text())
runtime_manifest = json.loads((runtime_objects / "manifest.json").read_text())
legacy = json.loads(args.legacy_ticket.read_text())
runtime_configuration = json.loads(
    (repo / "configs/runpod/qwen-small.json").read_text()
)["bootstrap"]["runtime"]
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


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/ticket":
            data = json.dumps(current_ticket).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        parts = self.path.strip("/").split("/")
        if (
            len(parts) != 2
            or parts[0] not in ("runtime", "model")
            or parts[1] in (".", "..")
        ):
            self.send_error(404)
            return
        path = (runtime_objects if parts[0] == "runtime" else objects) / parts[1]
        if not path.is_file():
            self.send_error(404)
            return
        began = time.monotonic()
        received = 0
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as source:
                while block := source.read(65536):
                    if parts[0] == "model" and args.model_delay:
                        time.sleep(args.model_delay)
                    self.wfile.write(block)
                    received += len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with lock:
                events.append(
                    {
                        "kind": parts[0],
                        "object": parts[1],
                        "started_monotonic": began,
                        "ended_monotonic": time.monotonic(),
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
        if args.warm_cache_from:
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
        before = len(events)
        runtime = {
            k: runtime_configuration[k]
            for k in ("program", "args", "environment", "compatibility")
        }
        runtime.update(
            {
                "artifact": {
                    "manifest": runtime_manifest,
                    "urls": {
                        c["name"]: base + "/runtime/" + c["name"]
                        for c in runtime_manifest["chunks"]
                    },
                },
                "caches": [],
                "hydration": {"retries": 0},
            }
        )
        current_ticket = {
            "model": "qwen-local",
            "revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "runtime": runtime,
            "startup": {
                "mode": "trusted_streaming",
                "model_root": "/dev/shm/switchyard",
                "require_model_ram": True,
                "early_launch": True,
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
            "/agent",
            args.image,
        ]
        subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)
        last = None
        try:
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
            assert first is not None
            result = {
                "request_to_ready_seconds": ready_at,
                "request_to_first_useful_token_seconds": first,
                "completion": "".join(fragments),
                "tokenized_count": tokenized.json()["count"],
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
            }
            if args.warm_cache_from:
                result["scope"] = (
                    "Fresh private container and runtime/model reception; compiler cache copied from a prior matching local run before the request epoch. This tests upstream cache reuse, without claiming a published artifact-v1 cache bake. Correctness hashing was performed only in the separate loader comparison."
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
