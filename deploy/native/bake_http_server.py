"""Temporary authenticated HTTP control for offline native CUDA cache bakes."""

import base64
import hmac
import json
import os
import re
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

token = os.environ.pop("SWITCHYARD_BAKE_TOKEN")
settings = json.loads(os.environ.pop("SWITCHYARD_BAKE_SETTINGS"))
script = base64.b64decode(os.environ.pop("SWITCHYARD_BAKE_SCRIPT"))
root = Path(os.environ.pop("SWITCHYARD_BAKE_OUTPUT", "/work/native-model-bakes"))
root.mkdir(parents=True, exist_ok=True)
state = {"phase": "observing", "hardware": None, "error": None, "exit_code": None}
lock = threading.Lock()
start = threading.Event()
log = bytearray()


def drain(source):
    while line := source.readline(65536):
        text = re.sub(
            r"https?://[^\s\"'<>]+", "[url]", line.decode("utf8", errors="replace")
        )
        text = text.replace(token, "[redacted]")
        with lock:
            log.extend(text.encode())
            del log[:-65536]


def worker():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        identity = result.stdout.strip()
        gpu, _gpu_uuid, driver, cap, memory = [
            part.strip() for part in identity.split(",")
        ]
        with lock:
            state["hardware"] = identity
        print(
            json.dumps(
                {
                    "stage": "bake_hardware_observed",
                    "gpu": gpu,
                    "driver": driver,
                    "capability": cap,
                }
            ),
            flush=True,
        )
        if (
            gpu != settings["gpu"]
            or cap != settings["capability"]
            or not re.fullmatch(r"\d+\.\d+\.\d+", driver)
            or int(memory.split()[0]) * 1024 * 1024 < settings["minimum_gpu_bytes"]
        ):
            raise ValueError("hardware does not satisfy fixed native admission")
        if driver in settings["excluded_drivers"]:
            raise ValueError("observed driver is excluded by bake platform policy")
        if (
            settings.get("driver_target") is not None
            and driver != settings["driver_target"]
        ):
            raise ValueError("observed driver differs from explicit bake target")
        with lock:
            state["phase"] = "waiting"
        if not start.wait(timeout=500):
            raise ValueError("controller did not release the fixed bake gate")
        with lock:
            state["phase"] = "baking"
        process = subprocess.Popen(
            ["bash", "-s"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        reader = threading.Thread(target=drain, args=(process.stdout,), daemon=True)
        reader.start()
        try:
            process.stdin.write(script)
            process.stdin.close()
            exit_code = process.wait(timeout=540)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            reader.join(timeout=2)
            process.stdout.close()
        with lock:
            state["exit_code"] = exit_code
            state["phase"] = "complete" if exit_code == 0 else "failed"
            if exit_code:
                state["error"] = "native bake process failed"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        with lock:
            state["phase"] = "failed"
            state["error"] = str(error)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + token
        ):
            self.send_error(401)
            return
        if self.path == "/status":
            with lock:
                snapshot = dict(state)
            snapshot["completed"] = [
                name
                for name in settings["models"]
                if (root / name / "bake.tar.gz").is_file()
            ]
            data = json.dumps(snapshot).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/log":
            with lock:
                data = bytes(log)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        name = self.path.removeprefix("/bundle/")
        if not self.path.startswith("/bundle/") or name not in settings["models"]:
            self.send_error(404)
            return
        path = root / name / "bake.tar.gz"
        if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as source:
            while chunk := source.read(65536):
                self.wfile.write(chunk)

    def do_POST(self):
        if not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + token
        ):
            self.send_error(401)
            return
        if self.path != "/start":
            self.send_error(404)
            return
        if self.headers.get("Content-Length", "0") != "0":
            self.send_error(400)
            return
        with lock:
            phase = state["phase"]
        if phase not in ["waiting", "baking", "complete"]:
            self.send_error(409)
            return
        start.set()
        self.send_response(204)
        self.end_headers()


threading.Thread(target=worker, daemon=True).start()
ThreadingHTTPServer(
    (settings.get("bind", "0.0.0.0"), settings.get("port", 8080)), Handler
).serve_forever()
