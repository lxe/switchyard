"""Launch one owned GPU container with the real Rust receiver and CUDA adapter."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

repo = Path(__file__).resolve().parents[3]
root = Path(os.environ["CHUNK_CUDA_OUT"]).resolve()
root.mkdir(parents=True)
boot = Path(os.environ["CHUNK_CUDA_BOOT"]).resolve()
program = Path(os.environ["CHUNK_CUDA_PROGRAM"]).resolve()
source = root / "source"
source.mkdir()
size = 64 * 1024 * 1024
header = json.dumps(
    {"large": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]}},
    separators=(",", ":"),
).encode()
header += b" " * ((-len(header)) % 8)
with (source / "model.safetensors").open("wb") as out:
    out.write(len(header).to_bytes(8, "little"))
    out.write(header)
    for block in range(size // (256 << 10)):
        pattern = bytes((i * 13 + 17 + block) % 256 for i in range(256)) * 1024
        out.write(pattern)
(source / "switchyard-final-state.json").write_text("{}\n")
subprocess.run(
    [
        str(program),
        "pack",
        str(source),
        str(root / "objects"),
        "--extent-bytes",
        str(256 << 10),
    ],
    check=True,
)
manifest = json.loads((root / "objects/manifest.json").read_text())
objects = {e["object"]: root / "objects" / e["object"] for e in manifest["extents"]}
first = manifest["extents"][1]["object"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        name = self.path.strip("/")
        path = objects.get(name)
        if path is None:
            self.send_error(404)
            return
        length = path.stat().st_size
        start, end = 0, length - 1
        ranged = "Range" in self.headers
        if ranged:
            start, end = map(
                int, self.headers["Range"].removeprefix("bytes=").split("-")
            )
        if name == first:
            time.sleep(0.07)
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Length", str(end - start + 1))
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
        self.end_headers()
        with path.open("rb") as data:
            data.seek(start)
            try:
                self.wfile.write(data.read(end - start + 1))
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when the cancellation test closes its receiver.

    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
(root / "ticket.json").write_text(
    json.dumps(
        {
            "manifest": manifest,
            "urls": {
                name: f"http://host.docker.internal:{server.server_port}/{name}"
                for name in objects
            },
        }
    )
)
(root / "buffers.json").write_text(
    json.dumps(
        {
            "chunk_bytes": 256 << 10,
            "ram_bytes": 1 << 20,
            "spool_bytes": 2 << 20,
            "download_concurrency": 6,
            "download_buffer_bytes": 64 << 10,
            "pinned_bytes": 2 << 20,
            "transfer_concurrency": 4,
        }
    )
)
name = "switchyard-chunk-proof-" + str(os.getpid())
libs = (
    [
        "/opt/switchyard/runtime/lib",
        "/opt/switchyard/runtime/python/lib",
        "/opt/switchyard/runtime/venv/lib/python3.12/site-packages/torch/lib",
    ]
    + [
        "/opt/switchyard/runtime/" + str(p.relative_to(boot))
        for p in sorted(
            (boot / "venv/lib/python3.12/site-packages/nvidia").glob("*/lib")
        )
    ]
    + ["/usr/local/nvidia/lib64", "/usr/local/nvidia/lib"]
)
command = [
    "docker",
    "run",
    "--name",
    name,
    "--gpus",
    "device=0",
    "--memory",
    "2g",
    "--memory-swap",
    "2g",
    "--add-host",
    "host.docker.internal:host-gateway",
    "-v",
    str(root) + ":/proof",
    "-v",
    str(boot) + ":/opt/switchyard/runtime:ro",
    "-v",
    str(program) + ":/stream:ro",
    "-v",
    str(repo / "crates/switchyard-weights/python") + ":/adapter:ro",
    "-v",
    str(Path(__file__).with_name("test_chunk_cuda.py")) + ":/test.py:ro",
    "-e",
    "PYTHONHOME=/opt/switchyard/runtime/python",
    "-e",
    "PYTHONPATH=/adapter:/opt/switchyard/runtime/venv/lib/python3.12/site-packages",
    "-e",
    "LD_LIBRARY_PATH=" + ":".join(libs),
    "-e",
    "PYTHONDONTWRITEBYTECODE=1",
    "--entrypoint",
    "/opt/switchyard/runtime/python/bin/python3.12",
    os.environ.get("CHUNK_CUDA_IMAGE", "switchyard-local-criu"),
    "/test.py",
]
try:
    subprocess.run(command, check=True)
    print("ACCEPTANCE_RECEIPT " + str(root / "receipt.json"), flush=True)
finally:
    server.shutdown()
    subprocess.run(
        ["docker", "rm", "--force", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
