"""Reproducible throttled-loopback hydration + pinned-vLLM GPU benchmark.

Requires the caller-selected Python environment with torch/transformers/vLLM.
Nothing is installed; all artifacts, sockets, plugin metadata and logs stay in output.
"""

import argparse
import hashlib
import http.server
import io
import json
import os
import signal
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--chunk-delay", type=float, default=0.004)
parser.add_argument("--piece-bytes", type=int, default=1024 * 1024)
parser.add_argument("--model-source", type=Path)
parser.add_argument("--hidden-size", type=int, default=512)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
source = out / "source"
source.mkdir(exist_ok=True)
if args.model_source:
    source = args.model_source.resolve()
else:
    torch.manual_seed(123)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 4,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).to(torch.bfloat16)
    model.save_pretrained(source, safe_serialization=True)
    del model
objects = out / "objects"
objects.mkdir(exist_ok=True)
binary = repo / "target/debug/switchyard-weights"
legacy = repo / "target/debug/examples/legacy_hydrate"
manifest_path = Path(
    subprocess.check_output(
        [
            str(binary),
            "pack",
            str(source),
            str(objects),
            "--extent-bytes",
            str(args.piece_bytes),
        ],
        text=True,
    ).strip()
)
manifest = json.loads(manifest_path.read_text())
# Construct the exact current packaging: one TAR per source file, sequential raw pieces.
chunks = []
files = {}
for shard, path in enumerate(sorted(p for p in source.rglob("*") if p.is_file())):
    name = str(path.relative_to(source))
    data = path.read_bytes()
    files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    for seq, offset in enumerate(range(0, len(data), args.piece_bytes)):
        piece = data[offset : offset + args.piece_bytes]
        key = f"legacy-{shard:03}-{seq:03}"
        (objects / key).write_bytes(piece)
        chunks.append(
            {
                "name": key,
                "shard": shard,
                "seq": seq,
                "size": len(piece),
                "raw": len(piece),
                "sha256": hashlib.sha256(piece).hexdigest(),
            }
        )
legacy_manifest = {
    "format": "shards-v1",
    "compress": "none",
    "chunks": chunks,
    "files": files,
    "total_size": sum(c["size"] for c in chunks),
    "raw_size": sum(c["raw"] for c in chunks),
    "sha256": hashlib.sha256("".join(c["sha256"] for c in chunks).encode()).hexdigest(),
}
lock = threading.Lock()
transfer_events = []
active = 0
peak = 0


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        global active, peak
        key = self.path.lstrip("/")
        path = objects / key
        if path.parent != objects or not path.is_file():
            self.send_error(404)
            return
        with lock:
            active += 1
            peak = max(peak, active)
        started = time.perf_counter()
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as f:
                while True:
                    data = f.read(64 * 1024)
                    if not data:
                        break
                    time.sleep(args.chunk_delay)
                    self.wfile.write(data)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with lock:
                active -= 1
                transfer_events.append(
                    {"object": key, "start": started, "end": time.perf_counter()}
                )

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}"
urls = {e["object"]: base + "/" + e["object"] for e in manifest["extents"]}
(out / "extent-ticket.json").write_text(
    json.dumps({"manifest": manifest, "urls": urls})
)
legacy_ticket = {
    "model": "synthetic-llama",
    "revision": "seed-123",
    "manifest": legacy_manifest,
    "urls": {c["name"]: base + "/" + c["name"] for c in chunks},
    "engine_args": [],
    "concurrency": 20,
    "stall_s": 120,
    "lifetime_s": 300,
    "startup": {"mode": "trusted_streaming", "require_model_ram": True},
}
(out / "legacy-ticket.json").write_text(json.dumps(legacy_ticket))
# A private discovery entrypoint is visible to vLLM subprocesses through PYTHONPATH.
plugin = out / "plugins"
dist = plugin / "switchyard_benchmark-0.0.0.dist-info"
dist.mkdir(parents=True, exist_ok=True)
(dist / "METADATA").write_text(
    "Metadata-Version: 2.1\nName: switchyard-benchmark\nVersion: 0.0.0\n"
)
(dist / "entry_points.txt").write_text(
    "[vllm.general_plugins]\nswitchyard_benchmark = loader_plugin:register\n"
)
metadata = out / "metadata"
metadata.mkdir(exist_ok=True)
for p in source.iterdir():
    if p.is_file() and not p.name.endswith(".safetensors"):
        (metadata / p.name).write_bytes(p.read_bytes())
results = {}
processes = []
try:
    for mode in ["legacy", "extents"]:
        socket = out / f"{mode}.sock"
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(plugin),
                        str(Path(__file__).parent),
                        str(repo / "crates/switchyard-weights/python"),
                    ]
                ),
                "VLLM_PLUGINS": "switchyard_benchmark",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "SWITCHYARD_BENCH_RECEIPT": str(out / f"{mode}-weights.json"),
                "SWITCHYARD_BENCH_SOCKET": str(socket),
                "SWITCHYARD_BENCH_LEGACY": str(legacy),
                "SWITCHYARD_BENCH_TICKET": str(out / "legacy-ticket.json"),
                "SWITCHYARD_BENCH_ROOT": str(out / f"{mode}-root"),
            }
        )
        Path(env["SWITCHYARD_BENCH_ROOT"]).mkdir(exist_ok=True)
        epoch = time.perf_counter()
        before = len(transfer_events)
        peak = 0
        bridge = None
        if mode == "extents":
            bridge_log = (out / "bridge.jsonl").open("w")
            bridge = subprocess.Popen(
                [
                    str(binary),
                    "serve",
                    str(out / "extent-ticket.json"),
                    str(out / "extents-root/model"),
                    str(socket),
                    "--trusted",
                    "--wait-for-consumer",
                    "--timeout-seconds",
                    "180",
                ],
                stderr=bridge_log,
                start_new_session=True,
            )
            processes.append(bridge)
            while not socket.exists():
                if bridge.poll() is not None:
                    raise RuntimeError("bridge failed")
                time.sleep(0.01)
        with (out / f"{mode}-vllm.log").open("w") as log:
            worker = subprocess.Popen(
                [
                    os.sys.executable,
                    str(Path(__file__).parent / "vllm_worker.py"),
                    mode,
                    str(metadata),
                    str(out / f"{mode}-inference.json"),
                ],
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            processes.append(worker)
            if worker.wait(timeout=240) != 0:
                raise RuntimeError(f"{mode} vLLM failed: see {out}/{mode}-vllm.log")
        if bridge:
            if bridge.wait(timeout=20) != 0:
                raise RuntimeError("bridge failed: see bridge.jsonl")
            bridge_log.close()
        weight_receipt = json.loads((out / f"{mode}-weights.json").read_text())
        for tensor in weight_receipt.get("tensors", []):
            tensor["copy_started_seconds"] = (
                tensor.pop("copy_started_monotonic") - epoch
            )
            tensor["copy_finished_seconds"] = (
                tensor.pop("copy_finished_monotonic") - epoch
            )
        results[mode] = {
            "weights": weight_receipt,
            "inference": json.loads((out / f"{mode}-inference.json").read_text()),
            "wall_seconds": time.perf_counter() - epoch,
            "peak_http_requests": peak,
            "http": [
                {**e, "start": e["start"] - epoch, "end": e["end"] - epoch}
                for e in transfer_events[before:]
            ],
        }
    assert (
        results["legacy"]["weights"]["parameters_sha256"]
        == results["extents"]["weights"]["parameters_sha256"]
    )
    assert (
        results["legacy"]["inference"]["tokens"]
        == results["extents"]["inference"]["tokens"]
    )
    for name, info in files.items():
        for mode in ["legacy", "extents"]:
            assert (
                hashlib.sha256(
                    (out / f"{mode}-root/model" / name).read_bytes()
                ).hexdigest()
                == info["sha256"]
            )
    results["fixture"] = {
        "file_bytes": sum(f["size"] for f in files.values()),
        "extent_bytes": args.piece_bytes,
        "chunk_delay_seconds": args.chunk_delay,
        "torch": torch.__version__,
    }
    results["bridge"] = [
        json.loads(line)
        for line in (out / "bridge.jsonl").read_text().splitlines()
        if line.startswith("{")
    ]
    end = max(e["end"] for e in results["extents"]["http"])
    results["early_gpu_tensors"] = sum(
        t["copy_finished_seconds"] < end
        for t in results["extents"]["weights"]["tensors"]
    )
    assert results["early_gpu_tensors"] > 0, (
        "No GPU copy finished before the network transfers"
    )
    (out / "receipt.json").write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                "receipt": str(out / "receipt.json"),
                "early_gpu_tensors": results["early_gpu_tensors"],
                **{
                    mode: {
                        "weight_seconds": results[mode]["weights"]["weight_seconds"],
                        "engine_ready_seconds": results[mode]["inference"][
                            "engine_ready_seconds"
                        ],
                        "tokens": results[mode]["inference"]["tokens"],
                    }
                    for mode in ["legacy", "extents"]
                },
            },
            indent=2,
        )
    )
finally:
    for p in processes:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    server.shutdown()
