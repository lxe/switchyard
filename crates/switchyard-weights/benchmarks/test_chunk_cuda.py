"""Owned-container test of bounded final-destination copies; no readiness inference."""

import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path("/proof")
stop = threading.Event()
samples = []


def memory():
    root = Path("/sys/fs/cgroup")
    with (ROOT / "container-memory.jsonl").open("w") as out:
        while not stop.is_set():
            stat = dict(
                line.split() for line in (root / "memory.stat").read_text().splitlines()
            )
            value = {
                "monotonic_s": time.monotonic(),
                "current_bytes": int((root / "memory.current").read_text()),
                "peak_bytes": int((root / "memory.peak").read_text()),
                **{
                    k: int(stat[k])
                    for k in ("anon", "file", "file_dirty", "file_writeback", "shmem")
                },
            }
            samples.append(value)
            out.write(json.dumps(value) + "\n")
            out.flush()
            stop.wait(0.02)


thread = threading.Thread(target=memory, daemon=True)
thread.start()


def producer(case):
    output = ROOT / case
    output.mkdir()
    log = (output / "receiver.jsonl").open("w")
    process = subprocess.Popen(
        [
            "/stream",
            "stream",
            str(ROOT / "ticket.json"),
            str(output / "state"),
            str(output / "weights.sock"),
            "--buffers",
            str(ROOT / "buffers.json"),
            "--timeout-seconds",
            "30",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 10
    while not (output / "weights.sock").exists():
        if process.poll() is not None:
            raise RuntimeError(
                "producer exited: " + (output / "receiver.jsonl").read_text()[-1000:]
            )
        if time.monotonic() > deadline:
            raise TimeoutError("producer socket")
        time.sleep(0.01)
    return output, process, log


out, process, log = producer("success")
# Receiver starts before Torch import and CUDA construction, retaining early data
# within the RAM/spool limits. The server deliberately completes ranges out of order.
time.sleep(0.5)
import torch
from switchyard_chunk_loader import ChunkConsumer

size = 64 * 1024 * 1024


class SlowConsumer(ChunkConsumer):
    peak_inflight = 0

    def submit(self, message, slot):
        # Test-only GPU delay keeps several DMA completion events outstanding.
        with torch.cuda.stream(self.stream):
            torch.cuda._sleep(20_000_000)
        super().submit(message, slot)
        self.peak_inflight = max(self.peak_inflight, len(self.active))


results = {}
try:
    destination = torch.empty(size, dtype=torch.uint8, device="cuda:0")
    pointer = destination.data_ptr()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    consumer = SlowConsumer(
        str(out / "weights.sock"), {"large": destination}, timeout=30
    )
    consumer.load()
    process.wait(timeout=30)
    log.close()
    if process.returncode:
        raise RuntimeError(
            "producer failed: " + (out / "receiver.jsonl").read_text()[-2000:]
        )
    assert destination.data_ptr() == pointer
    assert torch.cuda.max_memory_allocated() == baseline, (
        "temporary GPU allocation detected"
    )
    assert consumer.peak_inflight > 1, (
        "multiple outstanding CUDA transfers not exercised"
    )
    pattern = bytes((i * 13 + 17) % 256 for i in range(256))
    expected = torch.frombuffer(bytearray(pattern * 1024), dtype=torch.uint8)
    for offset in range(0, size, len(expected)):
        actual = destination.narrow(0, offset, len(expected)).cpu()
        assert torch.equal(actual, expected), f"wrong final bytes at {offset}"
    events = []
    for line in (out / "receiver.jsonl").read_text().splitlines():
        if line.startswith("SWITCHYARD_WEIGHT_STREAM "):
            events.append(json.loads(line.split(" ", 1)[1])["progress"])
    stats = [e["progress"] for e in events]
    limits = json.loads((ROOT / "buffers.json").read_text())
    assert max(s["ram_bytes"] for s in stats) <= limits["ram_bytes"]
    assert max(s["spool_bytes"] for s in stats) <= limits["spool_bytes"]
    assert max(s["spool_bytes"] for s in stats) == limits["spool_bytes"]
    assert stats[-1]["ram_bytes"] == stats[-1]["spool_bytes"] == 0
    lifecycle = [e for e in events if e["lifecycle"]]
    by_id = {}
    for e in lifecycle:
        by_id.setdefault(e["chunk_id"], []).append(e["lifecycle"])
    assert all(
        states
        == ["reserved", "downloading", "ready", "copying", "consumed", "reclaimed"]
        for states in by_id.values()
    ), "invalid chunk lifecycle"
    assert not list((out / "state").parent.glob(".weight-buffer-*"))
    results["success"] = {
        "bytes": size,
        "ram_peak_bytes": max(s["ram_bytes"] for s in stats),
        "spool_peak_bytes": max(s["spool_bytes"] for s in stats),
        "gpu_temporary_bytes": torch.cuda.max_memory_allocated() - baseline,
        "peak_transfers": consumer.peak_inflight,
        "gpu_dma_seconds": consumer.dma_seconds,
        "staging_seconds": consumer.stage_seconds,
        "reclaimed_bytes": stats[-1]["reclaimed_bytes"],
        "network_wait_seconds": stats[-1]["network_wait_seconds"],
        "spool_write_seconds": stats[-1]["spool_write_seconds"],
        "backpressure_seconds": stats[-1]["exhausted_buffer_seconds"],
    }
    del consumer, destination

    # A partial consumer cancellation must settle its outstanding events, close
    # control, and dispose only privately owned backing. No shared artifact changes.
    out, process, log = producer("cancel")

    class CancelConsumer(SlowConsumer):
        def submit(self, message, slot):
            super().submit(message, slot)
            if len(self.seen) == 12:
                raise RuntimeError("intentional consumer cancellation")

    destination = torch.empty(size, dtype=torch.uint8, device="cuda:0")
    consumer = CancelConsumer(
        str(out / "weights.sock"), {"large": destination}, timeout=30
    )
    try:
        consumer.load()
        raise AssertionError("cancellation did not run")
    except RuntimeError as error:
        assert str(error) == "intentional consumer cancellation"
    process.wait(timeout=30)
    log.close()
    assert process.returncode != 0
    assert not list((out / "state").parent.glob(".weight-buffer-*"))
    results["cancel"] = {
        "outstanding_events_settled": not consumer.active,
        "private_spool_removed": True,
    }
    del consumer, destination

    # A noncontiguous parameter must be rejected before the first CUDA copy.
    out, process, log = producer("noncontiguous")
    storage = torch.empty(size * 2, dtype=torch.uint8, device="cuda:0")
    destination = storage[::2]
    try:
        ChunkConsumer(str(out / "weights.sock"), {"large": destination})
        raise AssertionError("noncontiguous destination accepted")
    except ValueError as error:
        assert "layout differs" in str(error)
    process.wait(timeout=30)
    log.close()
    assert process.returncode != 0
    results["noncontiguous"] = {"rejected_before_copy": True}
finally:
    if process.poll() is None:
        process.kill()
        process.wait()
    stop.set()
    thread.join()
results["container_memory"] = {
    key: max(s[key] for s in samples)
    for key in (
        "current_bytes",
        "peak_bytes",
        "anon",
        "file",
        "file_dirty",
        "file_writeback",
        "shmem",
    )
}
(ROOT / "receipt.json").write_text(json.dumps(results, indent=2) + "\n")
print("CHUNK_CUDA_ACCEPTANCE " + json.dumps(results), flush=True)
