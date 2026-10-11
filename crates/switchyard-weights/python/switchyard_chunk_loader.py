"""Bounded CUDA-event pool copying artifact byte chunks into final parameters."""

import json
import os
import select
import socket
import time
import torch

DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def record(event, **values):
    print(
        "SWITCHYARD_WEIGHT_STREAM "
        + json.dumps(
            {
                "event": event,
                "monotonic_s": time.monotonic()
                + float(os.environ.get("VLLM_CPU_SNAPSHOT_CLOCK_OFFSET_S", "0")),
                "clock_uncertainty_s": float(
                    os.environ.get("VLLM_CPU_SNAPSHOT_CLOCK_UNCERTAINTY_S", "0")
                ),
                **values,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


class ChunkConsumer:
    def __init__(self, socket_path, destinations, *, timeout=120):
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(timeout)
        self.socket.connect(socket_path)
        # No subsequent frames exist until we request a chunk.
        hello = bytearray()
        while not hello.endswith(b"\n"):
            value = self.socket.recv(65536)
            if not value or len(hello) >= 16 * 1024 * 1024:
                raise RuntimeError("chunk hello bound")
            hello.extend(value)
        hello = json.loads(hello)
        if hello["format"] != "switchyard-weight-chunks-v1":
            raise ValueError("chunk protocol mismatch")
        manifest = hello["tensors"]
        limits = hello["limits"]
        if manifest["format"] != "switchyard-tensor-destinations-v1" or manifest[
            "worker"
        ] != {"rank": 0, "tensor_parallel_size": 1, "pipeline_parallel_size": 1}:
            raise ValueError("chunk tensor format/partition mismatch")
        expected = {item["name"]: item for item in manifest["tensors"]}
        if len(expected) != len(manifest["tensors"]) or set(expected) != set(
            destinations
        ):
            raise ValueError("chunk destination names differ")
        self.views = {}
        self.by_entry = {}
        self.remaining = {}
        self.seen = set()
        # Validate EVERY destination before issuing ANY copy; no flattening of
        # noncontiguous parameters, dtype conversion, or temporary GPU allocation.
        for name, tensor in destinations.items():
            item = expected[name]
            if (
                item["layout"] != "dense_c"
                or not tensor.is_contiguous()
                or tensor.device.type != "cuda"
                or list(tensor.shape) != item["shape"]
                or list(tensor.stride()) != item["stride"]
                or tensor.dtype != DTYPES[item["dtype"]]
                or tensor.numel() * tensor.element_size() != item["bytes"]
            ):
                raise ValueError("chunk destination layout differs: " + name)
            self.views[name] = tensor.detach().reshape(-1).view(torch.uint8)
            self.remaining[name] = item["bytes"]
            cursor = 0
            for mapping in item["ranges"]:
                if mapping["destination_offset"] != cursor or mapping["length"] <= 0:
                    raise ValueError("chunk mapping coverage")
                cursor += mapping["length"]
                self.by_entry.setdefault(mapping["entry"], []).append((name, mapping))
            if cursor != item["bytes"]:
                raise ValueError("chunk mapping size")
        self.chunk_bytes = limits["chunk_bytes"]
        count = limits["transfer_concurrency"]
        slot_bytes = limits["pinned_bytes"] // count
        if slot_bytes < self.chunk_bytes + 4096 or not 1 <= count <= 64:
            raise ValueError("chunk pinned pool limits")
        device = next(iter(destinations.values())).device
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        self.owners = [
            torch.empty(slot_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(count)
        ]
        self.buffers = [
            t.narrow(0, (-t.data_ptr()) % 4096, self.chunk_bytes) for t in self.owners
        ]
        self.free = list(range(count))
        self.active = {}
        self.wire = bytearray()
        self.pending = False
        self.ended = False
        self.timeout = timeout
        self.total_bytes = 0
        self.dma_seconds = 0.0
        self.stage_seconds = 0.0
        record(
            "consumer_available",
            pinned_bytes=count * slot_bytes,
            transfer_concurrency=count,
            direct_gpu_destinations=True,
        )

    def send(self, value):
        self.socket.sendall(json.dumps(value, separators=(",", ":")).encode() + b"\n")

    def poll_message(self):
        if b"\n" not in self.wire:
            if not select.select([self.socket], [], [], 0.001)[0]:
                return None
            value = self.socket.recv(65536)
            if not value:
                raise RuntimeError("chunk bridge disconnected")
            self.wire.extend(value)
        if len(self.wire) > 1024 * 1024:
            raise RuntimeError("chunk descriptor exceeds bound")
        if b"\n" not in self.wire:
            return None
        line, _, tail = self.wire.partition(b"\n")
        self.wire = bytearray(tail)
        return json.loads(line)

    def submit(self, message, slot):
        chunk = message["chunk"]
        ident = chunk["id"]
        if (
            ident in self.seen
            or not 0 < chunk["length"] <= chunk["padded_length"] <= self.chunk_bytes
            or chunk["padded_length"] % 4096
        ):
            raise ValueError("chunk bounds or duplicate")
        self.seen.add(ident)
        host = self.buffers[slot][: chunk["padded_length"]]
        began = time.monotonic()
        flags = os.O_RDONLY | os.O_CLOEXEC
        if message["location"] == "spool":
            flags |= os.O_DIRECT
        elif message["location"] != "ram":
            raise ValueError("unknown chunk backing")
        fd = os.open(message["path"], flags)
        try:
            view = memoryview(host.numpy())
            if os.preadv(fd, [view], 0) != chunk["padded_length"]:
                raise RuntimeError("chunk backing truncated")
            del view
        finally:
            os.close(fd)
        self.stage_seconds += time.monotonic() - began
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        copied = 0
        with torch.cuda.stream(self.stream):
            start.record(self.stream)
            for name, mapping in self.by_entry.get(chunk["entry"], []):
                first = max(chunk["offset"], mapping["source_offset"])
                last = min(
                    chunk["offset"] + chunk["length"],
                    mapping["source_offset"] + mapping["length"],
                )
                if first >= last:
                    continue
                length = last - first
                destination = (
                    mapping["destination_offset"] + first - mapping["source_offset"]
                )
                self.remaining[name] -= length
                if self.remaining[name] < 0:
                    raise ValueError("duplicate tensor byte coverage")
                self.views[name].narrow(0, destination, length).copy_(
                    host.narrow(0, first - chunk["offset"], length), non_blocking=True
                )
                copied += length
            end.record(self.stream)
        self.active[slot] = (ident, start, end, copied)
        record(
            "cuda_copy_submitted",
            chunk_id=ident,
            bytes=copied,
            location=message["location"],
        )

    def load(self):
        last_progress = time.monotonic()
        try:
            while not self.ended or self.active:
                for slot, (ident, start, end, count) in list(self.active.items()):
                    if not end.query():
                        continue
                    self.dma_seconds += start.elapsed_time(end) / 1000
                    self.total_bytes += count
                    record("cuda_copy_completed", chunk_id=ident, bytes=count)
                    # Host backing was closed after staging. The pinned slot is
                    # reused ONLY after this event; no device-wide synchronization.
                    self.send({"op": "ack", "id": ident, "cuda_complete": True})
                    del self.active[slot]
                    self.free.append(slot)
                    last_progress = time.monotonic()
                if self.free and not self.pending and not self.ended:
                    self.send({"op": "next"})
                    self.pending = True
                if self.pending:
                    message = self.poll_message()
                    if message is not None:
                        self.pending = False
                        last_progress = time.monotonic()
                        if message.get("done"):
                            self.ended = True
                        else:
                            self.submit(message, self.free.pop())
                else:
                    time.sleep(0.0001)
                if time.monotonic() - last_progress > self.timeout:
                    raise TimeoutError("chunk transfer progress deadline")
            if any(self.remaining.values()):
                raise ValueError("incomplete final tensor coverage")
            self.send(
                {
                    "op": "finish",
                    "gpu": {
                        "copies": len(self.seen),
                        "bytes": self.total_bytes,
                        "first_copy_finished_seconds": None,
                        "last_copy_finished_seconds": None,
                        "staging_seconds": self.stage_seconds,
                        "dma_seconds": self.dma_seconds,
                    },
                }
            )
            record(
                "chunk_load_complete",
                bytes=self.total_bytes,
                chunks=len(self.seen),
                gpu_dma_seconds=self.dma_seconds,
                staging_seconds=self.stage_seconds,
            )
        finally:
            for _, _, end, _ in self.active.values():
                end.synchronize()
            self.active.clear()
            self.socket.close()
