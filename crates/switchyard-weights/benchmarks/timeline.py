"""Live, sanitized cold-start timeline on the host's monotonic clock.

Native spans overlap: durations are measurements, not additive phase totals.
JSONL retains every supplied native event; console imports use a 50 ms threshold.
No command arguments, environment values, request bodies, or URLs are emitted.
"""

import json
import threading
import time
from pathlib import Path


class Timeline:
    def __init__(self, output: Path, epoch: float):
        self.epoch = epoch
        self.output = output.open("w", encoding="utf-8")
        self.lock = threading.Lock()
        self.seen = set()
        self.artifact_states = {}

    def emit(self, event, *, at=None, console=True, **details):
        elapsed = (time.monotonic() if at is None else at) - self.epoch
        item = {"elapsed_s": elapsed, "event": event, **details}
        with self.lock:
            self.output.write(json.dumps(item, separators=(",", ":")) + "\n")
            self.output.flush()
            if console:
                fields = " ".join(f"{key}={value}" for key, value in details.items())
                print(f"[{elapsed:10.3f}s] {event} {fields}".rstrip(), flush=True)

    def state(self, state, agent_epoch):
        for milestone in state.get("startup_milestones", []):
            key = (milestone["event"], milestone["elapsed_s"])
            if key not in self.seen:
                self.seen.add(key)
                self.emit(key[0], at=agent_epoch + key[1], clock="agent_epoch_estimate")
        for artifact in state.get("artifacts", []):
            name = artifact["name"]
            reception = artifact.get("reception") or {}
            # Report actual completion durations; polling observation time is
            # explicitly different from the artifact's own elapsed duration.
            stage = (
                artifact.get("phase"),
                bool(reception.get("reception_complete_seconds")),
                bool(reception.get("extraction_complete_seconds")),
            )
            if self.artifact_states.get(name) != stage:
                self.artifact_states[name] = stage
                details = {"artifact": name, "phase": stage[0]}
                for field in (
                    "network_received_bytes",
                    "file_written_bytes",
                    "buffer_high_water_bytes",
                    "reception_complete_seconds",
                    "extraction_complete_seconds",
                ):
                    if field in reception:
                        details[field] = reception[field]
                self.emit("artifact_observed", **details)

    def native(self, item):
        kind = item.get("kind", item.get("event", "native_event"))
        at = item.get(
            "ended_monotonic",
            item.get(
                "started_monotonic", item.get("monotonic_s", item.get("monotonic"))
            ),
        )
        details = {
            key: item[key]
            for key in (
                "name",
                "module",
                "pid",
                "depth",
                "error_type",
                "duration_s",
                "worker_method",
                "cuda_initialized",
                "accepted",
                "requested",
                "selected",
                "applied",
                "explicit_memory_limit",
                "free_memory_bytes",
                "strategy",
                "cache_policy",
                "applicable",
                "mode",
                "available",
                "objects_preserved",
                "gpu_use",
                "profile_run_skipped",
                "profiling_forward_skipped",
                "automatic_memory_sizing",
                "kv_cache_memory_bytes",
                "runtime_headroom_bytes",
                "tensor_count",
                "tensor_bytes",
            )
            if key in item
        }
        duration = None
        if "ended_monotonic" in item and "started_monotonic" in item:
            duration = item["ended_monotonic"] - item["started_monotonic"]
            details["duration_s"] = round(duration, 6)
        self.emit(
            kind,
            at=at,
            console=kind != "import" or duration is None or duration >= 0.05,
            **details,
        )

    def follow_native_file(self, path, stop):
        offset = 0
        pending = ""
        while True:
            try:
                with path.open(encoding="utf-8") as source:
                    source.seek(offset)
                    pending += source.read()
                    offset = source.tell()
                lines = pending.split("\n")
                pending = lines.pop()
                for line in lines:
                    if line:
                        self.native(json.loads(line))
            except FileNotFoundError:
                pass
            if stop.wait(0.1):
                # One final drain includes events written before process exit.
                try:
                    with path.open(encoding="utf-8") as source:
                        source.seek(offset)
                        for line in (pending + source.read()).splitlines():
                            if line:
                                self.native(json.loads(line))
                except FileNotFoundError:
                    pass
                return

    def close(self):
        with self.lock:
            self.output.close()
