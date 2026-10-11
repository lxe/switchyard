"""Observe the owned container's cgroup, including file cache and dirty pages."""

import json
import threading
import time
from pathlib import Path


class ContainerMemory:
    def __init__(self, pid, output, timeline):
        relative = next(
            line.split(":", 2)[2]
            for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        self.root = Path("/sys/fs/cgroup") / relative.lstrip("/")
        self.output = output
        self.timeline = timeline
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.maximum = {}
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def sample(self):
        stat = dict(
            line.split()
            for line in (self.root / "memory.stat").read_text().splitlines()
        )
        values = {
            "current_bytes": int((self.root / "memory.current").read_text()),
            "peak_bytes": int((self.root / "memory.peak").read_text()),
            **{
                key + "_bytes": int(stat[key])
                for key in ("anon", "file", "file_dirty", "file_writeback", "shmem")
            },
        }
        with self.lock:
            for key, value in values.items():
                self.maximum[key] = max(self.maximum.get(key, 0), value)
        return values

    def run(self):
        with self.output.open("w") as output:
            try:
                while True:
                    at = time.monotonic()
                    values = self.sample()
                    output.write(json.dumps({"monotonic_s": at, **values}) + "\n")
                    output.flush()
                    if self.timeline:
                        self.timeline.emit("container_memory", at=at, **values)
                    if self.stop.wait(1):
                        return
            except OSError as error:
                self.error = type(error).__name__

    def summary(self):
        self.sample()
        with self.lock:
            return {
                "maximum_observed": dict(self.maximum),
                "sampling_error": self.error,
            }

    def close(self):
        self.stop.set()
        self.thread.join(2)
