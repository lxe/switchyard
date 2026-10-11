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
        self.pid = pid
        self.sample_index = 0
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

    def resources(self):
        def read(path):
            try:
                return Path(path).read_text()
            except OSError:
                return None

        values = {}
        for name in (
            "cpu.stat",
            "cpu.pressure",
            "memory.pressure",
            "io.pressure",
            "io.stat",
            "memory.events",
            "cpu.max",
            "memory.max",
            "memory.swap.max",
            "io.max",
        ):
            values[name] = read(self.root / name)
        values["host_diskstats"] = "\n".join(
            line
            for line in (read("/proc/diskstats") or "").splitlines()
            if "nvme" in line or " dm-" in line
        )
        values["host_writeback"] = "\n".join(
            line
            for line in (read("/proc/meminfo") or "").splitlines()
            if line.startswith(("Dirty:", "Writeback:"))
        )
        threads = []
        processes = []
        for pid in (read(self.root / "cgroup.procs") or str(self.pid)).split():
            processes.append({"pid": int(pid), "io": read(f"/proc/{pid}/io")})
            try:
                tasks = list(Path(f"/proc/{pid}/task").iterdir())
            except OSError:
                continue
            for task in tasks:
                threads.append(
                    {
                        "pid": int(pid),
                        "tid": int(task.name),
                        "stat": read(task / "stat"),
                        "schedstat": read(task / "schedstat"),
                        "wchan": read(task / "wchan"),
                    }
                )
        values["threads"] = threads
        values["processes"] = processes
        return values

    def run(self):
        with self.output.open("w") as output:
            try:
                while True:
                    at = time.monotonic()
                    values = self.sample()
                    output.write(
                        json.dumps(
                            {"monotonic_s": at, **values, "resources": self.resources()}
                        )
                        + "\n"
                    )
                    output.flush()
                    if self.timeline:
                        self.timeline.emit(
                            "container_memory",
                            at=at,
                            console=self.sample_index % 5 == 0,
                            **values,
                        )
                    self.sample_index += 1
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
