"""Read-only root sampler for kernel wait channels of an owned container.

Unprivileged Linux reads hide wait channels for container-root processes. This
helper exits when its stop file appears or the selected container PID disappears.
"""

import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--stop", type=Path, required=True)
args = parser.parse_args()
relative = next(
    line.split(":", 2)[2]
    for line in Path(f"/proc/{args.pid}/cgroup").read_text().splitlines()
    if line.startswith("0::")
)
root = Path("/sys/fs/cgroup") / relative.lstrip("/")


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


with args.output.open("w") as output:
    while not args.stop.exists() and Path(f"/proc/{args.pid}").exists():
        at = time.monotonic()
        threads = []
        for pid in (read(root / "cgroup.procs") or "").split():
            try:
                tasks = list(Path(f"/proc/{pid}/task").iterdir())
            except OSError:
                continue
            for task in tasks:
                threads.append(
                    {
                        "pid": int(pid),
                        "tid": int(task.name),
                        "comm": read(task / "comm"),
                        "schedstat": read(task / "schedstat"),
                        "stat": read(task / "stat"),
                        "wchan": read(task / "wchan"),
                    }
                )
        output.write(
            json.dumps(
                {
                    "monotonic_s": at,
                    "sampler_cpu_seconds": time.process_time(),
                    "threads": threads,
                }
            )
            + "\n"
        )
        output.flush()
        time.sleep(0.25)
