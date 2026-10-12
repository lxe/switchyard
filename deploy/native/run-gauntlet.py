"""Observe the Rust live gauntlet and retain provider logs for its owned pods."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

from control import env_file, write


class LogControl:
    """Only the read-only interface needed by the provider log collector."""

    def __init__(self, fleet, key, output):
        self.config = {"provider": {"api_url": fleet["compute"]["api_url"]}}
        self.key = key
        self.output = output

    def event(self, stage, **fields):
        with (self.output / "log-observations.ndjson").open("a") as destination:
            destination.write(json.dumps({"stage": stage, **fields}) + "\n")


def credential(fleet, paths):
    name = fleet["compute"]["key_env"]
    if name in os.environ:
        return os.environ[name]
    # Match the Rust CLI's first-file-wins precedence.
    for path in paths:
        values = env_file(path)
        if name in values:
            return values[name]
    raise ValueError("Configured provider credential is absent")


def observation_phases(path):
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - 131072))
        lines = source.read().splitlines()
    phases = {}
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        state = event.get("state") or {}
        phases[event.get("pod")] = state.get("phase", event.get("status"))
    return phases


def emit_log(path, offset, key):
    with path.open(errors="replace") as source:
        source.seek(offset)
        for line in source.read().splitlines():
            safe = re.sub(r"https?://[^\s]+", "[url]", line).replace(key, "[redacted]")
            if safe:
                print(safe, flush=True)
        return source.tell()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    fleet = json.loads(args.config.read_text())
    key = credential(fleet, args.env_file)
    if not key:
        raise ValueError("Configured provider credential is empty")
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        "native_launch", Path(__file__).with_name("run-launch.py")
    )
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    command = [str(args.binary.resolve()), "live", "gauntlet", "--config",
               str(args.config.resolve()), "--workloads", str(args.workloads.resolve()),
               "--out", str(args.out.resolve())]
    for path in args.env_file:
        command.extend(["--env-file", str(path.resolve())])
    if args.model:
        command.extend(["--model", args.model])
    source = {
        "cli_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "fleet_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "workloads_sha256": hashlib.sha256(args.workloads.read_bytes()).hexdigest(),
        "selected_model": args.model,
        "provider_log_api": fleet["compute"]["api_url"],
    }
    write(args.out / "observer-source.json", source)
    stopped = threading.Event()
    workers, seen = {}, {}
    offset = 0
    started = time.monotonic()
    log_path = args.out / "driver.log"
    print("phase=gauntlet_started", flush=True)
    with log_path.open("w") as log:
        # The tmux foreground receives Ctrl-C; forward it once to the Rust
        # process so its ownership-safe cleanup remains in control.
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                offset = emit_log(log_path, offset, key)
                for path in args.out.glob("*/agent-observations.ndjson"):
                    for pod, phase in observation_phases(path).items():
                        identity = (path.parent.name, pod)
                        if seen.get(identity) != phase:
                            seen[identity] = phase
                            print(f"elapsed_s={time.monotonic()-started:.1f} model={identity[0]} pod={pod} phase={phase}", flush=True)
                for path in args.out.glob("*/owned-pods.json"):
                    try:
                        owners = json.loads(path.read_text())
                    except (OSError, ValueError):
                        continue
                    for owner in owners:
                        pod = owner["pod"]["id"]
                        if pod in workers:
                            continue
                        directory = path.parent / "provider-logs" / pod
                        directory.mkdir(parents=True)
                        worker = threading.Thread(target=launch.collect_logs,
                            args=(LogControl(fleet, key, directory), pod, stopped, []), daemon=True)
                        workers[pod] = worker
                        worker.start()
                time.sleep(1)
            emit_log(log_path, offset, key)
        except BaseException:
            if process.poll() is None:
                print("phase=waiting_for_owned_cleanup", flush=True)
                process.send_signal(signal.SIGINT)
                process.wait()
            raise
        finally:
            stopped.set()
            for worker in workers.values():
                worker.join(timeout=4)
            write(args.out / "observer-outcome.json", {
                "returncode": process.returncode,
                "elapsed_s": time.monotonic() - started,
                "observed_owned_pods": sorted(workers),
                "cleanup_authority": "Rust owned-pods and pending-launches journals; inspect per-model outcome.json",
            })
    print(f"phase=gauntlet_exited status={process.returncode} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
