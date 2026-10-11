"""Bound one CLI experiment, collect sanitized telemetry, and confirm exact-owner cleanup."""
import argparse
import datetime
import json
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def credential(path):
    for line in path.read_text().splitlines():
        match = re.match(r"\s*(?:export\s+)?RUNPOD_API_KEY\s*=\s*(.*)", line)
        if match:
            return match[1].strip().strip("\"'")
    raise SystemExit("provider credential is unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--provider-env", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--reservation", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    config = json.loads(args.config.read_text())
    cap = config["limits"]["max_run_usd"]
    deadline = config["limits"]["run_timeout_s"]
    ledger = json.loads(args.ledger.read_text()) if args.ledger.exists() else {"runs": []}
    spent = sum(row["estimated_rental_usd"] for row in ledger["runs"])
    if not command or cap > args.reservation - spent or args.out.exists():
        raise SystemExit("fresh output and cumulative budget preflight required")
    if any(not row["all_absent"] for row in ledger["runs"]):
        raise SystemExit("previous owned allocation cleanup is unconfirmed")
    key = credential(args.provider_env)
    args.out.mkdir(parents=True)
    began = time.monotonic()
    since = datetime.datetime.now(datetime.timezone.utc).isoformat()
    owned = {}
    tokens = {}
    seen = set()
    event_ids = {}
    stopped = threading.Event()
    reason = None

    def api(path, method="GET", timeout=4, extra=None):
        headers = {"Authorization": "Bearer " + key, "User-Agent": "Switchyard/0.5"}
        headers.update(extra or {})
        return urllib.request.urlopen(urllib.request.Request(
            "https://api.runpod.io/v2" + path, method=method, headers=headers), timeout=timeout)

    def sanitize(value):
        value = str(value).replace(key, "[redacted]")
        for token in tokens.values():
            value = value.replace(token, "[redacted]")
        value = re.sub(r"https?://[^\s\"'<>]+", "[url]", value)
        return re.sub(r"Bearer\s+\S+", "Bearer [redacted]", value)

    def journals():
        valid = True
        for path in args.out.rglob("owned-pods.json"):
            try:
                rows = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for row in rows:
                pod = row["pod"]
                if not pod["name"].startswith("switchyard-"):
                    valid = False
                    continue
                owned[pod["id"]] = row
        return valid and len(owned) <= 1

    def halt(value):
        nonlocal reason
        reason = reason or value
        if child.poll() is None:
            child.send_signal(signal.SIGINT)

    def telemetry():
        while not stopped.is_set():
            if not journals():
                halt("ownership invariant failed")
            for pod_id, row in list(owned.items()):
                try:
                    with api("/pods/" + pod_id) as response:
                        metadata = json.load(response)
                    if metadata.get("name") != row["pod"]["name"]:
                        halt("ownership changed")
                        continue
                    token = metadata.get("env", {}).get("SWITCHYARD_AGENT_TOKEN")
                    if token:
                        tokens[pod_id] = token
                        request = urllib.request.Request(
                            "https://" + pod_id + "-8080.proxy.runpod.net/state",
                            headers={"Authorization": "Bearer " + token, "User-Agent":"Switchyard/0.5"})
                        try:
                            with urllib.request.urlopen(request, timeout=3) as response:
                                state = json.load(response)
                            with (args.out / "agent-state.ndjson").open("a") as out:
                                out.write(json.dumps({"at_s":time.monotonic()-began,
                                                      "pod_id":pod_id,"state":state}) + "\n")
                            if state.get("phase") == "failed":
                                halt("agent failed; no retry")
                        except (urllib.error.URLError, TimeoutError, OSError):
                            pass
                    del metadata
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
            for path in args.out.rglob("allocation-rejections.json"):
                try:
                    if json.loads(path.read_text()):
                        halt("allocation rejected; no reprovision")
                except (OSError, ValueError):
                    pass
            stopped.wait(1)

    def logs():
        while not stopped.is_set():
            for pod_id in list(owned):
                params = urllib.parse.urlencode({"source":"container","tail":5000,"since":since})
                try:
                    extra = {"Last-Event-ID":event_ids[pod_id]} if pod_id in event_ids else None
                    with api("/pods/"+pod_id+"/logs?"+params, timeout=3, extra=extra) as response:
                        until = time.monotonic() + 3
                        marker = None
                        while not stopped.is_set() and time.monotonic() < until:
                            line = response.readline().decode("utf8", errors="replace").strip()
                            if line.startswith("id:"):
                                marker = line[3:].strip()
                            elif line.startswith("data:"):
                                try:
                                    record = json.loads(line[5:])
                                except ValueError:
                                    continue
                                identity = (pod_id, record.get("ts"), record.get("line"))
                                if identity not in seen:
                                    seen.add(identity)
                                    with (args.out / "container-events.ndjson").open("a") as out:
                                        out.write(json.dumps({"pod_id":pod_id,"at_s":time.monotonic()-began,
                                            "ts":record.get("ts"),"line":sanitize(record.get("line", ""))})+"\n")
                                if record.get("line", "").strip() in (
                                    'Error: "invalid ticket"',
                                    'Error: "ticket download was rejected"',
                                    'Error: "ticket download failed"',
                                    'Error: "ticket body download timed out"',
                                    'Error: "ticket body download interrupted"',
                                    'Error: "ticket JSON or schema is invalid"',
                                ):
                                    halt("agent ticket failed; no retry")
                                if marker:
                                    event_ids[pod_id] = marker
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
            stopped.wait(1)

    with (args.out / "controller.log").open("w") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        workers = [threading.Thread(target=target, daemon=True) for target in (telemetry, logs)]
        for worker in workers:
            worker.start()
        try:
            while child.poll() is None:
                if time.monotonic() - began > deadline:
                    halt("independent deadline exceeded")
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                    break
                time.sleep(0.25)
            child.wait(timeout=10)
        finally:
            stopped.set()
            for worker in workers:
                worker.join(timeout=5)
            if not journals():
                reason = reason or "ownership invariant failed"
            cleanup = []
            for pod_id, row in owned.items():
                status = None
                for attempt in range(12):
                    try:
                        with api("/pods/" + pod_id) as response:
                            metadata = json.load(response)
                        if metadata.get("name") != row["pod"]["name"]:
                            reason = "ownership changed; cleanup refused"
                            break
                        del metadata
                        with api("/pods/" + pod_id, "DELETE"):
                            pass
                    except urllib.error.HTTPError as error:
                        status = error.code
                        if status == 404:
                            break
                    except (urllib.error.URLError, TimeoutError, OSError):
                        status = None
                    time.sleep(1)
                cleanup.append({"pod_id":pod_id,"name":row["pod"]["name"],"get_status":status})
            receipts = []
            for path in args.out.rglob("rental-receipt.json"):
                receipts.append(json.loads(path.read_text()))
            cost = sum(r.get("estimated_rental_usd", 0) for r in receipts)
            # If CLI termination failed, charge the full independent observed wall interval.
            if owned and (not receipts or any(row["ended_s"] is None for row in owned.values())):
                cost = max(cost, (time.monotonic()-began) * config["limits"]["max_usd_per_hour"] / 3600)
            result = {"run":str(args.out),"started_at":since,"controller_exit":child.returncode,
                      "elapsed_s":time.monotonic()-began,"reason":reason,
                      "estimated_rental_usd":cost,"receipts":receipts,"cleanup":cleanup,
                      "all_absent":all(row["get_status"] == 404 for row in cleanup)}
            (args.out / "independent-receipt.json").write_text(json.dumps(result, indent=2)+"\n")
            ledger["runs"].append(result)
            args.ledger.parent.mkdir(parents=True, exist_ok=True)
            args.ledger.write_text(json.dumps(ledger, indent=2)+"\n")
            print(json.dumps({k:result[k] for k in ("run","controller_exit","elapsed_s","estimated_rental_usd","all_absent","reason")}))
    if child.returncode or reason or not result["all_absent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
