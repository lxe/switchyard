"""Read and sanitize provider pod logs independently of inference or scheduling."""

import json
import re
import time
import urllib.parse
import urllib.request


def collect_logs(control, pod_id, stopped, redactions=()):
    seen = set()
    reported_errors = set()
    while not stopped.is_set():
        # Every allocation has a fresh container; retain both provider and native
        # history without filtering on differences between controller/host clocks.
        params = urllib.parse.urlencode({"tail": 5000})
        request = urllib.request.Request(
            control.config["provider"]["api_url"]
            + "/pods/"
            + pod_id
            + "/logs?"
            + params,
            headers={
                "Authorization": "Bearer " + control.key,
                "User-Agent": "Switchyard/0.5",
                "Accept": "text/event-stream",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                until = time.monotonic() + 3
                while not stopped.is_set() and time.monotonic() < until:
                    line = (
                        response.readline(1048576)
                        .decode("utf8", errors="replace")
                        .strip()
                    )
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    item = json.loads(line[5:])
                    identity = (item.get("ts"), item.get("line"))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    item["line"] = re.sub(
                        r"https?://[^\s\"'<>]+", "[url]", item.get("line", "")
                    )
                    for secret in redactions:
                        item["line"] = item["line"].replace(secret, "[redacted]")
                    with (control.output / "container-events.ndjson").open(
                        "a"
                    ) as target:
                        target.write(json.dumps(item) + "\n")
        except (OSError, ValueError, TimeoutError) as error:
            kind = type(error).__name__
            status = getattr(error, "code", None)
            if (kind, status) not in reported_errors:
                reported_errors.add((kind, status))
                control.event(
                    "provider_log_read_notice", error_type=kind, http_status=status
                )
        stopped.wait(1)
