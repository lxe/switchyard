"""Summarize a terminal live gauntlet and independently observe owned-pod cleanup."""

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from control import env_file, write


def read(path):
    return json.loads(path.read_text())


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cleanup_status(api_url, key, pod):
    """A missing pod is the provider's cleanup evidence; never stop other pods."""
    request = urllib.request.Request(
        api_url + "/pods/" + pod,
        headers={"Authorization": "Bearer " + key, "User-Agent": "Switchyard/0.5"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return {
                "pod_id": pod,
                "http_status": response.status,
                "confirmed_absent": False,
            }
    except urllib.error.HTTPError as error:
        return {
            "pod_id": pod,
            "http_status": error.code,
            "confirmed_absent": error.code == 404,
        }


def pod_summary(directory, owner, observations):
    pod = owner["pod"]["id"]
    logs = records(directory / "provider-logs" / pod / "container-events.ndjson")
    markers = {}
    phases = {}
    memory = []
    prefix = {
        "observed_events": 0,
        "input_tokens": 0,
        "computed_tokens": 0,
        "reused_tokens": 0,
    }
    batches = {}
    for record in logs:
        line = record.get("line", "")
        if line.startswith("native_cache_family_") and "=" in line:
            name, value = line.split("=", 1)
            if name in {
                "native_cache_family_observed_gpu",
                "native_cache_family_selected_driver",
                "native_cache_family_compute_cap",
                "native_cache_family_status",
                "native_cache_family_policy",
            }:
                markers[name.removeprefix("native_cache_family_")] = value
        try:
            event = json.loads(line.removeprefix("native_host_memory="))
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if line.startswith("native_host_memory="):
            memory.append(event)
        elif event.get("event") in {
            "process_started",
            "device_initialization_started",
            "device_initialized",
            "weights_read_started",
            "weights_read_finished",
            "model_loaded",
            "tokenizer_loaded",
            "http_listening",
            "prefix_prefill",
            "batch_started",
            "batch_finished",
        }:
            phases.setdefault(event["event"], event["elapsed_s"])
        if event.get("event") == "prefix_prefill":
            prefix["observed_events"] += 1
            for field in ("input_tokens", "computed_tokens", "reused_tokens"):
                prefix[field] += event["details"].get(field, 0)
        if event.get("event") in {"batch_finished", "mixed_batch_finished"}:
            size = str(event["details"]["batch_size"])
            batches[size] = batches.get(size, 0) + 1
    states = [
        row["state"]
        for row in observations
        if row.get("pod") == pod and row.get("state")
    ]
    state = states[-1] if states else {}
    milestones = {
        event["event"]: event["elapsed_s"]
        for event in state.get("startup_milestones", [])
    }
    hydration = {}
    for role in ("runtime", "model"):
        start, finish = (
            milestones.get(role + "_hydration_started"),
            milestones.get(role + "_hydration_finished"),
        )
        if start is not None and finish is not None:
            hydration[role + "_elapsed_s"] = finish - start
    transfers = (state.get("model_transport") or {}).get("transfers", [])
    return {
        "pod": owner["pod"],
        "allocated_s": owner["allocated_s"],
        "ended_s": owner.get("ended_s"),
        "quote": owner["quoted"],
        "observed_platform": markers,
        "last_agent_phase": state.get("phase"),
        "inference_verified": state.get("inference_verified"),
        "agent_clock_milestones": milestones,
        "hydration": hydration,
        "raw_model_transport": {
            "observed_ranges": len(transfers),
            "received_bytes": sum(row["received_bytes"] for row in transfers),
            "written_bytes": sum(row["written_bytes"] for row in transfers),
            "body_wait_worker_seconds": sum(row["body_wait_seconds"] for row in transfers),
            "storage_write_worker_seconds": sum(row["storage_write_seconds"] for row in transfers),
        } if transfers else None,
        "engine_clock_first_phase_events": phases,
        "host_memory_samples": memory,
        "captured_prefix_prefill": prefix,
        "captured_finished_batch_sizes": batches,
        "evidence_sha256": {
            "provider_logs": sha256(
                directory / "provider-logs" / pod / "container-events.ndjson"
            )
            if logs
            else None,
        },
    }


def summarize(run, key):
    # Terminal receipts are written after owned cleanup. Their absence means
    # execution is incomplete, so no portable acceptance claim can be emitted.
    observer = read(run / "observer-outcome.json")
    identity = read(run / "observer-source.json")
    source = read(run / "run-source.json")
    fleet = read(run / "fleet-source.json")
    experiment = read(run / "workloads.json")
    expected = {case["name"]: case["calls"] for case in experiment["cases"]}
    selected = identity.get("selected_model")
    models = []
    for config in fleet["models"]:
        model = config["id"]
        if selected is not None and model != selected:
            continue
        directory = run / model
        if not directory.exists():
            models.append({
                "model": model,
                "revision": config["revision"],
                "engine": config["engine"],
                "model_archive_sha256": config.get("archive_sha256"),
                "model_artifact": config.get("model_artifact"),
                "executed": False,
                "functional_workloads_passed": False,
                "workloads": [
                    {"name": name, "expected_requests": count, "recorded": False}
                    for name, count in expected.items()
                ],
                "cleanup": [],
            })
            continue
        outcome = read(directory / "outcome.json")
        owners = read(directory / "owned-pods.json")
        pending = read(directory / "pending-launches.json")
        if any(not row.get("resolved") for row in pending):
            raise ValueError(
                "Unresolved allocation journal requires owned recovery before publication"
            )
        cleanup = [
            cleanup_status(fleet["compute"]["api_url"], key, row["pod"]["id"])
            for row in owners
        ]
        if not all(row["confirmed_absent"] for row in cleanup):
            raise ValueError("Provider has not confirmed every owned pod absent")
        reports = []
        for name, count in expected.items():
            path = directory / "workloads" / (name + ".json")
            if not path.exists():
                reports.append(
                    {"name": name, "expected_requests": count, "recorded": False}
                )
                continue
            report = read(path)
            if report["requests"] != count or report["name"] != name:
                raise ValueError("Workload report differs from selected experiment")
            report.pop("completions")
            report["failures"] = {
                str(request): "request_failed; see detailed local receipt"
                for request in report["failures"]
            }
            reports.append(
                {
                    **report,
                    "recorded": True,
                    "report_sha256": sha256(path),
                    "throughput_target_met": report["output_tokens_per_s"]
                    >= report["target_output_tokens_per_s"],
                }
            )
        complete = all(
            report.get("recorded")
            and report["completed"] == report["requests"]
            and report["failed"]
            == report["unfinished"]
            == report["invalid_actions"]
            == 0
            and not report["budget_exceeded"]
            for report in reports
        )
        observations = records(directory / "agent-observations.ndjson")
        models.append(
            {
                "model": model,
                "revision": config["revision"],
                "engine": config["engine"],
                "model_archive_sha256": config.get("archive_sha256"),
                "model_artifact": config.get("model_artifact"),
                "executed": True,
                "functional_workloads_passed": complete
                and outcome == {"error": None, "cleanup_error": None},
                "outcome": outcome,
                "cleanup": cleanup,
                "rental": read(directory / "rental-receipt.json"),
                "pods": [
                    pod_summary(directory, owner, observations) for owner in owners
                ],
                "workloads": reports,
                "cache_registry_events": records(
                    directory / "native-cache-registry.ndjson"
                ),
                "evidence_sha256": {
                    "observations": sha256(directory / "agent-observations.ndjson")
                    if (directory / "agent-observations.ndjson").exists()
                    else None
                },
            }
        )
    if not models:
        raise ValueError("Selected model is absent from retained fleet configuration")
    return {
        "format": "switchyard-native-gauntlet-evidence-v1",
        "controller_identity": {
            **identity,
            "checkout_commit_at_launch": source.get("source_commit"),
            "compiled_source_commit": "not embedded; CLI SHA identifies the executable",
        },
        "runtime": fleet["bootstrap"]["runtime"],
        "bootstrap_image": fleet["bootstrap"]["image"],
        "agent_sha256": fleet["bootstrap"]["agent_sha256"],
        "limits": fleet["limits"],
        "observer": observer,
        "models": models,
        "functional_workloads_passed": observer["returncode"] == 0
        and all(model["functional_workloads_passed"] for model in models),
        "limits_of_evidence": [
            "F32 serving; these cloud requests do not add a teacher-forced numerical accuracy gate.",
            "Throughput target misses are recorded independently of functional completion.",
            "Agent and engine clocks have separate epochs; hydration intervals overlap.",
            "Engine first phase events include HTTP arrival time, not isolated JIT compilation cost.",
            "Runtime identity comes from pinned published manifests; no serve-time content hashing is added.",
            "Detailed transport traces, completion hashes and scheduler decisions remain in ignored local results.",
            "Prefix and batch counts cover retained provider log events; delivery completeness is not assumed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import os

    name = read(args.run / "fleet-source.json")["compute"]["key_env"]
    key = os.environ.get(name)
    for path in args.env_file:
        if key is None:
            key = env_file(path).get(name)
    if not key:
        raise ValueError("Configured provider credential is absent")
    receipt = summarize(args.run, key)
    # Refuse to export an accidentally retained secret or signed capability.
    serialized = json.dumps(receipt)
    if (
        key in serialized
        or "X-Amz-Signature" in serialized
        or "X-Amz-Credential" in serialized
    ):
        raise ValueError("Receipt contains a credential or signed capability")
    write(args.output, receipt)
    print(
        json.dumps(
            {
                "models": len(receipt["models"]),
                "functional_workloads_passed": receipt["functional_workloads_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
