"""Summarize an actual-agent rollout without adding overlapping worker timings.

Detailed shard counters are sums of worker-seconds. Milestones and publication
measure elapsed time. Decoder read time includes source waits; their difference
estimates decoder work plus scheduling, not pure decompression CPU time.
"""

import argparse
import collections
import json
from pathlib import Path


def json_lines(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line]
        if path.exists()
        else []
    )


def summarize(root):
    receipt = json.loads((root / "receipt.json").read_text())
    names = {
        artifact["id"]: artifact["name"]
        for artifact in receipt["agent_state"]["artifacts"]
    }
    groups = collections.defaultdict(list)
    for item in json_lines(root / "hydration-profile.jsonl"):
        groups[item["artifact_id"]].append(item)
    artifacts = {}
    shared_workers = []
    for identity, items in groups.items():
        totals = collections.Counter()
        worker_events = []
        stage = {}
        first = next(x["elapsed_seconds"] for x in items if x["event"] == "prepared")
        slots = []
        blocking = []
        for item in items:
            event = item["event"]
            if "unix_seconds" in item and event in {
                "decoder_started",
                "decoder_finished",
            }:
                shared_workers.append(
                    (item["unix_seconds"], 1 if event == "decoder_started" else -1)
                )
            at = item["elapsed_seconds"]
            if item["object"] is None:
                stage[event] = {"elapsed_seconds": at, **item["metrics"]}
            if event == "shard_scheduled":
                slots.append(at - first)
            elif event == "decoder_started":
                worker_events.append((at, 1))
                blocking.append(item["metrics"]["blocking_queue_seconds"])
            elif event == "decoder_finished":
                worker_events.append((at, -1))
                totals.update(item["metrics"])
        active = peak = 0
        for at, delta in sorted(worker_events):
            active += delta
            peak = max(peak, active)
        artifacts[names[identity]] = {
            "artifact_id": identity,
            "stages": stage,
            "summed_worker_metrics": dict(totals),
            "max_worker_slot_wait_seconds": max(slots, default=0),
            "max_blocking_queue_seconds": max(blocking, default=0),
            "max_active_decoders": peak,
            "average_active_decoders": totals["worker_elapsed_seconds"]
            / stage["published"]["elapsed_seconds"],
        }
    shared_active = shared_peak = 0
    for at, delta in sorted(shared_workers):
        shared_active += delta
        shared_peak = max(shared_peak, shared_active)
    memory = json_lines(root / "container-memory.jsonl")
    waits = collections.Counter(
        thread["wchan"]
        for item in json_lines(root / "thread-waits.jsonl")
        for thread in item["threads"]
    )
    resources = memory[-1]["resources"] if memory else {}

    def stat(text):
        return {
            key: int(value)
            for key, value in (line.split() for line in (text or "").splitlines())
        }

    cpu = stat(resources.get("cpu.stat"))
    disk = {}
    if memory:

        def device(sample):
            return {
                row[2]: row
                for row in (
                    line.split()
                    for line in sample["resources"]["host_diskstats"].splitlines()
                )
            }

        initial, final = device(memory[0]), device(memory[-1])
        for name, end in final.items():
            start = initial[name]
            disk[name] = {
                "read_bytes": (int(end[5]) - int(start[5])) * 512,
                "write_bytes": (int(end[9]) - int(start[9])) * 512,
                "busy_seconds": (int(end[12]) - int(start[12])) / 1000,
            }
    return {
        "agent_binary_sha256": receipt["agent_binary_sha256"],
        "shared_hydration_concurrency": receipt["shared_hydration_concurrency"],
        "observed_shared_decoder_peak": shared_peak if shared_workers else None,
        "request_to_first_useful_token_seconds": receipt.get(
            "request_to_first_useful_token_seconds"
        ),
        "container_memory": receipt["container_memory"],
        "artifacts": artifacts,
        "kernel_wait_channel_samples": dict(waits),
        "cgroup_cpu_stat_final": cpu,
        "cgroup_io_pressure_final": resources.get("io.pressure"),
        "cgroup_memory_pressure_final": resources.get("memory.pressure"),
        "host_disk_activity_delta": disk,
        "host_disk_sample_elapsed_seconds": (
            memory[-1]["monotonic_s"] - memory[0]["monotonic_s"] if memory else None
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Completed extents directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, artifact in result["artifacts"].items():
        totals = artifact["summed_worker_metrics"]
        print(
            f"{name}: {artifact['stages']['published']['elapsed_seconds']:.3f}s elapsed, {artifact['max_active_decoders']} peak decoders; {totals['file_write_seconds']:.3f} write worker-s, {totals['permit_wait_seconds']:.3f} permit worker-s"
        )
    print("first token:", result["request_to_first_useful_token_seconds"])
    print("binary SHA-256:", result["agent_binary_sha256"])
