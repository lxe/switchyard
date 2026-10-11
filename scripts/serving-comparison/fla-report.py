"""Derive FLA repeat acceptance, native timing, and hardware evidence offline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def metric_values(path):
    values = {}
    for record in read(path):
        for line in record.get("metrics", "").splitlines():
            if not line or line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name.startswith("http_request_duration_seconds_"):
                handler = re.search(r'handler="([^"]+)"', line)
                if not handler or handler[1] != "/v1/chat/completions":
                    continue
            if name in METRICS:
                values[name] = values.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return values


def native_case(run, name):
    directory = Path(run) / "qwen3.5-0.8b/workloads"
    before = metric_values(directory / (name + "-metrics-before.json"))
    after = metric_values(directory / (name + "-metrics-after.json"))
    report = read(directory / (name + ".json"))
    delta = {key: value - before.get(key, 0.0) for key, value in after.items()}
    means = {
        key: delta[key + "_sum"] / delta[key + "_count"]
        for key in REQUEST_METRICS if delta.get(key + "_count", 0) > 0
    }
    item = {"case": name, "completed": report["completed"],
            "output_tokens": report["output_tokens"], "elapsed_s": report["elapsed_s"],
            "output_tokens_per_s": report["output_tokens_per_s"],
            "native_delta": delta, "native_request_means_s": means}
    assert all(delta[key + "_count"] == report["completed"] for key in REQUEST_METRICS)
    if name == "hot-single-long":
        completions = sorted(report["completions"], key=lambda row: row["request"])
        gaps = [{"completed_request": previous["request"], "next_request": current["request"],
                 "completed_s": previous["completed_s"], "next_started_s": current["started_s"],
                 "gap_s": current["started_s"] - previous["completed_s"]}
                for previous, current in zip(completions, completions[1:])]
        active = sum(row["completed_s"] - row["started_s"] for row in completions)
        span = completions[-1]["completed_s"] - completions[0]["started_s"]
        native = delta["vllm:request_inference_time_seconds_sum"]
        observations = read(Path(run) / "qwen3.5-0.8b/agent-observations.json")
        item["serial_timing"] = {
            "completions": [{key: row[key] for key in
                             ["request", "started_s", "first_content_s", "completed_s", "output_tokens"]}
                            for row in completions],
            "completion_to_next_send_gaps": gaps,
            "completion_to_next_send_gap_sum_s": sum(row["gap_s"] for row in gaps),
            "client_request_active_sum_s": active,
            "first_send_to_last_completion_s": span,
            "case_elapsed_outside_first_send_to_last_completion_s": report["elapsed_s"] - span,
            "native_inference_sum_s": native,
            "case_elapsed_minus_native_inference_sum_s": report["elapsed_s"] - native,
            "client_active_minus_native_inference_sum_s": active - native,
            "state_observation_start_times_s": [row["at_s"] for row in observations
                if completions[0]["started_s"] - 1 <= row["at_s"] <= completions[-1]["completed_s"]],
            "state_http_rtt_recorded": False,
            "workload_probe_count": report["probes"],
        }
        assert abs(span - active - sum(row["gap_s"] for row in gaps)) < 1e-8
    return item


def distribution(values):
    return {"samples": len(values), "minimum": min(values), "maximum": max(values),
            "mean": statistics.mean(values)} if values else {"samples": 0}


def acceptance(source, config):
    row = {key: copy.deepcopy(source[key]) for key in
           ["run", "controller_exit", "estimated_rental_usd", "all_absent", "cleanup", "reason",
            "startup_milestones", "artifact_reception", "dependency_budget", "cases"]}
    records = source["native_startup_observations"]
    profile = [json.loads(record["line"].split("SWITCHYARD_STARTUP_PROFILE ", 1)[1])
               for record in records if "SWITCHYARD_STARTUP_PROFILE " in record["line"]]
    row["worker_contexts"] = [record for record in profile if record["event"] == "worker_context"]
    row["process_start_markers"] = [record for record in profile if record["event"] == "process_started"]
    row["cuda_initialization"] = [record for record in profile if record.get("name") == "torch.cuda._lazy_init"]
    row["native_stage_durations"] = [record for record in profile if record["event"] == "end" and
        (record.get("name", "").startswith("vllm.v1.worker.gpu_worker.Worker.") or
         record.get("name", "").startswith("flashinfer.jit.core."))]
    row["native_sampling_provider_checks"] = [record for record in profile if "native_provider_count" in record]
    row["native_flashinfer_build_calls"] = sum(record.get("name") == "flashinfer.jit.core.JitSpecNvcc.build"
        and record["event"] == "begin" for record in profile)
    for key, needle in [("native_aot_load_messages", "Directly load AOT"),
                        ("preimport_messages", "preimport completed")]:
        row[key] = [record for record in records if needle in record["line"]]
    row["provider_discovery"] = [json.loads(record["line"].split("SWITCHYARD_NATIVE_PROVIDER_DISCOVERY ", 1)[1])
        for record in records if "SWITCHYARD_NATIVE_PROVIDER_DISCOVERY " in record["line"]]
    milestones = {record["event"]: record["elapsed_s"] for record in row["startup_milestones"]}
    for field, event in [("engine_spawn_to_inference_verified_s", "engine_spawned"),
                         ("model_gate_to_inference_verified_s", "model_gate_released")]:
        row[field] = milestones["inference_verified"] - milestones[event]
    contexts = row["worker_contexts"]
    api, worker = contexts[0]["pid"], contexts[-1]["pid"]
    api_start = next(record["monotonic_s"] for record in row["process_start_markers"] if record["pid"] == api)
    worker_init = next(record["monotonic_s"] for record in profile if record.get("name") ==
                       "vllm.v1.worker.gpu_worker.Worker.init_device" and record["event"] == "begin")
    row["api_marker_to_worker_init_s"] = worker_init - api_start
    row["api_context_to_worker_context_s"] = contexts[-1]["monotonic_s"] - contexts[0]["monotonic_s"]
    row["arrival_to_first_content_s"] = next(case["ttft_p50_s"] for case in row["cases"] if case["name"] == "cold-single")
    row["total_completed"] = sum(case["completed"] for case in row["cases"])
    row["literal_correct"] = sum(case.get("literal_correct", 0) for case in row["cases"])
    runtime = config["bootstrap"]["runtime"]
    row["runtime_id"] = runtime["artifact"]["id"]
    row["cache_id"] = config["models"][0]["caches"][0]["artifact"]["id"]
    row["launch_fingerprint"] = config["models"][0]["caches"][0]["compatibility"]["runtime_launch_configuration"]
    fingerprint = hashlib.sha256(json.dumps(
        [runtime["program"], runtime["args"], dict(sorted(runtime["environment"].items()))],
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert row["launch_fingerprint"] == fingerprint
    row["native_hardware_observations"] = source["native_hardware_observations"]
    inventory = next(record for record in row["native_hardware_observations"] if record["event"] == "inventory")
    warmup_end = next(record["monotonic_s"] for record in profile if record["event"] == "end" and
                     record.get("name") == "vllm.v1.worker.gpu_worker.Worker.compile_or_warm_up_model")
    samples = [record for record in row["native_hardware_observations"]
               if record["event"] == "sample" and record["monotonic_s"] >= warmup_end]
    row["hardware_serving_summary"] = {
        "scope": "Two-second samples after native worker warmup through final owned-process observation; short bursts can fall between samples.",
        "inventory": inventory, "warmup_end_monotonic_s": warmup_end, "serving_samples": len(samples),
        "gpu_uuids": sorted({gpu["uuid"] for sample in samples for gpu in sample.get("gpu", [])}),
        "driver_versions": sorted({gpu["driver_version"] for sample in samples for gpu in sample.get("gpu", [])}),
        "gpu": {field: distribution([float(gpu[field]) for sample in samples for gpu in sample.get("gpu", [])])
                for field in ["clocks.current.sm", "clocks.current.memory", "enforced.power.limit", "power.draw", "temperature.gpu", "utilization.gpu"]},
        "processes": {},
    }
    for label, pid in [("api", api), ("worker", worker)]:
        processes = [process for sample in samples for process in sample["processes"] if process["pid"] == pid]
        row["hardware_serving_summary"]["processes"][label] = {
            "pid": pid, "threads": sorted({int(process["Threads"]) for process in processes}),
            "cpu_core_equivalents": distribution([process["cpu_core_equivalents"] for process in processes if "cpu_core_equivalents" in process]),
            "cpus_allowed": sorted({process["Cpus_allowed_list"] for process in processes}),
            "mems_allowed": sorted({process["Mems_allowed_list"] for process in processes}),
        }
    assert row["total_completed"] == 222 and row["literal_correct"] == 9
    assert row["controller_exit"] == 0 and row["all_absent"]
    assert all(record["get_status"] == 404 for record in row["cleanup"])
    assert all(not case[key] for case in row["cases"] for key in ["failed", "unfinished", "invalid_actions", "budget_exceeded"])
    assert row["native_aot_load_messages"] and row["native_flashinfer_build_calls"] == 0
    assert row["native_sampling_provider_checks"] and all(record["native_provider_count"] == record["native_aot_artifact_count"] == 1
        for record in row["native_sampling_provider_checks"])
    assert runtime["environment"]["FLASHINFER_DISABLE_JIT"] == runtime["environment"]["VLLM_FORCE_AOT_LOAD"] == "1"
    return row


REQUEST_METRICS = ["vllm:request_decode_time_seconds", "vllm:request_inference_time_seconds",
                   "vllm:request_prefill_time_seconds", "vllm:time_to_first_token_seconds"]
METRICS = {name + suffix for name in REQUEST_METRICS for suffix in ["_sum", "_count"]} | {
    "vllm:iteration_tokens_total_sum", "vllm:iteration_tokens_total_count",
    "http_request_duration_seconds_sum", "http_request_duration_seconds_count"}
CASES = ["hot-single-long", "shared-prefix-burst-32", "unique-prefix-burst-32",
         "shared-prefix-queue-64", "unique-prefix-queue-64"]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--summary", type=Path, default=Path("results/comparison-summary.json"))
p.add_argument("--out", type=Path, default=Path("docs/serving-comparison-data"))
a = p.parse_args()
summary = read(a.summary)
runs = {row["run"]: row for row in summary["runs"]}
native_runs = ["results/cloud-vllm-0.8b-fla-nvml-rc2-mo-v2", "results/cloud-vllm-0.8b-fla-control-rc2-mo",
               "results/cloud-vllm-0.8b-fla-reverse-control-mo", "results/cloud-vllm-0.8b-fla-reverse-patched-mo"]
a.out.mkdir(parents=True, exist_ok=True)
write(a.out / "fla-native-decode-comparison.json", {
    "schema_version": 1,
    "definitions": {
        "native_decode": "Pinned vLLM stats: last NEW_TOKEN minus first NEW_TOKEN; includes decode preemptions and scheduling gaps, not exclusive CUDA-kernel time.",
        "native_inference": "Pinned vLLM stats: last NEW_TOKEN minus first SCHEDULED; excludes initial queueing and includes prefill/decode preemptions.",
        "client_started": "Controller monotonic timestamp immediately before building and sending each streaming HTTP request.",
        "case_elapsed": "Workload begin to loop completion, including admission and completion detection; before/after metric HTTP requests are outside this interval.",
        "serial_residual": "For dependency-ordered calls only, case elapsed minus the sum of native inference intervals. This includes dispatch gaps, HTTP/proxy/frontend/queue overhead, and case admission/completion detection; the individual causes are not isolated.",
        "iteration_tokens": "Native histogram counts and sums of tokens per engine iteration; these are cumulative workload deltas, without per-iteration timestamps or batch membership.",
    },
    "runtime_loop_inspection": {
        "workload_source": "crates/switchyard-live/src/runtime/workload.rs",
        "lifecycle_source": "crates/switchyard-live/src/runtime/lifecycle.rs",
        "state_client_source": "crates/switchyard-live/src/serving.rs",
        "finding": "The single scheduling loop drains completion events, then awaits observe() before deciding and dispatching. observe() sequentially awaits each live node's /state. ProbeHost actions likewise await /state before later actions. Observation next starts at least one second after its previous response; loop polling is 100 ms in both repeat configurations. Completion events arriving during either await cannot release dependent requests until a later loop pass.",
        "state_timeout_s": 4, "state_retry_or_backoff": False,
        "serial_probe_actions_recorded": False,
        "limitations": "The receipts record observation start times but no HTTP finish times or failed state calls. Before the 359.853 ms patched dispatch gap, an observation starts 188.493 ms before the preceding request completes. Its missing finish time prevents establishing whether HTTP covered that gap. Blocking observation plus polling is a candidate mechanism; its HTTP RTT and exact contribution remain unmeasured. No probe latency is attributed to this serial case.",
        "next_diagnostic": "Record observation/probe start, finish, status and duration; drain completions independently of control-plane HTTP; measure API validation/tokenization/enqueue, actual Torch intra/inter-op and tokenizer thread state, and timestamped batch size/iteration tokens. Compare local HTTP load and matched safe spawn/fork launches on one host.",
    },
    "runs": [{"run": run, "cases": [native_case(run, name) for name in CASES]} for run in native_runs],
})
configs = [read("results/vllm-fla-reverse-" + variant + "/config.json") for variant in ["control", "patched"]]
rows = [acceptance(runs[run], config) for run, config in zip(native_runs[-2:], configs)]
assert configs[0]["bootstrap"]["agent_sha256"] == configs[1]["bootstrap"]["agent_sha256"]
for key in ["program", "args", "environment"]:
    assert configs[0]["bootstrap"]["runtime"][key] == configs[1]["bootstrap"]["runtime"][key]
assert configs[0]["models"][0]["engine"] == configs[1]["models"][0]["engine"]
assert rows[0]["launch_fingerprint"] == rows[1]["launch_fingerprint"]
assert rows[0]["hardware_serving_summary"]["gpu_uuids"] == rows[1]["hardware_serving_summary"]["gpu_uuids"]
assert len(rows[0]["hardware_serving_summary"]["gpu_uuids"]) == 1
assert rows[0]["hardware_serving_summary"]["driver_versions"] == rows[1]["hardware_serving_summary"]["driver_versions"]
assert summary["all_owned_pods_absent"] and summary["estimated_total_rental_usd"] <= 1.0
for key in ["cpu_model", "cpu_quota", "affinity_cpus", "effective_cpuset", "thread_environment"]:
    assert rows[0]["hardware_serving_summary"]["inventory"][key] == rows[1]["hardware_serving_summary"]["inventory"][key]
observer_digest = hashlib.sha256(Path("scripts/serving-comparison/hardware-observer.py").read_bytes()).hexdigest()
source_entries_unchanged = []
for variant, config in zip(["control", "patched"], configs):
    manifest = read("results/vllm-fla-reverse-" + variant + "/cache-artifact/manifest.json")
    original = read("results/" + ("vllm-sm89-fla-control-rc2-v2" if variant == "control" else "vllm-sm89-fla-nvml-v3") + "/cache-artifact/manifest.json")
    assert manifest["entries"] == original["entries"]
    assert manifest["metadata"]["attributes"]["hardware_observer_sha256"] == observer_digest
    source_entries_unchanged.append(True)
write(a.out / "fla-reverse-order-acceptance.json", {
    "schema_version": 1,
    "scope": "Reverse-order repeat: unpatched scoped control, then FLA patched runtime; Qwen3.5-0.8B BF16 archive model, SM89, identical bounded 222-request corpus.",
    "selected_for_final_fleet": False,
    "provenance": {
        "same_physical_gpu": True, "same_launch_program_args_environment": True,
        "same_agent_engine_model": True, "original_compiler_entries_unchanged": source_entries_unchanged,
        "agent_sha256": configs[0]["bootstrap"]["agent_sha256"],
        "controller_sha256": hashlib.sha256(Path("results/switchyard-controller-rc2").read_bytes()).hexdigest(),
        "hardware_observer_sha256": observer_digest,
        "hardware_observer": "Separate stdlib Python process with -S; no observer thread is added to the serving parent; no CUDA initialization or clock/power changes.",
    },
    "runs": rows,
    "patched_minus_control_s": {key: rows[1][key] - rows[0][key] for key in
        ["arrival_to_first_content_s", "engine_spawn_to_inference_verified_s", "model_gate_to_inference_verified_s",
         "api_marker_to_worker_init_s", "api_context_to_worker_context_s"]},
    "estimated_comparison_total_usd": summary["estimated_total_rental_usd"],
    "remaining_comparison_reservation_usd": 1.0 - summary["estimated_total_rental_usd"],
    "all_owned_pods_absent": summary["all_owned_pods_absent"],
    "limitations": [
        "The two-second hardware sample cannot resolve every short burst or identify the cause of the native aggregate latency differences.",
        "CPU quotas, power limits and sampled GPU clocks match. OS thread counts differ between spawn and fork; actual Torch/tokenizer thread state and causal performance effects are unmeasured.",
        "The serial native inference interval is unchanged, while client dispatch gaps and other time outside native inference increase. Native unique-queue decode latency also increases; it is not explained away by the serial transport finding.",
        "Hydration, overlap and bootstrap vary. The process-start improvement repeats, but these single samples do not establish a reliable end-to-end first-token gain.",
        "FLA GPU acceptance covers the archive-model 0.8B configuration only. Raw-extents loading and larger models require separate patch acceptance. The strict final raw-extents fleet remains unpatched.",
    ],
})
print(json.dumps({"repeat_requests": sum(row["total_completed"] for row in rows),
                  "repeat_literals": sum(row["literal_correct"] for row in rows),
                  "all_owned_pods_absent": summary["all_owned_pods_absent"],
                  "estimated_total_rental_usd": summary["estimated_total_rental_usd"]}))
