"""Bounded HTTP workloads after native serving becomes available."""

import concurrent.futures
import json
import math
import statistics
import threading
import time
import urllib.request

from control import write

BASE = "List the integers from 1 to 100 in order, separated by spaces. Keep writing until you reach 100."


def stream_call(endpoint, headers, model, prompt, max_tokens, barrier=None):
    if barrier:
        barrier.wait(timeout=10)
    started = time.monotonic()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={**headers, "Content-Type": "application/json"},
    )
    first_content = None
    tokens = []
    content = []
    frames = []
    with urllib.request.urlopen(request, timeout=90) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            item = json.loads(data)
            frames.append(item)
            choice = item["choices"][0]
            tokens.extend(choice.get("token_ids", []))
            text = choice.get("delta", {}).get("content", "")
            if text:
                content.append(text)
                if first_content is None:
                    first_content = time.monotonic() - started
    finished = time.monotonic()
    if not tokens or first_content is None:
        raise ValueError("workload produced no streamed native tokens/content")
    return {
        "request": body,
        "request_id": frames[0]["id"],
        "started_monotonic_s": started,
        "finished_monotonic_s": finished,
        "elapsed_s": finished - started,
        "first_content_s": first_content,
        "first_token_engine_s": frames[0].get("token_elapsed_s"),
        "emitted_token_ids": tokens,
        "emitted_tokens": len(tokens),
        "content": "".join(content),
        "finish_reason": frames[-1]["choices"][0].get("finish_reason"),
        "queue_wait_s": None,
    }


def summarize(name, calls):
    elapsed = max(c["finished_monotonic_s"] for c in calls) - min(
        c["started_monotonic_s"] for c in calls
    )
    tokens = sum(c["emitted_tokens"] for c in calls)
    times = sorted(c["first_content_s"] for c in calls)
    return {
        "name": name,
        "concurrent_requests": len(calls),
        "requested_token_budget": sum(c["request"]["max_tokens"] for c in calls),
        "emitted_tokens_including_stop_ids": tokens,
        "elapsed_s": elapsed,
        "aggregate_tokens_per_s": tokens / elapsed,
        "first_content_s": {
            "min": times[0],
            "p50": statistics.median(times),
            "p95": times[math.ceil(0.95 * len(times)) - 1],
            "max": times[-1],
        },
        "calls": calls,
    }


def run_workloads(control, endpoint, headers, model, budget):
    max_tokens = budget["max_tokens"]
    concurrency = budget["concurrency"]
    if max_tokens != 32 or concurrency != 8:
        raise ValueError(
            "fleet workload requires the literal 32-token/eight-call budget"
        )
    reports = []
    control.event(
        "hot_workloads_started", max_tokens=max_tokens, concurrency=concurrency
    )
    reports.append(
        summarize(
            "hot_single", [stream_call(endpoint, headers, model, BASE, max_tokens)]
        )
    )
    for name, prompts in [
        ("equal_length_burst", [BASE] * concurrency),
        (
            "varied_length_burst",
            [BASE + " Use plain text." * i for i in range(concurrency)],
        ),
    ]:
        barrier = threading.Barrier(concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    stream_call, endpoint, headers, model, prompt, max_tokens, barrier
                )
                for prompt in prompts
            ]
            calls = [future.result() for future in futures]
        reports.append(summarize(name, calls))
    receipt = {
        "format": "switchyard-native-hot-http-workloads-v1",
        "model": model,
        "workloads": reports,
        "queue_wait_available": False,
        "clock": "local monotonic; native first token elapsed separately recorded",
        "batching": "observed native batch_started telemetry; proxy arrival times may split cohorts",
    }
    write(control.output / "workloads.json", receipt)
    control.event(
        "hot_workloads_finished",
        workloads=[
            {
                "name": r["name"],
                "emitted_tokens": r["emitted_tokens_including_stop_ids"],
                "elapsed_s": r["elapsed_s"],
                "tokens_per_s": r["aggregate_tokens_per_s"],
            }
            for r in reports
        ],
    )
    return receipt
