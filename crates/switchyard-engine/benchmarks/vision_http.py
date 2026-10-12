"""Owned local image-serving acceptance; no dummy inference or readiness requests."""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def post(endpoint, path, body):
    request = urllib.request.Request(endpoint + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=60)


def stream(endpoint, body):
    started = time.monotonic()
    frames, tokens, content, first, usage = [], [], [], None, None
    with post(endpoint, "/v1/chat/completions", body) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                frames.append("[DONE]")
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            frames.append(event)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                tokens.extend(choice.get("token_ids", []))
                text = choice.get("delta", {}).get("content", "")
                content.append(text)
                if text and first is None:
                    first = time.monotonic()
    if not tokens or first is None or not usage or frames[-1] != "[DONE]":
        raise RuntimeError("Incomplete useful SSE response")
    if usage["completion_tokens"] != len(tokens):
        raise RuntimeError("Stream usage disagrees with token IDs")
    return {"token_ids": tokens, "content": "".join(content), "frames": frames,
            "request_to_first_content_s": first - started,
            "elapsed_s": time.monotonic() - started, "usage": usage,
            "first_content_monotonic_s": first}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--bind", default="127.0.0.1:18191")
    parser.add_argument("--question", default="What colors appear in this image? Answer briefly.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    encoded = base64.b64encode(args.image.read_bytes()).decode()
    body = {"model": config["model"], "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
        {"type": "text", "text": args.question}]}], "max_tokens": 16, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}}
    endpoint = "http://" + args.bind
    environment = dict(os.environ, CUDA_CACHE_PATH=str(args.out / "cuda"))
    environment["CUDA_CACHE_MAXSIZE"] = str(512 << 20)
    began = time.monotonic()
    process = None
    receipt = {"scope": "Local native HTTP/SSE image serving; no cloud hydration or allocation",
               "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               "config": config, "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
               "cache": "Fresh private driver-cache directory; source file residency uncontrolled",
               "image_execution": "Independent model sequences, no tensor-batched image prefill"}
    log_path = args.out / "engine.ndjson"
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen([str(args.binary.resolve()), "--config", str(args.config.resolve()),
                "serve", "--bind", args.bind], stdout=log, stderr=subprocess.STDOUT, env=environment)
            deadline = began + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Engine stopped before listener event; see engine.ndjson")
                if '"http_listening"' in log_path.read_text():
                    receipt["launch_to_observed_listener_s"] = time.monotonic() - began
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Listener event deadline")
                time.sleep(0.01)
            print("phase=listener", flush=True)
            # CPU-only accounting checks do not execute an inference warmup.
            with post(endpoint, "/tokenize", {"model": config["model"], "messages": body["messages"],
                "add_generation_prompt": True}) as response:
                planned = json.load(response)
            receipt["planned_prompt"] = planned
            cold = stream(endpoint, body)
            if cold["usage"]["prompt_tokens"] != planned["count"]:
                raise RuntimeError("Image-expanded usage disagrees with token planning")
            receipt["launch_to_first_content_s"] = cold.pop("first_content_monotonic_s") - began
            receipt["first_image"] = cold
            print(f"phase=first_image launch_to_content_s={receipt['launch_to_first_content_s']:.3f}", flush=True)
            if args.reference:
                reference = json.loads(args.reference.read_text())["token_ids"]
                expected = reference[:reference.index(248044) + 1] if 248044 in reference else reference[:16]
                receipt["reference_token_ids"] = expected
                receipt["reference_exact_match"] = cold["token_ids"] == expected
                if not receipt["reference_exact_match"]:
                    raise RuntimeError("HTTP generation differs from supplied reference")
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                burst = list(pool.map(lambda _: stream(endpoint, body), range(8)))
            elapsed = time.monotonic() - started
            for result in burst:
                result.pop("first_content_monotonic_s")
                if result["token_ids"] != cold["token_ids"] or result["usage"]["prompt_tokens"] != planned["count"]:
                    raise RuntimeError("Repeated image generation or usage differs")
            receipt["image_burst"] = {"calls": burst, "elapsed_s": elapsed,
                "output_tokens_per_s": sum(len(r["token_ids"]) for r in burst) / elapsed}
            text_body = dict(body, messages=[{"role": "user", "content": "Name the capital of France."}])
            receipt["text_after_image"] = stream(endpoint, text_body)
            receipt["text_after_image"].pop("first_content_monotonic_s")
            receipt["passed"] = True
            print(f"phase=image_burst calls=8 output_tps={receipt['image_burst']['output_tokens_per_s']:.1f}", flush=True)
    except Exception as error:
        receipt["passed"] = False
        receipt["error"] = str(error)
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            receipt["owned_process_returncode"] = process.returncode
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
