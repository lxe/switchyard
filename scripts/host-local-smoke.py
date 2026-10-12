#!/usr/bin/env python3
"""Run the Rust host agent with two real CUDA engines over loopback raw ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.server
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--agent', type=Path, default=Path('target/release/switchyard-agent'))
    parser.add_argument('--engine', type=Path, default=Path('target/release/switchyard-engine'))
    parser.add_argument('--configs', nargs=2, default=['configs/engine/qwen3.5-0.8b-rtx5090.json', 'configs/engine/qwen3.5-2b-rtx5090.json'])
    parser.add_argument('--cache-seeds', nargs=2, help='Optional CUDA cache directories captured from the same engine and RTX 5090 host')
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = (root / 'http' / self.path.lstrip('/')).resolve()
            if not path.is_relative_to(root / 'http') or not path.is_file():
                self.send_error(404)
                return
            length = path.stat().st_size
            start, end = 0, length - 1
            ranged = self.headers.get('Range')
            if ranged:
                start, end = map(int, ranged.removeprefix('bytes=').split('-'))
                if not 0 <= start <= end < length:
                    self.send_error(416)
                    return
            self.send_response(206 if ranged else 200)
            self.send_header('Content-Length', str(end - start + 1))
            if ranged:
                self.send_header('Content-Range', f'bytes {start}-{end}/{length}')
            self.end_headers()
            try:
                with path.open('rb') as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = source.read(min(remaining, 1 << 20))
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    source = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=source.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{source.server_port}'
    subprocess.run(['target/release/examples/prepare_host_fixture', str(root), str(args.engine), base, *args.configs, *(args.cache_seeds or [])], check=True)
    ticket = json.loads((root / 'http/ticket.json').read_text())
    token = secrets.token_hex(32)
    env = dict(os.environ, SWITCHYARD_TICKET_URL=f'{base}/ticket.json', SWITCHYARD_AGENT_TOKEN=token, SWITCHYARD_AGENT_BIND='127.0.0.1:8080')
    log = (root / 'agent.log').open('w')
    epoch = time.monotonic()
    agent = subprocess.Popen([str(args.agent.resolve())], env=env, stdout=log, stderr=log)
    gpu_log = (root / 'gpu.csv').open('w')
    gpu = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu', '--format=csv', '-lms', '500'], stdout=gpu_log)
    samples, ready, phases = [], {}, {}

    def get(path):
        request = urllib.request.Request(f'http://127.0.0.1:8080{path}', headers={'Authorization': f'Bearer {token}'})
        return json.load(urllib.request.urlopen(request, timeout=10))

    def infer(endpoint, stage):
        started = time.monotonic()
        payload = json.dumps({'model': endpoint['ticket']['model'], 'messages': [{'role': 'user', 'content': 'Reply with exactly one word: hello.'}], 'max_tokens': 32, 'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:8080/endpoints/{endpoint['id']}/v1/chat/completions", data=payload, headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
        first, content, usage = None, '', None
        with urllib.request.urlopen(request, timeout=60) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                    continue
                value = json.loads(line[6:])
                for choice in value.get('choices', []):
                    text = choice.get('delta', {}).get('content', '')
                    if text:
                        first = first or time.monotonic()
                        content += text
                usage = value.get('usage') or usage
        if first is None or usage is None:
            raise RuntimeError('inference did not deliver content and usage')
        return {'model': endpoint['ticket']['model'], 'stage': stage, 'started_s': started - epoch, 'first_content_s': first - epoch, 'ttft_s': first - started, 'elapsed_s': time.monotonic() - started, 'content_sha256': hashlib.sha256(content.encode()).hexdigest(), 'usage': usage}

    pool = ThreadPoolExecutor(max_workers=16)
    initial_calls = []
    try:
        with (root / 'observations.ndjson').open('w') as telemetry:
            while len(ready) < len(ticket['endpoints']):
                if agent.poll() is not None or time.monotonic() - epoch > 120:
                    raise RuntimeError('agent exited or startup exceeded fixture timeout')
                try:
                    state = get('/state')
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.1)
                    continue
                telemetry.write(json.dumps({'at_s': time.monotonic() - epoch, 'state': state}) + '\n')
                telemetry.flush()
                for endpoint in state['endpoints']:
                    phase = endpoint['state']['phase']
                    if phases.get(endpoint['id']) != phase:
                        print(f"[{time.monotonic()-epoch:.3f}s] {endpoint['id']}: {phase}", flush=True)
                        phases[endpoint['id']] = phase
                    if phase == 'failed':
                        raise RuntimeError(endpoint['state']['error'])
                    if phase == 'ready' and endpoint['id'] not in ready:
                        ready[endpoint['id']] = time.monotonic() - epoch
                        deployment = next(e for e in ticket['endpoints'] if e['id'] == endpoint['id'])
                        initial_calls.append(pool.submit(infer, deployment, 'single'))
                time.sleep(0.1)
        samples.extend(future.result() for future in initial_calls)
        samples.extend(pool.map(lambda e: infer(e, 'mixed-burst'), ticket['endpoints'] * 8))
        for endpoint in ticket['endpoints']:
            rows = [r for r in samples if r['model'] == endpoint['ticket']['model']]
            if len({r['content_sha256'] for r in rows}) != 1:
                raise RuntimeError('single and batched output digests differ')
        receipt = {'scope': 'Fresh local hydration destinations over loopback HTTP with warm source files and existing CUDA libraries; no provider allocation or R2 network measurement.', 'agent_sha256': hashlib.sha256(args.agent.read_bytes()).hexdigest(), 'engine_sha256': hashlib.sha256(args.engine.read_bytes()).hexdigest(), 'gpu': 'NVIDIA GeForce RTX 5090', 'endpoint_ready_s': ready, 'hydrated_cuda_seeds': bool(args.cache_seeds), 'completed_calls': len(samples), 'samples': samples, 'plan': ticket['plan']}
        (root / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        for row in samples[:2]:
            print(f"{row['model']}: launch-to-first-content={row['first_content_s']:.3f}s, single-call TTFT={row['ttft_s']:.3f}s", flush=True)
        print(f'Completed {len(samples)} calls; single/batched digests match per model.', flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        agent.terminate()
        try:
            agent.wait(timeout=10)
        except subprocess.TimeoutExpired:
            agent.kill()
            agent.wait()
        gpu.terminate()
        gpu.wait()
        gpu_log.close()
        log.close()
        source.shutdown()
        source.server_close()


if __name__ == '__main__':
    main()
