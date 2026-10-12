"""Owned native PEFT serving test: real adapters, useful calls, exact base isolation."""
import argparse
import concurrent.futures
import functools
import hashlib
import http.client
import http.server
import json
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request


def post(endpoint, path, body, expected=200):
    request = urllib.request.Request(endpoint + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code != expected:
            raise
        return json.load(error)
    if response.status != expected:
        raise RuntimeError(f"Expected HTTP {expected}, got {response.status}")
    with response:
        return json.load(response)


def upload(bind, name, source):
    boundary = "switchyard-peft-upload"
    parts = []
    for field, filename in [("adapter_config", "adapter_config.json"), ("adapter_model", "adapter_model.safetensors")]:
        header = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
                  "Content-Type: application/octet-stream\r\n\r\n").encode()
        parts.append((header, source / filename))
    ending = f"--{boundary}--\r\n".encode()
    length = len(ending) + sum(len(header) + path.stat().st_size + 2 for header, path in parts)
    connection = http.client.HTTPConnection(bind, timeout=120)
    try:
        connection.putrequest("POST", "/v1/adapters/" + name)
        connection.putheader("Content-Type", "multipart/form-data; boundary=" + boundary)
        connection.putheader("Content-Length", str(length))
        connection.endheaders()
        for header, path in parts:
            connection.send(header)
            with path.open("rb") as source:
                while chunk := source.read(256 * 1024):
                    connection.send(chunk)
            connection.send(b"\r\n")
        connection.send(ending)
        response = connection.getresponse()
        content = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(content)
        return content
    finally:
        connection.close()


def stream(endpoint, model, message):
    body = {"model":model,"messages":[{"role":"user","content":message}],"max_tokens":16,
            "temperature":0,"stream":True,"stream_options":{"include_usage":True}}
    request = urllib.request.Request(endpoint + "/v1/chat/completions",data=json.dumps(body).encode(),
                                     headers={"Content-Type":"application/json"})
    tokens, first, usage, done = [], None, None, False
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event)
            if event["model"] != model:
                raise RuntimeError("Stream lost its adapter alias")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices",[]):
                tokens.extend(choice.get("token_ids",[]))
                if choice.get("delta",{}).get("content") and first is None:
                    first = time.monotonic()-started
    if not done or not tokens or first is None or usage["completion_tokens"] != len(tokens):
        raise RuntimeError("Incomplete adapter SSE response")
    return {"token_ids":tokens,"request_to_first_content_s":first,"usage":usage}


class QuietFiles(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1:18211")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    # Copy into the owned private tree; do not mutate a training run's adapter.
    local = args.out / "adapters" / "pocket"
    local.mkdir(parents=True)
    for name in ["adapter_config.json","adapter_model.safetensors"]:
        shutil.copyfile(args.adapter/name, local/name)
    config["lora"]["root"] = str(local.parent.resolve())
    config_path = args.out/"config.json"
    config_path.write_text(json.dumps(config,indent=2))
    endpoint = "http://"+args.bind
    source = http.server.ThreadingHTTPServer(("127.0.0.1",0),functools.partial(QuietFiles,directory=str(local.resolve())))
    serving = threading.Thread(target=source.serve_forever,daemon=True)
    serving.start()
    process = None
    receipt = {"scope":"Local native LoRA HTTP/SSE with a real trained PEFT adapter; no cloud acceptance",
               "binary_sha256":hashlib.sha256(args.binary.read_bytes()).hexdigest(),"model":config["model"],
               "adapter_model_sha256":hashlib.sha256((local/'adapter_model.safetensors').read_bytes()).hexdigest(),
               "adapter_config_sha256":hashlib.sha256((local/'adapter_config.json').read_bytes()).hexdigest()}
    messages = ["A customer says their package arrived damaged. What should support do?",
                "Reply politely to a customer asking for a refund.",
                "The tracking number has not updated for three days. Give a short useful reply.",
                "Summarize the next step for a delayed order."]
    log_path = args.out/'engine.ndjson'
    try:
        with log_path.open('w') as log:
            process = subprocess.Popen([str(args.binary.resolve()),'--config',str(config_path.resolve()),'serve','--bind',args.bind],stdout=log,stderr=subprocess.STDOUT)
            deadline = time.monotonic()+90
            while '"http_listening"' not in log_path.read_text():
                if process.poll() is not None:
                    raise RuntimeError('Engine stopped; see engine.ndjson')
                if time.monotonic()>deadline:
                    raise TimeoutError('Listener event deadline')
                time.sleep(.02)
            requests = []
            for message in messages:
                tokens = post(endpoint,'/tokenize',{"model":config['model'],"messages":[{"role":"user","content":message}],"add_generation_prompt":True})['tokens']
                requests.append({"prompt":tokens,"max_tokens":16,"ignore_eos":True})
            def complete(model, rows=requests):
                return post(endpoint,'/v1/completions',{"model":model,"prompt":[r['prompt'] for r in rows],"max_tokens":16,"ignore_eos":True})
            print('phase=base_inference',flush=True)
            base_before = complete(config['model'])
            name = config['model']+'+pocket'
            start = time.monotonic()
            loaded = post(endpoint,'/v1/load_lora_adapter',{"lora_name":name,"lora_path":str(local.resolve())})
            receipt['local_load'] = {**loaded,'elapsed_s':time.monotonic()-start}
            print('phase=adapter_inference',flush=True)
            adapted = complete(name)
            if adapted['model'] != name:
                raise RuntimeError('Completion lost its adapter alias')
            adapted_repeat = complete(name)
            base_after = complete(config['model'])
            ids = lambda value: [r['token_ids'] for r in value['choices']]
            if ids(base_before)!=ids(base_after) or ids(adapted)!=ids(adapted_repeat):
                raise RuntimeError('Base isolation or repeated-adapter cache check failed')
            receipt['base_unchanged'] = True
            receipt['adapter_repeat_equal'] = True
            receipt['adapter_changed_generation'] = ids(base_before)!=ids(adapted)
            receipt['requests'] = requests
            receipt['base_token_ids'] = ids(base_before)
            receipt['adapter_token_ids'] = ids(adapted)
            receipt['sse'] = stream(endpoint,name,messages[0])
            if receipt['sse']['token_ids'] != ids(adapted)[0][:len(receipt['sse']['token_ids'])]:
                raise RuntimeError('SSE adapter result differs from nonstream result')
            # Distinct aliases may queue together; each must select its own engine.
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                jobs=[(config['model'] if i%2==0 else name) for i in range(8)]
                outputs=list(pool.map(lambda model:complete(model,requests[:1]),jobs))
            for model,output in zip(jobs,outputs):
                expected=ids(base_before)[0] if model==config['model'] else ids(adapted)[0]
                if ids(output)[0]!=expected:
                    raise RuntimeError('Concurrent adapter/base routing contaminated state')
            receipt['concurrent_alias_calls']=len(jobs)
            print('phase=multipart_upload',flush=True)
            second_name=name+'-upload'
            receipt['multipart_load']=upload(args.bind,second_name,local)
            if receipt['multipart_load']['revision']!=loaded['revision']:
                raise RuntimeError('Multipart transport changed adapter identity')
            if ids(complete(second_name,requests[:1]))[0]!=ids(adapted)[0]:
                raise RuntimeError('Uploaded adapter result differs')
            post(endpoint,'/v1/load_lora_adapter',{"lora_name":name,"lora_path":str(local.resolve())},expected=409)
            receipt['duplicate_rejected']=True
            with urllib.request.urlopen(endpoint+'/v1/models') as response:
                listed=json.load(response)
            if {config['model'],name,second_name}!={m['id'] for m in listed['data']}:
                raise RuntimeError('Adapter model catalog incomplete')
            post(endpoint,'/v1/unload_lora_adapter',{"lora_name":second_name})
            post(endpoint,'/v1/unload_lora_adapter',{"lora_int_id":loaded['lora_int_id']})
            post(endpoint,'/v1/completions',{"model":name,"prompt":[1,2],"max_tokens":1},expected=404)
            print('phase=remote_stream_load',flush=True)
            start=time.monotonic()
            remote=post(endpoint,'/v1/load_lora_adapter',{"lora_name":name,"adapter_config":json.loads((local/'adapter_config.json').read_text()),"adapter_url":f'http://127.0.0.1:{source.server_port}/adapter_model.safetensors'})
            receipt['remote_load']={**remote,'elapsed_s':time.monotonic()-start}
            if remote['revision']!=loaded['revision']:
                raise RuntimeError('JSON transport changed adapter identity')
            receipt['transport_revision_equal']=True
            post(endpoint,'/v1/unload_lora_adapter',{"lora_int_id":loaded['lora_int_id']},expected=404)
            if ids(complete(name,requests[:1]))[0]!=ids(adapted)[0]:
                raise RuntimeError('Remote adapter or stale-ID isolation failed')
            post(endpoint,'/v1/unload_lora_adapter',{"lora_name":name})
            receipt['stale_id_rejected']=True
            receipt['unloaded']=True
            if list(local.parent.glob('.upload-*')):
                raise RuntimeError('Upload staging directory leaked')
            receipt['upload_staging_clean']=True
            print('phase=passed',flush=True)
    finally:
        source.shutdown()
        source.server_close()
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        receipt['owned_engine_returncode']=None if process is None else process.returncode
        (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
