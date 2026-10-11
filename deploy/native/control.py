"""Local provider and object-store control. Secrets are never receipt fields."""

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def env_file(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if match:
            values[match[1]] = match[2].strip().strip("\"'")
    return values


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


class Control:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.started = time.monotonic()
        self.key = env_file(config["provider"]["key_file"])[
            config["provider"]["key_env"]
        ]
        self.owned = None
        self.name = None
        secrets = env_file(config["storage"]["env_file"])
        self.aws_env = {
            **os.environ,
            "AWS_ACCESS_KEY_ID": secrets["PODTRAIN_R2_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": secrets["PODTRAIN_R2_SECRET_ACCESS_KEY"],
            "AWS_DEFAULT_REGION": "auto",
            "AWS_EC2_METADATA_DISABLED": "true",
        }
        transfer_config = self.output / "aws-transfer.conf"
        transfer_config.write_text(
            "[default]\nregion = auto\ns3 =\n"
            "    max_concurrent_requests = 10\n"
            "    max_queue_size = 20\n"
            "    multipart_threshold = 16MB\n"
            "    multipart_chunksize = 16MB\n"
            "    preferred_transfer_client = classic\n"
        )
        self.aws_env["AWS_CONFIG_FILE"] = str(transfer_config.resolve())

    def event(self, stage, **fields):
        item = {"stage": stage, "elapsed_s": time.monotonic() - self.started, **fields}
        with (self.output / "controller-events.ndjson").open("a") as target:
            target.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)

    def api(self, path, method="GET", body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.config["provider"]["api_url"] + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.key,
                "User-Agent": "Switchyard/0.5",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
            return json.loads(content) if content else None

    def upload(self, path, key):
        store = self.config["storage"]
        subprocess.run(
            [
                "aws",
                "--endpoint-url",
                store["endpoint"],
                "s3",
                "cp",
                str(path),
                "s3://" + store["bucket"] + "/" + key,
                "--only-show-errors",
            ],
            env=self.aws_env,
            check=True,
        )

    def upload_immutable(self, path, key, expected_sha256):
        """Check existing bytes before publication in this single-writer namespace."""
        store = self.config["storage"]
        head = subprocess.run(
            [
                "aws",
                "--endpoint-url",
                store["endpoint"],
                "s3api",
                "head-object",
                "--bucket",
                store["bucket"],
                "--key",
                key,
            ],
            env=self.aws_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode == 0:
            metadata = json.loads(head.stdout)
            if metadata["ContentLength"] != Path(path).stat().st_size:
                raise ValueError("existing artifact object has a different size")
            digest = hashlib.sha256()
            with urllib.request.urlopen(self.url(key), timeout=600) as source:
                while part := source.read(65536):
                    digest.update(part)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("existing artifact object has different bytes")
            return
        if "(404)" not in head.stderr and "(NotFound)" not in head.stderr:
            raise RuntimeError("artifact existence check failed")
        self.upload(path, key)

    def url(self, key):
        store = self.config["storage"]
        return subprocess.check_output(
            [
                "aws",
                "--endpoint-url",
                store["endpoint"],
                "s3",
                "presign",
                "s3://" + store["bucket"] + "/" + key,
                "--expires-in",
                str(store["url_lifetime_s"]),
            ],
            env=self.aws_env,
            text=True,
        ).strip()

    def put_json(self, value, key):
        store = self.config["storage"]
        subprocess.run(
            [
                "aws",
                "--endpoint-url",
                store["endpoint"],
                "s3",
                "cp",
                "-",
                "s3://" + store["bucket"] + "/" + key,
                "--content-type",
                "application/json",
                "--only-show-errors",
            ],
            input=json.dumps(value).encode(),
            env=self.aws_env,
            check=True,
        )

    def remove_object(self, key):
        store = self.config["storage"]
        subprocess.run(
            [
                "aws",
                "--endpoint-url",
                store["endpoint"],
                "s3",
                "rm",
                "s3://" + store["bucket"] + "/" + key,
                "--only-show-errors",
            ],
            env=self.aws_env,
            check=True,
        )

    def allocate(self, name, entrypoint, command, environment, ssh=False):
        if self.owned is not None:
            raise RuntimeError("an owned pod is still allocated")
        provider = self.config["provider"]
        self.name = name
        body = {
            "name": name,
            "image": self.config["image"],
            "gpu": {
                "id": provider["gpu_id"],
                "count": 1,
                "minCudaVersion": provider["min_cuda"],
                "minRamPerGpu": provider["min_ram_gb"],
                "minVcpuCountPerGpu": provider["min_vcpus"],
            },
            "cloud": provider["cloud"],
            "disk": provider["disk_gb"],
            "dataCenterIds": provider["data_centers"],
            "entrypoint": entrypoint,
            "cmd": command,
            "ports": ["22/tcp"] if ssh else ["8080/http"],
            "startSsh": ssh,
            "env": environment,
        }
        if provider.get("allowed_cuda_versions"):
            body["gpu"].pop("minCudaVersion")
            body["gpu"]["allowedCudaVersions"] = provider["allowed_cuda_versions"]
        self.event("allocation_started", name=name)
        try:
            pod = self.api("/pods", "POST", body)
        except urllib.error.HTTPError as error:
            if error.code in (400, 401, 403):
                # Retain useful rejection diagnostics, scrubbing every submitted
                # environment value as well as URLs before writing a receipt.
                detail = error.read(65536).decode("utf8", errors="replace")
                for value in sorted(environment.values(), key=len, reverse=True):
                    if value:
                        detail = detail.replace(value, "[redacted]")
                detail = re.sub(r"https?://[^\s\"'<>]+", "[url]", detail)
                write(
                    self.output / "allocation-rejected.json",
                    {
                        "http_status": error.code,
                        "detail": detail,
                    },
                )
                raise RuntimeError(
                    "RunPod allocation rejected: HTTP " + str(error.code)
                ) from None
            raise
        except (urllib.error.URLError, TimeoutError):
            rows = self.api("/pods")["pods"]
            found = [row for row in rows if row["name"] == name]
            if len(found) != 1:
                raise RuntimeError(
                    "allocation outcome unresolved; inspect owned name before retry"
                ) from None
            pod = found[0]
        self.owned = pod["id"]
        summary = {
            key: pod.get(key)
            for key in ["id", "name", "status", "cost", "dataCenterId", "cudaVersion"]
        }
        write(self.output / "owned-pod.json", summary)
        if pod["cost"] > self.config["limits"]["max_hourly_usd"]:
            raise RuntimeError("allocated hourly rate exceeds fixed limit")
        self.event("allocation_accepted", **summary)
        return pod

    def cleanup(self):
        if self.owned is None:
            return
        pod_id = self.owned
        try:
            pod = self.api("/pods/" + pod_id)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            write(
                self.output / "cleanup.json",
                {"pod_id": pod_id, "name": self.name, "confirmed_absent": True},
            )
            self.owned = None
            return
        if pod["name"] != self.name:
            raise RuntimeError("owned pod name changed; cleanup refused")
        self.api("/pods/" + pod_id, "DELETE")
        absent = False
        for _ in range(12):
            try:
                self.api("/pods/" + pod_id)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    absent = True
                    break
            time.sleep(1)
        write(
            self.output / "cleanup.json",
            {"pod_id": pod_id, "name": self.name, "confirmed_absent": absent},
        )
        self.event("cleanup", pod_id=pod_id, confirmed_absent=absent)
        if not absent:
            raise RuntimeError("owned pod termination is unconfirmed")
        self.owned = None
