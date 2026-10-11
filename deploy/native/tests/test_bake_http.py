"""Exercise authentication and the hardware gate without allocating or using a GPU."""

import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


class BakeControlTests(unittest.TestCase):
    def fixture(self, driver, *, target=None, script_tail=""):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        fake = root / "nvidia-smi"
        fake.write_text(
            "#!/bin/sh\nprintf '%s\\n' 'NVIDIA GeForce RTX 4090, GPU-fixture, "
            + driver
            + ", 8.9, 24564 MiB'\n"
        )
        fake.chmod(0o700)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        settings = {
            "gpu": "NVIDIA GeForce RTX 4090",
            "capability": "8.9",
            "minimum_gpu_bytes": 20 * 1024**3,
            "excluded_drivers": ["580.159.04"],
            "driver_target": target,
            "models": ["qwen3.5-0.8b"],
            "bind": "127.0.0.1",
            "port": port,
        }
        script = (
            "touch '"
            + str(root / "started")
            + "'\nmkdir -p '"
            + str(root / "qwen3.5-0.8b")
            + "'\nprintf fixture > '"
            + str(root / "qwen3.5-0.8b/bake.tar.gz")
            + "'\n"
            + script_tail
        )
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).parents[1] / "bake_http_server.py")],
            env={
                **os.environ,
                "PATH": str(root) + os.pathsep + os.environ["PATH"],
                "SWITCHYARD_BAKE_TOKEN": "fixture-secret",
                "SWITCHYARD_BAKE_SETTINGS": json.dumps(settings),
                "SWITCHYARD_BAKE_SCRIPT": base64.b64encode(script.encode()).decode(),
                "SWITCHYARD_BAKE_OUTPUT": str(root),
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def stop():
            process.terminate()
            process.wait(timeout=5)

        self.addCleanup(stop)
        origin = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                status = self.request(origin, "/status")
                if status["phase"] in ["waiting", "failed"]:
                    return root, origin, status
            except urllib.error.URLError:
                pass
            time.sleep(0.02)
        self.fail("fixture control did not start")

    @staticmethod
    def request(origin, path, *, token="fixture-secret", method="GET", data=None):
        request = urllib.request.Request(
            origin + path,
            method=method,
            data=data,
            headers={"Authorization": "Bearer " + token},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            body = response.read()
            return json.loads(body) if path == "/status" else body

    def test_authentication_and_fixed_start_gate(self):
        root, origin, status = self.fixture("595.00.00")
        self.assertEqual(status["phase"], "waiting")
        self.assertFalse((root / "started").exists())
        for path, method, data, token, expected in [
            ("/status", "GET", None, "wrong", 401),
            ("/start", "POST", b"", "wrong", 401),
            ("/shell", "POST", b"", "fixture-secret", 404),
            ("/start", "POST", b"command", "fixture-secret", 400),
            ("/bundle/../started", "GET", None, "fixture-secret", 404),
        ]:
            with self.subTest(path=path, method=method, token=token):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.request(origin, path, method=method, data=data, token=token)
                self.assertEqual(error.exception.code, expected)
                error.exception.close()
        self.assertFalse((root / "started").exists())
        self.request(origin, "/start", method="POST", data=b"")
        deadline = time.monotonic() + 3
        while (
            time.monotonic() < deadline
            and self.request(origin, "/status")["phase"] != "complete"
        ):
            time.sleep(0.02)
        self.assertTrue((root / "started").exists())
        self.assertEqual(self.request(origin, "/bundle/qwen3.5-0.8b"), b"fixture")

    def test_covered_driver_never_runs_bake(self):
        root, origin, status = self.fixture("580.159.04")
        self.assertEqual(status["phase"], "failed")
        self.assertIn("excluded by bake platform policy", status["error"])
        self.assertFalse((root / "started").exists())
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(origin, "/start", method="POST", data=b"")
        self.assertEqual(error.exception.code, 409)
        error.exception.close()

    def test_target_mismatch_never_runs_bake(self):
        root, _origin, status = self.fixture("595.00.00", target="580.173.02")
        self.assertEqual(status["phase"], "failed")
        self.assertIn("differs from explicit bake target", status["error"])
        self.assertFalse((root / "started").exists())

    def test_failed_bake_retains_sanitized_stderr(self):
        _root, origin, _status = self.fixture(
            "595.00.00",
            script_tail="printf '%s\\n' 'https://example.invalid/object?secret=value fixture-secret failure' >&2\nexit 7\n",
        )
        self.request(origin, "/start", method="POST", data=b"")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = self.request(origin, "/status")
            if status["phase"] == "failed":
                break
            time.sleep(0.02)
        self.assertEqual(status["exit_code"], 7)
        self.assertEqual(self.request(origin, "/log"), b"[url] [redacted] failure\n")


if __name__ == "__main__":
    unittest.main()
