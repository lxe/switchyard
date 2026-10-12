"""Exercise transient proxy failure and bounded idempotent fixed-start retries."""

import io
import json
import sys
import tarfile
import tempfile
import time
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))
import http_model_bakes


class Response(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}


class GateRetryTests(unittest.TestCase):
    def run_fixture(self, *, persistent):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        token = "fixture-secret"
        model = "qwen3.5-0.8b"
        events = []
        attempts = []
        completed = False
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            file = tarfile.TarInfo("bake.json")
            file.size = 2
            tar.addfile(file, io.BytesIO(b"{}"))
        bundle = archive.getvalue()

        def urlopen(request, **_kwargs):
            nonlocal completed
            if request.full_url.endswith("/status"):
                return Response(
                    json.dumps(
                        {
                            "phase": "complete" if completed else "waiting",
                            "hardware": "NVIDIA GeForce RTX 4090, GPU-fixture, 580.159.04, 8.9, 24564 MiB",
                            "completed": [model] if completed else [],
                        }
                    ).encode()
                )
            if request.full_url.endswith("/start"):
                attempts.append(request)
                if persistent or len(attempts) == 1:
                    headers = Message()
                    headers["Server"] = "fixture-edge"
                    raise urllib.error.HTTPError(
                        request.full_url,
                        404,
                        "not found",
                        headers,
                        io.BytesIO(
                            b"https://example.invalid/?signature=value fixture-secret"
                        ),
                    )
                completed = True
                return Response(b"", 204)
            if request.full_url.endswith("/bundle/" + model):
                return Response(bundle)
            raise AssertionError(request.full_url)

        control = SimpleNamespace(
            event=lambda stage, **fields: events.append((stage, fields))
        )
        record = Mock()
        arguments = (
            SimpleNamespace(output=output, tool=Path("fixture-tool")),
            [{"engine": {"model": model}}],
            [{}],
            {},
            control,
            SimpleNamespace(tarfile=tarfile),
            [],
            record,
            {"id": "fixture"},
            token,
            time.monotonic() + 10,
        )
        with (
            patch.object(
                http_model_bakes.urllib.request, "urlopen", side_effect=urlopen
            ),
            patch.object(http_model_bakes.time, "sleep"),
        ):
            if persistent:
                with self.assertRaisesRegex(RuntimeError, "retained gate response"):
                    http_model_bakes.consume(*arguments)
            else:
                http_model_bakes.consume(*arguments)
        return output, attempts, events, record

    def test_transient_start_404_retries_without_changing_fixed_operation(self):
        output, attempts, events, record = self.run_fixture(persistent=False)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(
            all(
                request.get_method() == "POST" and request.data == b""
                for request in attempts
            )
        )
        self.assertEqual(events[1][0], "fixed_gate_retry")
        self.assertEqual(record.call_count, 1)
        diagnostic = json.loads((output / "gate-rejection.json").read_text())
        self.assertEqual(diagnostic["body"], "[url] [redacted]")
        self.assertEqual(diagnostic["server"], "fixture-edge")

    def test_persistent_start_404_has_ten_retries_and_no_download(self):
        _output, attempts, events, record = self.run_fixture(persistent=True)
        self.assertEqual(len(attempts), 11)
        self.assertEqual(
            sum(stage == "fixed_gate_retry" for stage, _fields in events), 10
        )
        record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
