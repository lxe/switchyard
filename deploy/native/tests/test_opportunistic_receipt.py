"""A cold serving receipt must not inherit another driver's baked parity."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

directory = Path(__file__).parents[1]
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location(
    "native_receipt", directory / "write-receipt.py"
)
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)


class OpportunisticReceiptTests(unittest.TestCase):
    def fixture(self, root, *, stale_bake=False):
        config = {
            "engine": {"model": "tiny", "max_weight_bytes": 1024},
            "model_revision": "revision",
            "numerical_accuracy": "local evidence remains separate",
            "image": "image",
            "limits": {},
            "provider": {"min_ram_gb": 64, "min_vcpus": 8, "disk_gb": 30},
        }
        artifacts = {}
        for role in ["runtime", "model", "preparation"]:
            artifact = root / role
            artifact.mkdir()
            (artifact / "manifest.json").write_text('{"chunks": [], "entries": []}')
            artifacts[role] = {
                "directory": str(artifact),
                "id": role,
                "prefix": role,
                "compressed_bytes": 0,
            }
        prepared = {
            **artifacts,
            "cache": None,
            "cache_compatibility": None,
            "cache_policy": "opportunistic",
            "cache_family_status": "miss",
            "observed_driver": "580.178.04",
            "gpu_bake_completed": False,
        }
        if stale_bake:
            prepared["bake_receipt"] = str(root / "unrelated-bake/receipt.json")
        launch = {
            "allocation_started_wall_s": 1,
            "source_commit": "source",
            "engine_sha256": "engine",
            "pod": {"id": "owned"},
            "request": {},
            "frames": [],
            "workloads": None,
            "agent_listener_allocation_s": 1,
            "ready_allocation_s": 2,
            "first_content_allocation_s": 3,
            "request_to_first_content_s": 1,
        }
        logs = [
            {
                "line": "native_host_gpu=NVIDIA L40, GPU-id, 580.178.04, 8.9, 46068 MiB",
                "ts": 1,
                "source": "stdout",
            }
        ]
        for event in [
            "process_started",
            "device_initialized",
            "weights_read_finished",
            "model_loaded",
            "http_listening",
            "batch_finished",
        ]:
            item = {
                "event": event,
                "details": {"bytes": 50} if event == "weights_read_finished" else {},
            }
            logs.append({"line": json.dumps(item), "ts": 1, "source": "stdout"})
        for name, value in [
            ("config", config),
            ("build", {}),
            ("selected-prepared", prepared),
            ("receipt", launch),
            ("cleanup", {"confirmed_absent": True}),
            ("agent-state-final", {"phase": "ready", "inference_verified": False}),
        ]:
            (root / (name + ".json")).write_text(json.dumps(value))
        (root / "container-events.ndjson").write_text(
            "".join(json.dumps(x) + "\n" for x in logs)
        )
        (root / "controller-events.ndjson").write_text("")
        return [
            "write-receipt.py",
            "--config",
            str(root / "config.json"),
            "--build",
            str(root / "build.json"),
            "--launch",
            str(root),
            "--output",
            str(root / "portable.json"),
        ]

    def test_cold_receipt_records_miss_without_bake_or_cache_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = self.fixture(root)
            with (
                patch.object(sys, "argv", arguments),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                receipt.main()
            result = json.loads((root / "portable.json").read_text())
            self.assertIsNone(result["offline_bake"])
            self.assertIsNone(result["cache_compatibility"])
            self.assertFalse(result["numerical_parity_measured_in_cloud"])
            self.assertEqual(result["cache_family_status"], "miss")
            self.assertEqual(result["hardware"]["driver"], "580.178.04")
            self.assertNotIn("cache", [x["role"] for x in result["artifacts"]])

    def test_cacheless_receipt_rejects_stale_bake_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = self.fixture(root, stale_bake=True)
            with patch.object(sys, "argv", arguments), self.assertRaisesRegex(
                ValueError, "baked cache claims"
            ):
                receipt.main()
            self.assertFalse((root / "portable.json").exists())


if __name__ == "__main__":
    unittest.main()
