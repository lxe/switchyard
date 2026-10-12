"""Keep strict driver selection and explicitly permit an unbaked native launch."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

directory = Path(__file__).parents[1]
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location(
    "native_launch", directory / "run-launch.py"
)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LaunchCachePolicyTests(unittest.TestCase):
    def fixture(self):
        config = {"engine": {"model": "tiny"}, "provider": {"gpu_id": "NVIDIA L40"}}
        compatibility = {"engine_sha256": "frozen", "gpu_compute_capability": "8.9"}
        prepared = {
            "runtime": {"id": "old"},
            "model": {"id": "weights"},
            "compatibility": {
                **compatibility,
                "native_config_sha256": hashlib.sha256(
                    (json.dumps(config["engine"], indent=2) + "\n").encode()
                ).hexdigest(),
            },
            "gpu_bake_completed": True,
            "shared_runtime": {"compatibility": copy.deepcopy(compatibility)},
            "bake_receipt": "prior-bake",
            "cache": {"id": "old-seed"},
            "cache_compatibility": {"driver_version": "580.159.04"},
        }
        shared = {
            "runtime": {"id": "new"},
            "compatibility": compatibility,
            "launch_configuration": {
                "artifact": {"id": "new"},
                "program": "bin/launch-engine.sh",
                "environment": {
                    "NATIVE_ENGINE_CACHE_POLICY": "opportunistic",
                    "NATIVE_ENGINE_EXPECTED_GPU": "NVIDIA L40",
                    "NATIVE_ENGINE_EXPECTED_COMPUTE_CAP": "8.9",
                },
            },
        }
        return config, prepared, shared

    def test_cacheless_selection_drops_old_cache_and_bake_claims(self):
        config, prepared, shared = self.fixture()
        before = copy.deepcopy(prepared)
        selected = launch.cacheless_prepared(config, prepared, shared)
        self.assertEqual(prepared, before)
        self.assertEqual(selected["runtime"]["id"], "new")
        self.assertEqual(selected["model"], before["model"])
        self.assertIsNone(selected["cache"])
        self.assertIsNone(selected["cache_compatibility"])
        self.assertFalse(selected["gpu_bake_completed"])
        self.assertNotIn("bake_receipt", selected)

    def test_cacheless_selection_rejects_changed_engine_config_gpu_or_policy(self):
        for mutation in ["config", "engine", "gpu", "policy", "artifact"]:
            with self.subTest(mutation=mutation):
                config, prepared, shared = self.fixture()
                if mutation == "config":
                    config["engine"]["model"] = "other"
                elif mutation == "engine":
                    shared["compatibility"]["engine_sha256"] = "different"
                elif mutation == "artifact":
                    shared["launch_configuration"]["artifact"]["id"] = "different"
                else:
                    key = (
                        "NATIVE_ENGINE_EXPECTED_GPU"
                        if mutation == "gpu"
                        else "NATIVE_ENGINE_CACHE_POLICY"
                    )
                    shared["launch_configuration"]["environment"][key] = "different"
                with self.assertRaises(ValueError):
                    launch.cacheless_prepared(config, prepared, shared)

    def test_strict_unknown_driver_stops_before_agent_download(self):
        selector = launch.driver_selector({"580.159.04": {}}, family=True)
        result = subprocess.run(
            ["bash", "-c", selector + "printf agent_downloaded"],
            env={**os.environ, "native_driver": "580.178.04"},
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 78)
        self.assertNotIn(b"agent_downloaded", result.stdout)

    def test_opportunistic_unknown_driver_uses_cacheless_ticket(self):
        selector = launch.driver_selector({}, opportunistic=True)
        result = subprocess.run(
            ["bash", "-c", selector + 'printf "%s" "$SWITCHYARD_TICKET_URL"'],
            env={
                **os.environ,
                "native_driver": "580.178.04",
                "NATIVE_CACHE_TICKET_URL_0": "cacheless",
            },
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"cacheless")

    def test_direct_cacheless_route_constructs_ticket_without_touching_provider(self):
        config, prepared, shared = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config.update(
                storage={"prefix": "test"},
                model_revision="revision",
                limits={"download_concurrency": 4, "pod_timeout_s": 60},
            )
            prepared["agent"] = {"key": "agent"}
            prepared["model"]["prefix"] = "model"
            prepared["legacy_manifest"] = str(root / "legacy.json")
            (root / "legacy.json").write_text('{"chunks": []}')
            for name, value in [
                ("config", config),
                ("prepared", prepared),
                ("shared", shared),
            ]:
                (root / (name + ".json")).write_text(json.dumps(value))
            captured = {}

            class Control:
                def __init__(self, _config, _output):
                    pass

                def url(self, key):
                    return key

                def put_json(self, ticket, _key):
                    captured["ticket"] = ticket

                def allocate(self, _name, _entrypoint, command, _environment):
                    captured["bootstrap"] = command[0]
                    raise RuntimeError("fixture stops before provider call")

                def cleanup(self):
                    captured["cleanup"] = True

                def remove_object(self, _key):
                    captured["ticket_removed"] = True

            arguments = [
                "run-launch.py",
                "--config",
                str(root / "config.json"),
                "--prepared",
                str(root / "prepared.json"),
                "--shared-runtime",
                str(root / "shared.json"),
                "--cache-policy",
                "opportunistic",
                "--output",
                str(root / "output"),
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(launch, "Control", Control),
                patch.object(
                    launch,
                    "ticket_artifact",
                    side_effect=lambda _control, reference: reference,
                ),
                self.assertRaisesRegex(RuntimeError, "fixture stops"),
            ):
                launch.main()
            self.assertEqual(captured["ticket"]["runtime"]["caches"], [])
            self.assertEqual(captured["ticket"]["engine_args"][0], "--config-json")
            self.assertNotIn("driver_mismatch", captured["bootstrap"])
            self.assertTrue(captured["cleanup"])
            self.assertTrue(captured["ticket_removed"])
            subprocess.run(
                ["bash", "-n"], input=captured["bootstrap"].encode(), check=True
            )


if __name__ == "__main__":
    unittest.main()
