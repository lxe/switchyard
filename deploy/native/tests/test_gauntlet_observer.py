"""Check selected-fleet credentials and actual observer subprocess ownership."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("gauntlet_observer", ROOT / "run-gauntlet.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


class ObserverTests(unittest.TestCase):
    def test_credentials_follow_cli_environment_and_first_file_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "one.env", root / "two.env"]
            paths[0].write_text("FIXTURE_PROVIDER_KEY=first\n")
            paths[1].write_text("FIXTURE_PROVIDER_KEY=second\n")
            fleet = {"compute": {"key_env": "FIXTURE_PROVIDER_KEY"}}
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(observer.credential(fleet, paths), "first")
            with patch.dict(os.environ, {"FIXTURE_PROVIDER_KEY": "environment"}):
                self.assertEqual(observer.credential(fleet, paths), "environment")

    def test_actual_child_failure_retains_identity_and_does_not_claim_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fixture-cli"
            binary.write_text("#!/usr/bin/env python3\nraise SystemExit(7)\n")
            binary.chmod(0o755)
            fleet = root / "fleet.json"
            fleet.write_text(json.dumps({"compute": {"api_url": "https://selected.invalid/v2",
                                                     "key_env": "FIXTURE_PROVIDER_KEY"}}))
            workloads = root / "workloads.json"
            workloads.write_text("{}")
            environment = root / "key.env"
            environment.write_text("FIXTURE_PROVIDER_KEY=fixture-private-value\n")
            output = root / "output"
            process = subprocess.run([sys.executable, str(ROOT / "run-gauntlet.py"),
                "--binary", str(binary), "--config", str(fleet), "--workloads", str(workloads),
                "--env-file", str(environment), "--out", str(output)],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 7)
            source = json.loads((output / "observer-source.json").read_text())
            self.assertEqual(source["provider_log_api"], "https://selected.invalid/v2")
            self.assertEqual(len(source["cli_sha256"]), 64)
            outcome = json.loads((output / "observer-outcome.json").read_text())
            self.assertEqual(outcome["returncode"], 7)
            self.assertEqual(outcome["observed_owned_pods"], [])
            self.assertNotIn("cleanup_confirmed", outcome)
            evidence = "".join(p.read_text() for p in output.glob("*.json"))
            self.assertNotIn("fixture-private-value", evidence + process.stdout + process.stderr)

    def test_latest_agent_phases_survive_partial_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.ndjson"
            path.write_text('partial\n{"pod":"one","state":{"phase":"runtime"}}\n'
                            '{"pod":"one","state":{"phase":"running"}}\n'
                            '{"pod":"two","status":"starting"}\n')
            self.assertEqual(observer.observation_phases(path), {"one": "running", "two": "starting"})


if __name__ == "__main__":
    unittest.main()
