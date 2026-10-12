"""Portable receipts must distinguish completion, throughput and observed cleanup."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "gauntlet_receipt", ROOT / "write-gauntlet-receipt.py"
)
receipt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receipt)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        (self.model / "workloads").mkdir(parents=True)
        self.save("observer-outcome.json", {"returncode": 0})
        self.save("observer-source.json", {"selected_model": "model"})
        self.save("run-source.json", {"source_commit": "checkout-only"})
        self.save(
            "fleet-source.json",
            {
                "compute": {"api_url": "https://fixture.invalid"},
                "bootstrap": {
                    "runtime": {"compatibility": {"engine_sha256": "pinned"}},
                    "image": "pinned-image",
                    "agent_sha256": "agent",
                },
                "limits": {"hydration_concurrency": 20},
                "models": [
                    {
                        "id": "model",
                        "revision": "revision",
                        "engine": {},
                        "archive_sha256": "weights",
                    }
                ],
            },
        )
        self.save("workloads.json", {"cases": [{"name": "single", "calls": 1}]})
        self.save("model/outcome.json", {"error": None, "cleanup_error": None})
        self.save("model/pending-launches.json", [{"resolved": True}])
        self.save(
            "model/owned-pods.json",
            [{"pod": {"id": "owned"}, "allocated_s": 1, "ended_s": 2, "quoted": {}}],
        )
        self.save("model/rental-receipt.json", {"estimated_rental_usd": 0.01})
        (self.model / "agent-observations.ndjson").write_text("")
        self.report = {
            "name": "single",
            "requests": 1,
            "completed": 1,
            "failed": 0,
            "unfinished": 0,
            "invalid_actions": 0,
            "budget_exceeded": False,
            "output_tokens_per_s": 10,
            "target_output_tokens_per_s": 200,
            "completions": [],
            "failures": {},
        }
        self.save("model/workloads/single.json", self.report)

    def save(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def summarize(self, absent=True):
        status = {
            "pod_id": "owned",
            "http_status": 404 if absent else 200,
            "confirmed_absent": absent,
        }
        with patch.object(receipt, "cleanup_status", return_value=status):
            return receipt.summarize(self.root, "private-key")

    def test_complete_run_retains_missed_target_without_claiming_compiled_commit(self):
        result = self.summarize()
        self.assertTrue(result["functional_workloads_passed"])
        self.assertFalse(result["models"][0]["workloads"][0]["throughput_target_met"])
        self.assertEqual(
            result["controller_identity"]["checkout_commit_at_launch"], "checkout-only"
        )
        self.assertIn(
            "not embedded", result["controller_identity"]["compiled_source_commit"]
        )

    def test_missing_workload_is_not_acceptance(self):
        (self.model / "workloads/single.json").unlink()
        self.assertFalse(self.summarize()["functional_workloads_passed"])

    def test_failed_or_unfinished_call_cannot_pass(self):
        for field in ("failed", "unfinished", "invalid_actions", "budget_exceeded"):
            with self.subTest(field=field):
                self.save("model/workloads/single.json", {**self.report, field: 1})
                self.assertFalse(self.summarize()["functional_workloads_passed"])

    def test_unresolved_allocation_or_existing_pod_requires_recovery(self):
        with self.assertRaisesRegex(ValueError, "Provider"):
            self.summarize(absent=False)
        self.save("model/pending-launches.json", [{"resolved": False}])
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            self.summarize()

    def test_report_must_match_selected_workload(self):
        self.save("model/workloads/single.json", {**self.report, "requests": 2})
        with self.assertRaisesRegex(ValueError, "differs"):
            self.summarize()

    def test_terminal_observer_is_required(self):
        (self.root / "observer-outcome.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.summarize()

    def test_raw_files_keep_the_pinned_layout_and_content_identity(self):
        fleet = json.loads((self.root / "fleet-source.json").read_text())
        model = fleet["models"][0]
        del model["archive_sha256"]
        reference = {"kind": "files", "prefix": "weights/raw", "id": "layout", "content_id": "content"}
        model["model_artifact"] = reference
        self.save("fleet-source.json", fleet)
        result = self.summarize()
        self.assertTrue(result["functional_workloads_passed"])
        self.assertEqual(result["models"][0]["model_artifact"], reference)
        self.assertIsNone(result["models"][0]["model_archive_sha256"])

    def test_legacy_observation_with_null_raw_transport_is_supported(self):
        (self.model / "agent-observations.ndjson").write_text(json.dumps({
            "pod": "owned", "state": {"phase": "ready", "model_transport": None}
        }) + "\n")
        result = self.summarize()
        self.assertTrue(result["functional_workloads_passed"])
        self.assertIsNone(result["models"][0]["pods"][0]["raw_model_transport"])

    def test_stopped_multimodel_run_reports_unexecuted_models(self):
        fleet = json.loads((self.root / "fleet-source.json").read_text())
        fleet["models"].append({"id": "later", "revision": "revision", "engine": {}, "archive_sha256": "other"})
        self.save("fleet-source.json", fleet)
        self.save("observer-source.json", {"selected_model": None})
        self.save("observer-outcome.json", {"returncode": 1})
        result = self.summarize()
        self.assertFalse(result["functional_workloads_passed"])
        later = result["models"][1]
        self.assertFalse(later["executed"])
        self.assertFalse(later["workloads"][0]["recorded"])
        self.assertEqual(later["cleanup"], [])

    def test_preallocation_failure_has_no_observations_or_owned_pods(self):
        self.save("model/owned-pods.json", [])
        self.save(
            "model/outcome.json",
            {"error": "allocation_rejected", "cleanup_error": None},
        )
        self.save("observer-outcome.json", {"returncode": 1})
        (self.model / "agent-observations.ndjson").unlink()
        (self.model / "workloads/single.json").unlink()
        result = self.summarize()
        self.assertFalse(result["functional_workloads_passed"])
        self.assertEqual(result["models"][0]["cleanup"], [])
        self.assertIsNone(result["models"][0]["evidence_sha256"]["observations"])


if __name__ == "__main__":
    unittest.main()
