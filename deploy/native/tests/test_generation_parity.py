"""Exercise complete-sequence acceptance and retained cross-GPU failures."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from generation_parity import compare, validate_reference, verify


class GenerationParityTests(unittest.TestCase):
    def setUp(self):
        self.request = {"prompt": [3], "max_tokens": 2, "ignore_eos": True}
        self.row = {
            "token_ids": [10, 20],
            "prompt_tokens": 1,
            "finish_reason": "length",
        }
        self.reference = {
            "format": "switchyard-native-generation-reference-v1",
            "configuration": {"dtype": "f32"},
            "expected_cloud_build_commit": "source",
            "model_artifact_id": "weights",
            "model_revision": "revision",
            "weights_verified_against_artifact_manifest": True,
            "local_engine_sha256": "a" * 64,
            "requests": [self.request],
            "generated": [self.row],
        }
        self.receipt = {
            "config": {"dtype": "f32"},
            "requests": [self.request],
            "engine_sha256": "b" * 64,
        }

    def test_complete_ids_match_independently_of_timing(self):
        row = {**self.row, "elapsed_s": 99, "token_latency_s": [40, 50]}
        result = compare(self.reference, self.receipt, [row])
        self.assertTrue(result["passed"])
        self.assertEqual(result["exact_rows"], 1)

    def test_token_mismatch_preserves_complete_outputs_and_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "parity.json"
            with self.assertRaisesRegex(ValueError, "requires diagnosis"):
                verify(
                    self.reference,
                    self.receipt,
                    [{**self.row, "token_ids": [10, 21]}],
                    output,
                )
            result = json.loads(output.read_text())
            self.assertFalse(result["passed"])
            self.assertEqual(result["expected_token_ids"], [[10, 20]])
            self.assertEqual(result["actual_token_ids"], [[10, 21]])

    def test_missing_or_extra_rows_fail(self):
        for rows in [[], [self.row, self.row]]:
            with self.subTest(rows=len(rows)):
                result = compare(self.reference, self.receipt, rows)
                self.assertFalse(result["passed"])
                self.assertGreaterEqual(result["exact_rows"], 0)

    def test_configuration_and_request_changes_reject(self):
        for change in [
            {"config": {"dtype": "bf16"}},
            {"requests": [{**self.request, "max_tokens": 1}]},
        ]:
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "differ from local reference"),
            ):
                compare(self.reference, {**self.receipt, **change}, [self.row])

    def test_reference_requires_exact_artifact_and_full_output(self):
        config = {
            "engine": {"dtype": "f32"},
            "engine_source_commit": "source",
            "model_revision": "revision",
        }
        prepared = {"model": {"id": "weights"}}
        validate_reference(self.reference, config, prepared)
        with self.assertRaisesRegex(ValueError, "fixed model/build inputs"):
            validate_reference(self.reference, config, {"model": {"id": "other"}})
        truncated = copy.deepcopy(self.reference)
        truncated["generated"][0]["token_ids"] = [10]
        with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
            validate_reference(truncated, config, prepared)


if __name__ == "__main__":
    unittest.main()
