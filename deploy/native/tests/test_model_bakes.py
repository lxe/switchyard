"""Verify that offline artifact publication follows confirmed GPU cleanup."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

directory = Path(__file__).parents[1]
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location(
    "native_model_bakes", directory / "run-model-bakes.py"
)
bakes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bakes)


class ModelBakeLifecycleTests(unittest.TestCase):
    def fixture(self, *, cleanup_error=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        events = []

        def cleanup():
            events.append("cleanup")
            if cleanup_error:
                raise RuntimeError("owned pod termination is unconfirmed")
            (output / "cleanup.json").write_text('{"confirmed_absent": true}')

        return output, events, SimpleNamespace(output=output, cleanup=cleanup)

    def test_cleanup_and_receipt_copy_precede_publication(self):
        output, events, control = self.fixture()
        child = output / "model"
        child.mkdir()

        def operation(record):
            events.append("generation")
            record("component")

        def publish(component):
            self.assertEqual(component, "component")
            self.assertEqual(
                (child / "cleanup.json").read_text(), '{"confirmed_absent": true}'
            )
            events.append("publication")

        with patch.object(bakes, "publish_model", side_effect=publish):
            bakes.bake_and_publish(control, [child], operation)
        self.assertEqual(events, ["generation", "cleanup", "publication"])

    def test_generation_failure_cleans_up_and_skips_publication(self):
        _output, events, control = self.fixture()

        def operation(record):
            record("component")
            raise RuntimeError("native model bake failed")

        with patch.object(bakes, "publish_model") as publish:
            with self.assertRaisesRegex(RuntimeError, "native model bake failed"):
                bakes.bake_and_publish(control, [], operation)
            publish.assert_not_called()
        self.assertEqual(events, ["cleanup"])

    def test_unconfirmed_cleanup_prevents_publication(self):
        _output, events, control = self.fixture(cleanup_error=True)
        with patch.object(bakes, "publish_model") as publish:
            with self.assertRaisesRegex(RuntimeError, "termination is unconfirmed"):
                bakes.bake_and_publish(control, [], lambda record: record("component"))
            publish.assert_not_called()
        self.assertEqual(events, ["cleanup"])


if __name__ == "__main__":
    unittest.main()
