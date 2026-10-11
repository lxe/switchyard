import json
import tempfile
import unittest
from pathlib import Path

from benchmark_artifacts import artifacts, named_directories, runtime_ticket


class LocalArtifactTests(unittest.TestCase):
    def test_components_keep_identity_compatibility_and_scoped_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"id": "a" * 64, "chunks": [{"name": "part.tar.zst"}]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            configuration = [
                {
                    "name": "build",
                    "artifact": {"id": manifest["id"]},
                    "compatibility": {"source_runtime": "selected"},
                }
            ]
            directories = named_directories([f"build={root}"])
            components = artifacts(configuration, directories, "component")
            runtime = {
                "program": "venv/bin/python",
                "args": [],
                "environment": {"CUDA_HOME": "{components_root}/build/cuda"},
                "compatibility": {"vllm": "0.31.0+cu129"},
            }
            boot = {"id": "b" * 64, "chunks": [{"name": "part.tar.zst"}]}
            ticket = runtime_ticket(runtime, boot, components, [], "http://fixture")
            component = ticket["components"][0]
            self.assertEqual(component["artifact"]["manifest"]["id"], "a" * 64)
            self.assertEqual(
                component["artifact"]["urls"]["part.tar.zst"],
                "http://fixture/components-build/part.tar.zst",
            )
            self.assertEqual(
                ticket["artifact"]["urls"]["part.tar.zst"],
                "http://fixture/runtime/part.tar.zst",
            )
            self.assertEqual(component["compatibility"], {"source_runtime": "selected"})
            ticket["environment"]["CUDA_HOME"] = "changed"
            self.assertEqual(
                runtime["environment"]["CUDA_HOME"], "{components_root}/build/cuda"
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                artifacts(
                    [{**configuration[0], "artifact": {"id": "c" * 64}}],
                    directories,
                    "component",
                )
            with self.assertRaisesRegex(ValueError, "differ"):
                artifacts(configuration, {}, "component")

    def test_no_supplements_preserves_legacy_ticket_shape(self):
        ticket = runtime_ticket(
            {"program": "python", "args": [], "environment": {}, "compatibility": {}},
            {"id": "a" * 64, "chunks": []},
            [],
            [],
            "http://fixture",
        )
        self.assertNotIn("components", ticket)
        self.assertEqual(ticket["caches"], [])
        for value in ["../escape=/tmp", "build=/tmp=unusual"]:
            if value.startswith(".."):
                with self.assertRaises(ValueError):
                    named_directories([value])
        with self.assertRaises(ValueError):
            named_directories(["build=/tmp", "build=/other"])


if __name__ == "__main__":
    unittest.main()
