"""A family launch must select the exact private seed before executing the engine."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class DriverFamilyLaunch(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binary = self.root / "bin"
        self.binary.mkdir()
        self.cache = self.root / "private-cache"
        self.result = self.root / "engine-executed"
        wrapper = Path(__file__).resolve().parents[1] / "launch-engine-family.sh"
        shutil.copy2(wrapper, self.binary / wrapper.name)
        for name, content in {
            "nvidia-smi": '#!/bin/bash\nprintf "%s\\n" "$TEST_GPU_IDENTITY"\n',
            "switchyard-engine": '#!/bin/bash\nprintf "%s\\0" "$CUDA_CACHE_PATH" "$@" > "$TEST_RESULT"\n',
        }.items():
            path = self.binary / name
            path.write_text(content)
            path.chmod(0o755)
        self.gpu = "NVIDIA GeForce RTX 4090"
        for driver in ["580.126.20", "580.159.04"]:
            variant = self.cache / "drivers" / driver
            (variant / "cuda").mkdir(parents=True)
            (variant / "identity.tsv").write_text(self.gpu + "\t" + driver + "\t8.9\n")

    def tearDown(self):
        self.temporary.cleanup()

    def launch(self, identity, arguments=()):
        return subprocess.run(
            [str(self.binary / "launch-engine-family.sh"), *arguments],
            env={
                **os.environ,
                "PATH": str(self.binary) + ":" + os.environ["PATH"],
                "NATIVE_ENGINE_CACHE_ROOT": str(self.cache),
                "NATIVE_ENGINE_EXPECTED_GPU": self.gpu,
                "NATIVE_ENGINE_EXPECTED_COMPUTE_CAP": "8.9",
                "TEST_GPU_IDENTITY": identity,
                "TEST_RESULT": str(self.result),
            },
            capture_output=True,
            check=False,
        )

    def test_selects_observed_driver_and_forwards_literal_configuration(self):
        arguments = [
            "--config-json",
            '{"literal":"$HOME $(touch injected)","model":"qwen3.5-2b"}',
            "serve",
            "--bind",
            "127.0.0.1:8000",
        ]
        for driver in ["580.126.20", "580.159.04"]:
            with self.subTest(driver=driver):
                result = self.launch(self.gpu + ", " + driver + ", 8.9", arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                fields = self.result.read_bytes().split(b"\0")[:-1]
                self.assertEqual(
                    fields[0].decode(), str(self.cache / "drivers" / driver / "cuda")
                )
                self.assertEqual([field.decode() for field in fields[1:]], arguments)
                self.assertFalse((self.root / "injected").exists())

    def test_rejects_missing_driver_gpu_and_capability_before_engine_exec(self):
        for identity in [
            self.gpu + ", 580.173.02, 8.9",
            "NVIDIA GeForce RTX 5090, 580.159.04, 8.9",
            self.gpu + ", 580.159.04, 12.0",
        ]:
            with self.subTest(identity=identity):
                result = self.launch(identity)
                self.assertEqual(result.returncode, 78, result.stderr)
                self.assertFalse(self.result.exists())

    def test_rejects_changed_component_marker_before_engine_exec(self):
        (self.cache / "drivers/580.159.04/identity.tsv").write_text(
            self.gpu + "\t580.126.20\t8.9\n"
        )
        result = self.launch(self.gpu + ", 580.159.04, 8.9")
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.result.exists())


if __name__ == "__main__":
    unittest.main()
