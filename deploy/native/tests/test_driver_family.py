"""Use an exact family seed or compile in an empty private driver cache."""

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

    def launch(self, identity, arguments=(), policy="strict"):
        return subprocess.run(
            [str(self.binary / "launch-engine-family.sh"), *arguments],
            env={
                **os.environ,
                "PATH": str(self.binary) + ":" + os.environ["PATH"],
                "NATIVE_ENGINE_CACHE_ROOT": str(self.cache),
                "NATIVE_ENGINE_EXPECTED_GPU": self.gpu,
                "NATIVE_ENGINE_EXPECTED_COMPUTE_CAP": "8.9",
                "NATIVE_ENGINE_CACHE_POLICY": policy,
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
                self.assertIn(b"native_cache_family_status=hit\n", result.stdout)

    def test_rejects_wrong_gpu_capability_and_invalid_driver_before_engine_exec(self):
        for identity in [
            "NVIDIA GeForce RTX 5090, 580.159.04, 8.9",
            self.gpu + ", 580.159.04, 12.0",
            self.gpu + ", 580.159.04/../other, 8.9",
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

    def test_opportunistic_missing_driver_compiles_in_exact_private_directory(self):
        seed = self.cache / "drivers/580.159.04/cuda/known-driver.cubin"
        seed.write_bytes(b"validated seed")
        arguments = ["--config-json", '{"model":"qwen3.5-4b"}', "serve"]
        result = self.launch(self.gpu + ", 580.178.04, 8.9", arguments, "opportunistic")
        self.assertEqual(result.returncode, 0, result.stderr)
        fields = self.result.read_bytes().split(b"\0")[:-1]
        cache = Path(fields[0].decode())
        self.assertEqual(cache, self.cache / "drivers/580.178.04/cuda")
        self.assertEqual(cache.stat().st_mode & 0o777, 0o700)
        self.assertEqual(list(cache.iterdir()), [])
        self.assertEqual([x.decode() for x in fields[1:]], arguments)
        self.assertEqual(
            (cache.parent / "identity.tsv").read_text(),
            self.gpu + "\t580.178.04\t8.9\n",
        )
        (cache / "new-driver.cubin").write_bytes(b"compiled here")
        self.assertIn(b"native_cache_family_status=miss\n", result.stdout)
        self.assertIn(
            b"native_cache_family_selected_driver=580.178.04\n", result.stdout
        )
        self.assertIn(
            ("native_cache_family_cuda_path=" + str(cache) + "\n").encode(),
            result.stdout,
        )
        again = self.launch(self.gpu + ", 580.178.04, 8.9", policy="opportunistic")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn(b"native_cache_family_status=hit\n", again.stdout)
        self.assertEqual((cache / "new-driver.cubin").read_bytes(), b"compiled here")
        self.assertEqual(seed.read_bytes(), b"validated seed")
        self.assertEqual(list(self.cache.parent.glob(".native-cuda-stage.*")), [])

    def test_incomplete_existing_component_is_not_treated_as_a_miss(self):
        variant = self.cache / "drivers/580.178.04"
        variant.mkdir()
        result = self.launch(self.gpu + ", 580.178.04, 8.9", policy="opportunistic")
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.result.exists())
        self.assertFalse((variant / "cuda").exists())

    def test_private_cache_parent_cannot_redirect_into_a_seed(self):
        shutil.rmtree(self.cache)
        self.cache.symlink_to(self.binary, target_is_directory=True)
        result = self.launch(self.gpu + ", 580.178.04, 8.9", policy="opportunistic")
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.result.exists())

    def test_missing_driver_remains_fatal_under_legacy_strict_policy(self):
        result = self.launch(self.gpu + ", 580.178.04, 8.9")
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.result.exists())
        self.assertFalse((self.cache / "drivers/580.178.04").exists())

    def test_opportunistic_policy_creates_entirely_absent_family_root(self):
        shutil.rmtree(self.cache)
        result = self.launch(self.gpu + ", 580.178.04, 8.9", policy="opportunistic")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.cache.stat().st_mode & 0o777, 0o700)
        self.assertTrue((self.cache / "drivers/580.178.04/cuda").is_dir())
        self.assertIn(b"native_cache_family_status=miss\n", result.stdout)

    def test_symlinks_in_any_selected_path_are_rejected(self):
        variant = self.cache / "drivers/580.159.04"
        for target in [
            variant / "cuda",
            variant / "identity.tsv",
            variant,
            self.cache / "drivers",
        ]:
            with self.subTest(target=target):
                saved = target.with_name(target.name + ".saved")
                target.rename(saved)
                target.symlink_to(saved, target_is_directory=saved.is_dir())
                for policy in ["strict", "opportunistic"]:
                    result = self.launch(self.gpu + ", 580.159.04, 8.9", policy=policy)
                    self.assertEqual(result.returncode, 78, result.stderr)
                    self.assertFalse(self.result.exists())
                target.unlink()
                saved.rename(target)

    def test_symlink_in_ancestor_is_rejected_before_cold_cache_creation(self):
        saved = self.root.with_name(self.root.name + ".saved")
        self.root.rename(saved)
        self.root.symlink_to(saved, target_is_directory=True)
        try:
            result = self.launch(self.gpu + ", 580.178.04, 8.9", policy="opportunistic")
            self.assertEqual(result.returncode, 78, result.stderr)
            self.assertFalse((self.cache / "drivers/580.178.04").exists())
        finally:
            self.root.unlink()
            saved.rename(self.root)

    def test_invalid_policy_is_rejected(self):
        result = self.launch(self.gpu + ", 580.159.04, 8.9", policy="automatic")
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.result.exists())


if __name__ == "__main__":
    unittest.main()
