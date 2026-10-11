"""CPU-only container command and existing runtime-packager contract fixtures."""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "container_builder", SCRIPTS / "build-native-gpu-container.py"
)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


class ContainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_pinned_cpu_container_has_bounded_readonly_source_mounts(self):
        args = argparse.Namespace(
            jobs=2,
            memory="8g",
            cache_root=self.root / "cache",
            out=self.root / "out",
            image=wrapper.DEFAULT_IMAGE,
            arch=["89", "120a"],
        )
        command = wrapper.docker_command(
            args,
            self.root / "source",
            self.root / "tools",
            "a" * 40,
            "owned-test-container",
        )
        self.assertEqual(
            wrapper.image_pin(wrapper.DEFAULT_IMAGE), wrapper.DEFAULT_IMAGE
        )
        self.assertIn("linux/amd64", command)
        self.assertIn(
            f"type=bind,source={self.root}/source,target=/source,readonly", command
        )
        self.assertIn(
            f"type=bind,source={self.root}/tools,target=/tools,readonly", command
        )
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--privileged", command)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", command)
        self.assertEqual(command[-4:], ["--arch", "89", "--arch", "120a"])
        for image in (
            "nvidia/cuda:13.0.0-devel-ubuntu24.04",
            "nvidia/cuda:13.4.0-devel-ubuntu24.04@sha256:" + "a" * 64,
        ):
            with self.assertRaises(argparse.ArgumentTypeError):
                wrapper.image_pin(image)

    def test_snapshot_rejects_uncommitted_rust_inputs(self):
        def git(*args):
            return subprocess.run(
                ["git", *args],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        git("init", "-q")
        git("config", "user.name", "CPU fixture")
        git("config", "user.email", "fixture@example.invalid")
        (self.root / "Cargo.toml").write_text("[workspace]\n")
        (self.root / "crates").mkdir()
        source = self.root / "crates/model.rs"
        source.write_text("// committed source\n")
        git("add", ".")
        git("commit", "-qm", "fixture")
        self.assertEqual(
            wrapper.source_revision(self.root), git("rev-parse", "HEAD").strip()
        )
        source.write_text("// pending decode source\n")
        with self.assertRaisesRegex(ValueError, "commit the final"):
            wrapper.source_revision(self.root)
        git("restore", "crates/model.rs")
        (self.root / "crates/probe.rs").write_text("// untracked source\n")
        with self.assertRaisesRegex(ValueError, "commit the final"):
            wrapper.source_revision(self.root)

    def libraries(self):
        cuda = self.root / "cuda"
        (cuda / "lib64").mkdir(parents=True)
        runtime = cuda / "lib64/libcudart.so.13"
        runtime.write_bytes(b"toolkit library")
        cpp = self.root / "libstdc++.so.6"
        cpp.write_bytes(b"Ubuntu C++ runtime")
        return (
            cuda,
            f"libcudart.so.13 => {runtime} (0x1)\nlibstdc++.so.6 => {cpp} (0x2)\nlibcuda.so.1 => not found\n",
        )

    def evidence(self):
        evidence = self.root / "container-evidence"
        evidence.mkdir(exist_ok=True)
        for name, contents in {
            "nvcc-gpu-arch.txt": "compute_89\ncompute_120\n",
            "cuda-environment-version.txt": "13.0.0\n",
            "cuda-packages.txt": "cuda-nvcc-13-0\t13.0.88-1\n",
            "nvcc.txt": "Cuda compilation tools, release 13.0, V13.0.88\n",
            "os-release.txt": 'ID=ubuntu\nVERSION_ID="24.04"\n',
            "architecture.txt": "x86_64\n",
        }.items():
            (evidence / name).write_text(contents)
        return evidence

    def test_driver_stubs_stay_external_and_missing_runtime_libraries_fail(self):
        cuda, text = self.libraries()
        selected = wrapper.runtime_libraries(text, cuda)
        self.assertEqual(set(selected), {"libcudart.so.13", "libstdc++.so.6"})
        with self.assertRaisesRegex(ValueError, "missing runtime dependency"):
            wrapper.runtime_libraries(text + "libcublas.so.13 => not found\n", cuda)

    def receipt_command(self, evidence, *extra):
        tree = self.root / "receipt-tree"
        tree.mkdir(exist_ok=True)
        (tree / "fixture").write_bytes(b"CPU runtime inventory fixture")
        return [
            sys.executable,
            str(SCRIPTS.parent / "deploy/native/runtime-receipt.py"),
            "--tree",
            str(tree),
            "--build",
            str(self.root / "runtime-build.json"),
            "--source-commit",
            "a" * 40,
            "--builder",
            wrapper.DEFAULT_IMAGE,
            "--compute-cap",
            "89",
            "--evidence",
            str(evidence),
            *extra,
        ]

    def test_receipt_numeric_default_remains_compatible(self):
        wrapper.command(self.receipt_command(self.evidence()))
        receipt = json.loads((self.root / "runtime-build.json").read_text())
        self.assertEqual(receipt["cuda_compiler_target"], "compute_89")
        self.assertEqual(receipt["gpu_compute_capability"], "8.9")

    def test_receipt_rejects_missing_evidence_and_inconsistent_exact_target(self):
        evidence = self.evidence()
        for path in sorted(evidence.iterdir()):
            with self.subTest(missing=path.name):
                original = path.read_bytes()
                path.unlink()
                with self.assertRaises(subprocess.CalledProcessError):
                    wrapper.command(self.receipt_command(evidence))
                self.assertFalse((self.root / "runtime-build.json").exists())
                path.write_bytes(original)
        for target in ("compute_120a", "compute_89a", "sm_89", "compute_89z"):
            with self.subTest(compiler_target=target):
                with self.assertRaises(subprocess.CalledProcessError):
                    wrapper.command(
                        self.receipt_command(evidence, "--compiler-target", target)
                    )
                self.assertFalse((self.root / "runtime-build.json").exists())

    def matrix(self):
        matrix = self.root / "output/matrix"
        target = matrix / "targets/sm_120a"
        (target / "bin").mkdir(parents=True)
        files = {}
        for name in wrapper.JOB.EXPORTED_BINARIES:
            binary = target / "bin" / name
            binary.write_bytes(b"CPU fixture executable")
            files[f"bin/{name}"] = wrapper.JOB.file_info(binary)
        for name in (
            "kernels/ptx/fixture.ptx",
            "kernels/cubin/fixture.cubin",
            "kernels/libmoe.a",
        ):
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"CPU fixture kernel export")
            files[name] = wrapper.JOB.file_info(path)
        producer = matrix / "producer.py"
        producer.write_bytes(wrapper.JOB.BUILD_JOB_SOURCE)
        receipt = {
            "format": "switchyard-native-gpu-target-v1",
            "compatibility": {
                "source_commit": "a" * 40,
                "builder_identity": wrapper.DEFAULT_IMAGE,
                "build_job_sha256": wrapper.JOB.file_info(producer)["sha256"],
            },
            "files": files,
        }
        receipt["artifact_id"] = wrapper.JOB.digest(receipt)
        wrapper.JOB.write_json(target / "build.json", receipt)
        descriptor = {
            "format": "switchyard-native-gpu-build-v1",
            "targets": [
                {
                    "path": "targets/sm_120a",
                    "architecture": "sm_120a",
                    "artifact_id": receipt["artifact_id"],
                }
            ],
            "gpu_execution_validated": False,
        }
        descriptor["artifact_id"] = wrapper.JOB.digest(descriptor)
        wrapper.JOB.write_json(matrix / "build.json", descriptor)
        return matrix, target

    def test_runtime_bridge_uses_existing_receipt_and_packager_layout(self):
        matrix, target = self.matrix()
        cuda, text = self.libraries()

        def runner(command):
            if command[0] == "env":
                return text
            return wrapper.command(command)

        wrapper.assemble_matrix(
            matrix,
            SCRIPTS.parent,
            "a" * 40,
            wrapper.DEFAULT_IMAGE,
            self.evidence(),
            cuda,
            runner=runner,
        )
        runtime = target / "runtime"
        build = json.loads((runtime / "build.json").read_text())
        self.assertEqual(build["format"], "switchyard-native-runtime-build-v1")
        self.assertEqual(build["gpu_compute_capability"], "12.0")
        self.assertEqual(build["cuda_compiler_target"], "compute_120a")
        self.assertEqual(build["cuda_build"], "13.0.0")
        self.assertEqual(build["cuda_compiler_version"], "13.0.88")
        self.assertEqual(
            build["cuda_package_versions"], {"cuda-nvcc-13-0": "13.0.88-1"}
        )
        self.assertEqual(build["source_commit"], "a" * 40)
        self.assertEqual(
            {p["path"] for p in build["files"]},
            {
                "bin/switchyard-engine",
                "bin/switchyard-engine-bake",
                "lib/libcudart.so.13",
                "lib/libstdc++.so.6",
            },
        )
        self.assertFalse((runtime / "tree/bin/switchyard-engine-bench").exists())
        self.assertFalse((runtime / "tree/bin/switchyard-engine-merged-probe").exists())
        self.assertFalse((runtime / "tree/bin/switchyard-engine-mixed-bench").exists())
        self.assertFalse((runtime / "tree/lib/libcuda.so.1").exists())

    def test_modified_export_fails_before_a_runtime_receipt_is_created(self):
        matrix, target = self.matrix()
        for name in (
            "bin/switchyard-engine",
            "bin/switchyard-engine-bake",
            "bin/switchyard-engine-bench",
            "kernels/ptx/fixture.ptx",
            "kernels/cubin/fixture.cubin",
            "kernels/libmoe.a",
        ):
            with self.subTest(name=name):
                path = target / name
                original = path.read_bytes()
                path.write_bytes(b"changed after compiler receipt")
                with self.assertRaisesRegex(ValueError, "exported file differs"):
                    wrapper.assemble_matrix(
                        matrix,
                        SCRIPTS.parent,
                        "a" * 40,
                        wrapper.DEFAULT_IMAGE,
                        self.evidence(),
                    )
                self.assertFalse((target / "runtime/build.json").exists())
                self.assertFalse((matrix.parent / "build.json").exists())
                path.write_bytes(original)

    def test_modified_producer_fails_before_runtime_assembly(self):
        matrix, target = self.matrix()
        (matrix / "producer.py").write_text("different producer")
        with self.assertRaisesRegex(ValueError, "retained producer differs"):
            wrapper.assemble_matrix(
                matrix, SCRIPTS.parent, "a" * 40, wrapper.DEFAULT_IMAGE, self.evidence()
            )
        self.assertFalse((target / "runtime").exists())

    def test_publication_preserves_the_original_matrix_identity(self):
        matrix, _ = self.matrix()
        original = (matrix / "build.json").read_bytes()
        wrapper.publish_matrix(matrix.parent)
        self.assertEqual((matrix.parent / "build.json").read_bytes(), original)
        self.assertTrue((matrix.parent / "targets/sm_120a/build.json").exists())
        self.assertTrue((matrix.parent / "producer.py").exists())
        self.assertFalse(matrix.exists())


if __name__ == "__main__":
    unittest.main()
