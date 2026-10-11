"""CPU fixtures for architecture isolation and incomplete-build publication."""

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "native_gpu_build", Path(__file__).parents[1] / "build-native-gpu.py"
)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class CompilerFixture:
    """Fake command boundary; file selection and receipts use real filesystem IO."""

    def __init__(
        self, root, wrong_arch=None, fail=False, include_bins=False, extra_bins=()
    ):
        self.root = root
        self.wrong_arch = wrong_arch
        self.fail = fail
        self.include_bins = include_bins
        self.extra_bins = extra_bins
        self.commands = []
        self.target_dirs = []

    def __call__(self, command, *, cwd, env, log=None):
        self.commands.append(command)
        arch = env["CUDA_COMPUTE_CAP"]
        actual = self.wrong_arch or arch
        if command[0] == "cargo":
            if self.fail:
                raise RuntimeError("compiler failed")
            out = Path(env["CARGO_TARGET_DIR"]) / "out"
            self.target_dirs.append(out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "unary.ptx").write_text(
                f".version 8.8\n.target sm_{actual}\n.address_size 64\n"
            )
            (out / "libmoe.a").write_bytes(b"!<arch>\nfixture")
            messages = [
                {
                    "reason": "build-script-executed",
                    "package_id": "candle-kernels",
                    "out_dir": str(out),
                }
            ]
            if self.include_bins:
                for name in (
                    "switchyard-engine",
                    "switchyard-engine-bake",
                    "switchyard-engine-bench",
                    *self.extra_bins,
                ):
                    executable = out / name
                    executable.write_bytes(b"\x7fELFfixture")
                    messages.append(
                        {
                            "reason": "compiler-artifact",
                            "package_id": "switchyard-engine",
                            "target": {"name": name, "kind": ["bin"]},
                            "executable": str(executable),
                        }
                    )
            return "\n".join(json.dumps(m) for m in messages)
        if command[0] == "ptxas":
            Path(command[-1]).write_bytes(b"\x7fELFfixture")
            return ""
        if command[0] == "cuobjdump":
            return f"ELF file 1: kernel.sm_{actual}.cubin\n"
        if command[0] == "readelf":
            return "Shared library: [libcudart.so.13]\nName: GLIBC_2.39\n"
        raise AssertionError(command)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = argparse.Namespace(
            out=self.root / "published",
            build_root=self.root / "cargo",
            workspace=self.root,
            kernels_only=True,
            offline=True,
            no_cubins=False,
        )
        self.args.out.mkdir()
        self.base = {
            "backend": "candle-cuda",
            "rust_target": "x86_64-unknown-linux-gnu",
            "source_inputs_sha256": "source-a",
        }
        self.metadata = {
            "packages": [{"id": "candle-kernels", "name": "candle-kernels"}]
        }
        self.env = {"NATIVE_PTXAS": "ptxas", "NATIVE_CUOBJDUMP": "cuobjdump"}

    def test_one_invocation_produces_distinct_targets_without_executing_them(self):
        compiler = CompilerFixture(self.root)
        receipts = [
            build.build_target(
                self.args, arch, self.base, self.metadata, self.env, runner=compiler
            )
            for arch in ("89", "120a")
        ]
        self.assertNotEqual(
            receipts[0]["compatibility_key"], receipts[1]["compatibility_key"]
        )
        self.assertNotEqual(compiler.target_dirs[0], compiler.target_dirs[1])
        for target in receipts:
            receipt = json.loads(
                (self.args.out / target["path"] / "build.json").read_text()
            )
            self.assertEqual(
                set(receipt["files"]),
                {
                    "kernels/ptx/unary.ptx",
                    "kernels/cubin/unary.cubin",
                    "kernels/libmoe.a",
                },
            )
            self.assertFalse(receipt["gpu_execution_validated"])
            self.assertFalse(receipt["driver_jit_cache_included"])
            self.assertFalse(receipt["model_bake_included"])
        self.assertEqual(
            {c[0] for c in compiler.commands}, {"cargo", "ptxas", "cuobjdump"}
        )

    def test_mixed_architecture_ptx_cannot_publish_a_target_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "PTX target differs"):
            build.build_target(
                self.args,
                "89",
                self.base,
                self.metadata,
                self.env,
                runner=CompilerFixture(self.root, wrong_arch="120a"),
            )
        self.assertFalse((self.args.out / "targets/sm_89/build.json").exists())

    def test_wrong_device_archive_is_rejected_even_with_matching_ptx(self):
        compiler = CompilerFixture(self.root)

        def wrong_archive(command, **kwargs):
            if command[0] == "cuobjdump":
                return "ELF file 1: other.sm_86.cubin\n"
            return compiler(command, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "device objects differ"):
            build.build_target(
                self.args,
                "89",
                self.base,
                self.metadata,
                self.env,
                runner=wrong_archive,
            )
        self.assertFalse((self.args.out / "targets/sm_89/build.json").exists())

    def test_architecture_in_diagnostic_path_is_not_device_code(self):
        compiler = CompilerFixture(self.root)

        def no_device_code(command, **kwargs):
            if command[0] == "cuobjdump":
                return "File: /output/targets/sm_89/host-only.o\n"
            return compiler(command, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "device objects differ"):
            build.build_target(
                self.args,
                "89",
                self.base,
                self.metadata,
                self.env,
                runner=no_device_code,
            )

    def test_compiler_failure_cannot_publish_a_target_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "compiler failed"):
            build.build_target(
                self.args,
                "89",
                self.base,
                self.metadata,
                self.env,
                runner=CompilerFixture(self.root, fail=True),
            )
        self.assertFalse((self.args.out / "targets/sm_89/build.json").exists())

    def test_newer_architectures_require_an_explicit_suffix(self):
        for value in ("native", "all", "8.9", "89;120a", "120", "90"):
            with self.assertRaises(argparse.ArgumentTypeError):
                build.architecture(value)
        for value in ("80", "89", "90a", "120a", "121f"):
            self.assertEqual(build.architecture(value), value)

    def test_source_or_toolchain_changes_invalidate_the_build_cache_key(self):
        compiler = CompilerFixture(self.root)
        first = build.build_target(
            self.args, "89", self.base, self.metadata, self.env, runner=compiler
        )
        self.args.out = self.root / "second"
        self.args.out.mkdir()
        second = build.build_target(
            self.args,
            "89",
            {**self.base, "source_inputs_sha256": "source-b"},
            self.metadata,
            self.env,
            runner=compiler,
        )
        self.assertNotEqual(first["compatibility_key"], second["compatibility_key"])

    def test_failed_second_target_has_no_complete_matrix_receipt(self):
        self.args.out = self.root / "matrix"
        self.args.arch = ["89", "120a"]
        compiler = CompilerFixture(self.root)

        def fail_second(command, **kwargs):
            if command[0] == "cargo" and kwargs["env"]["CUDA_COMPUTE_CAP"] == "120a":
                raise RuntimeError("second target failed")
            return compiler(command, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "second target failed"):
            build.build_matrix(
                self.args, self.base, self.metadata, self.env, runner=fail_second
            )
        self.assertFalse((self.args.out / "build.json").exists())
        self.assertTrue((self.args.out / "targets/sm_89/build.json").exists())
        self.assertTrue((self.args.out / "failure.json").exists())

    def test_existing_output_is_preserved(self):
        owned = self.args.out / "keep.txt"
        owned.write_text("existing output")
        self.args.arch = ["89"]
        with self.assertRaises(FileExistsError):
            build.build_matrix(
                self.args,
                self.base,
                self.metadata,
                self.env,
                runner=CompilerFixture(self.root),
            )
        self.assertEqual(owned.read_text(), "existing output")
        self.assertFalse((self.args.out / "producer.py").exists())

    def test_default_mode_exports_binaries_and_abi_requirements_without_running_them(
        self,
    ):
        self.args.kernels_only = False
        self.metadata["packages"].append(
            {"id": "switchyard-engine", "name": "switchyard-engine"}
        )
        compiler = CompilerFixture(self.root, include_bins=True)
        output = build.build_target(
            self.args, "89", self.base, self.metadata, self.env, runner=compiler
        )
        receipt = json.loads(
            (self.args.out / output["path"] / "build.json").read_text()
        )
        self.assertEqual(
            set(receipt["runtime_requirements"]),
            {"switchyard-engine", "switchyard-engine-bake", "switchyard-engine-bench"},
        )
        self.assertEqual(
            receipt["runtime_requirements"]["switchyard-engine"]["symbol_versions"],
            ["GLIBC_2.39"],
        )
        self.assertEqual(
            {c[0] for c in compiler.commands},
            {"cargo", "ptxas", "cuobjdump", "readelf"},
        )

    def test_commit_label_change_reuses_compiler_outputs_but_changes_receipt(self):
        compiler = CompilerFixture(self.root)
        first = build.build_target(
            self.args,
            "89",
            {**self.base, "source_commit": "commit-a"},
            self.metadata,
            self.env,
            runner=compiler,
        )
        self.args.out = self.root / "next-commit"
        self.args.out.mkdir()
        second = build.build_target(
            self.args,
            "89",
            {**self.base, "source_commit": "commit-b"},
            self.metadata,
            self.env,
            runner=compiler,
        )
        self.assertNotEqual(first["compatibility_key"], second["compatibility_key"])
        self.assertEqual(compiler.target_dirs[0], compiler.target_dirs[1])

    def test_offline_diagnostics_are_not_exported_with_runtime_binaries(self):
        self.args.kernels_only = False
        self.metadata["packages"].append(
            {"id": "switchyard-engine", "name": "switchyard-engine"}
        )
        compiler = CompilerFixture(
            self.root, include_bins=True, extra_bins=("switchyard-engine-merged-probe",)
        )
        output = build.build_target(
            self.args, "89", self.base, self.metadata, self.env, runner=compiler
        )
        target = self.args.out / output["path"]
        receipt = json.loads((target / "build.json").read_text())
        self.assertEqual(
            receipt["excluded_binary_targets"], ["switchyard-engine-merged-probe"]
        )
        self.assertFalse((target / "bin/switchyard-engine-merged-probe").exists())
        self.assertFalse(any("merged-probe" in path for path in receipt["files"]))
        self.assertEqual(set(receipt["runtime_requirements"]), build.EXPORTED_BINARIES)
        self.assertEqual(
            {path.name for path in (target / "bin").iterdir()},
            build.EXPORTED_BINARIES,
        )
        for name, expected in receipt["files"].items():
            self.assertEqual(build.file_info(target / name), expected)


if __name__ == "__main__":
    unittest.main()
