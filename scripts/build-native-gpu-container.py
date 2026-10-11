#!/usr/bin/env python3
"""Compile a native architecture matrix in the pinned Ubuntu CUDA builder."""

import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path

DEFAULT_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d"
SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "native_gpu_job", SCRIPTS / "build-native-gpu.py"
)
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)
BASE_LIBRARIES = {
    "libc.so.6",
    "libm.so.6",
    "libpthread.so.0",
    "libdl.so.2",
    "librt.so.1",
    "libresolv.so.2",
    "libutil.so.1",
}
DRIVER_LIBRARIES = {"libcuda.so.1", "libnvidia-ml.so.1"}


def image_pin(value):
    if not re.fullmatch(
        r"nvidia/cuda:13\.0\.0-devel-ubuntu24\.04@sha256:[0-9a-f]{64}", value
    ):
        raise argparse.ArgumentTypeError(
            "use a digest-pinned nvidia/cuda:13.0.0-devel-ubuntu24.04 image"
        )
    return value


def command(command, *, cwd=None):
    return subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=True
    ).stdout


def source_revision(workspace):
    paths = ["Cargo.toml", "Cargo.lock", ".cargo", "crates"]
    dirty = command(["git", "diff", "--name-only", "HEAD", "--", *paths], cwd=workspace)
    untracked = command(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
        cwd=workspace,
    )
    if dirty.strip() or untracked.strip():
        raise ValueError(
            "commit the final Cargo/crate inputs before snapshotting the build source"
        )
    return command(["git", "rev-parse", "HEAD"], cwd=workspace).strip()


def docker_command(args, source, tools, revision, name):
    return [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        name,
        "--platform",
        "linux/amd64",
        "--cpus",
        str(args.jobs),
        "--memory",
        args.memory,
        "--mount",
        f"type=bind,source={source},target=/source,readonly",
        "--mount",
        f"type=bind,source={tools},target=/tools,readonly",
        "--mount",
        f"type=bind,source={args.cache_root},target=/build",
        "--mount",
        f"type=bind,source={args.out},target=/output",
        "--env",
        "CUDA_VISIBLE_DEVICES=",
        "--env",
        "NVIDIA_VISIBLE_DEVICES=void",
        "--env",
        "CUDA_CACHE_DISABLE=1",
        "--workdir",
        "/source",
        "--entrypoint",
        "/bin/bash",
        args.image,
        "/tools/build-native-gpu-container.sh",
        str(args.jobs),
        revision,
        args.image,
        *[argument for arch in args.arch for argument in ("--arch", arch)],
    ]


def runtime_libraries(text, cuda_root):
    libraries = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[1] != "=>":
            continue
        name = parts[0]
        if name in DRIVER_LIBRARIES:
            continue
        if parts[2:4] == ["not", "found"]:
            raise ValueError(f"missing runtime dependency: {name}")
        if not parts[2].startswith("/"):
            raise ValueError(f"unrecognized runtime dependency: {line}")
        path = Path(parts[2]).resolve(strict=True)
        if path.is_relative_to(cuda_root.resolve()) or name in {
            "libstdc++.so.6",
            "libgcc_s.so.1",
        }:
            previous = libraries.get(name)
            if previous is not None and previous != path:
                raise ValueError(f"conflicting runtime library: {name}")
            libraries[name] = path
        elif name not in BASE_LIBRARIES:
            raise ValueError(
                f"runtime packager needs an explicit dependency policy for {name}"
            )
    return libraries


def assemble_matrix(
    matrix,
    source_root,
    revision,
    image,
    evidence,
    cuda_root=Path("/usr/local/cuda"),
    *,
    runner=command,
):
    descriptor = json.loads((matrix / "build.json").read_text())
    identity = descriptor.pop("artifact_id")
    if (
        descriptor.get("format") != "switchyard-native-gpu-build-v1"
        or JOB.digest(descriptor) != identity
    ):
        raise ValueError("invalid architecture matrix receipt")
    environment = {
        **os.environ,
        "LD_LIBRARY_PATH": f"{cuda_root}/lib64:{cuda_root}/targets/x86_64-linux/lib",
    }
    for selected in descriptor["targets"]:
        target = matrix / selected["path"]
        receipt = json.loads((target / "build.json").read_text())
        target_id = receipt.pop("artifact_id")
        if (
            receipt.get("format") != "switchyard-native-gpu-target-v1"
            or JOB.digest(receipt) != target_id
            or target_id != selected["artifact_id"]
        ):
            raise ValueError("invalid target receipt")
        compatibility = receipt["compatibility"]
        if (
            compatibility["source_commit"] != revision
            or compatibility["builder_identity"] != image
        ):
            raise ValueError(
                "target source or builder differs from the container invocation"
            )
        if (
            JOB.file_info(matrix / "producer.py")["sha256"]
            != compatibility["build_job_sha256"]
        ):
            raise ValueError("retained producer differs from target receipt")
        for name, expected in receipt["files"].items():
            if JOB.file_info(target / name) != expected:
                raise ValueError(f"exported file differs from receipt: {name}")
        runtime = target / "runtime"
        (runtime / "tree/bin").mkdir(parents=True)
        (runtime / "tree/lib").mkdir()
        libraries = {}
        for name in ("switchyard-engine", "switchyard-engine-bake"):
            binary = target / "bin" / name
            shutil.copy2(binary, runtime / "tree/bin" / name)
            # ldd observes the trusted compiled ELF loader; the binary's main
            # and CUDA code are never executed. Driver libraries stay external.
            observed = runner(
                [
                    "env",
                    f"LD_LIBRARY_PATH={environment['LD_LIBRARY_PATH']}",
                    "ldd",
                    str(binary),
                ]
            )
            (runtime / f"{name}.ldd").write_text(observed)
            for library, path in runtime_libraries(observed, cuda_root).items():
                if library in libraries and libraries[library] != path:
                    raise ValueError(
                        f"conflicting runtime library across binaries: {library}"
                    )
                libraries[library] = path
        for name, path in libraries.items():
            shutil.copy2(path, runtime / "tree/lib" / name)
        # Use the existing packager's receipt format and physical capability
        # spelling. The matrix independently retains the exact a/f target.
        cap = selected["architecture"].removeprefix("sm_").rstrip("af")
        runner(
            [
                sys.executable,
                str(source_root / "deploy/native/runtime-receipt.py"),
                "--tree",
                str(runtime / "tree"),
                "--build",
                str(runtime / "build.json"),
                "--source-commit",
                revision,
                "--builder",
                image,
                "--compute-cap",
                cap,
                "--compiler-target",
                selected["architecture"].replace("sm_", "compute_", 1),
                "--evidence",
                str(evidence),
            ]
        )


def publish_matrix(output):
    matrix = output / "matrix"
    for child in sorted(matrix.iterdir()):
        if child.name != "build.json":
            child.rename(output / child.name)
    # Preserve the existing matrix identity, publishing its completion marker
    # only after the Ubuntu runtime bridge has completed successfully.
    (matrix / "build.json").rename(output / "build.json")
    matrix.rmdir()


def build(args):
    args.workspace = args.workspace.resolve(strict=True)
    args.out = args.out.absolute()
    args.cache_root = (
        args.cache_root or args.workspace / "target/native-gpu-container"
    ).absolute()
    if (
        args.out.exists()
        or args.out.is_symlink()
        or len(set(args.arch)) != len(args.arch)
    ):
        raise ValueError("a fresh output and unique architectures are required")
    if args.out.is_relative_to(args.cache_root) or args.cache_root.is_relative_to(
        args.out
    ):
        raise ValueError("output and compiler cache must be disjoint")
    for path in (args.out, args.cache_root):
        if "," in str(path):
            raise ValueError("Docker bind paths cannot contain commas")
    revision = source_revision(args.workspace)
    name = "switchyard-native-matrix-" + uuid.uuid4().hex
    if args.plan:
        print(
            json.dumps(
                {
                    "source_commit": revision,
                    "command": docker_command(
                        args,
                        "/host/source-snapshot",
                        "/host/build-tools",
                        revision,
                        name,
                    ),
                },
                indent=2,
            )
        )
        return
    args.out.mkdir(parents=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    tools = args.out / "container-inputs"
    tools.mkdir()
    for name in (
        "build-native-gpu.py",
        "build-native-gpu-container.py",
        "build-native-gpu-container.sh",
    ):
        shutil.copy2(SCRIPTS / name, tools / name)
    try:
        with tempfile.TemporaryDirectory(
            prefix="switchyard-native-source-"
        ) as temporary:
            temporary = Path(temporary)
            archive = temporary / "source.tar"
            subprocess.run(
                ["git", "archive", "--format=tar", f"--output={archive}", revision],
                cwd=args.workspace,
                check=True,
            )
            source = temporary / "source"
            source.mkdir()
            with tarfile.open(archive) as files:
                files.extractall(source, filter="data")
            container = "switchyard-native-matrix-" + uuid.uuid4().hex
            try:
                with (args.out / "container.log").open("w") as log:
                    subprocess.run(
                        docker_command(args, source, tools, revision, container),
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            finally:
                subprocess.run(
                    ["docker", "rm", "--force", container],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        publish_matrix(args.out)
    except BaseException as error:
        JOB.write_json(
            args.out / "container-failure.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise


def main():
    def terminated(_signal, _frame):
        raise KeyboardInterrupt("container build terminated")

    signal.signal(signal.SIGTERM, terminated)
    parser = argparse.ArgumentParser(description=__doc__)
    if len(sys.argv) > 1 and sys.argv[1] == "assemble":
        parser.add_argument("matrix", type=Path)
        parser.add_argument("--source-root", type=Path, required=True)
        parser.add_argument("--source-commit", required=True)
        parser.add_argument("--image", type=image_pin, required=True)
        parser.add_argument("--evidence", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        assemble_matrix(
            args.matrix, args.source_root, args.source_commit, args.image, args.evidence
        )
        return 0
    parser.add_argument("--arch", type=JOB.architecture, action="append", required=True)
    parser.add_argument("--workspace", type=Path, default=SCRIPTS.parent)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--image", type=image_pin, default=DEFAULT_IMAGE)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        build(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"native Ubuntu build: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("native Ubuntu build cancelled", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
