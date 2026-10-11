#!/usr/bin/env python3
"""Build pinned Candle CUDA artifacts for explicit targets, without a GPU."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

BUILD_JOB_SOURCE = Path(__file__).read_bytes()
EXPORTED_BINARIES = {
    "switchyard-engine",
    "switchyard-engine-bake",
    "switchyard-engine-bench",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_info(path):
    with path.open("rb") as source:
        sha = hashlib.file_digest(source, "sha256").hexdigest()
    return {"sha256": sha, "bytes": path.stat().st_size}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def architecture(value):
    if not re.fullmatch(r"[1-9][0-9]{1,2}[af]?", value):
        raise argparse.ArgumentTypeError("use a CUDA target such as 89, 90a, or 120a")
    if int(value.rstrip("af")) >= 90 and value[-1].isdigit():
        raise argparse.ArgumentTypeError(
            "Candle auto-suffixes targets >=90; select the suffix explicitly (e.g. 90a, 120a, 121f)"
        )
    if int(value.rstrip("af")) < 90 and not value[-1].isdigit():
        raise argparse.ArgumentTypeError("architecture suffixes require a target >=90")
    return value


def run(command, *, cwd, env, log=None):
    if log is None:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    else:
        # Keep compiler diagnostics readable while a long build is running.
        with log.open("w") as diagnostics:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
                check=False,
            )
            diagnostics.write(result.stdout)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command[0]}"
            + (f"; see {log}" if log else f"\n{result.stderr}")
        )
    return result.stdout


def tool(path, cwd, env):
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        **file_info(path),
        "version": run([str(path), "--version"], cwd=cwd, env=env).strip(),
    }


def source_inputs(workspace, metadata):
    # Hash actual engine inputs, including untracked source files. Cargo.lock
    # binds registry dependencies; the pinned Candle source is hashed as well.
    inputs = {
        name: file_info(workspace / name)["sha256"]
        for name in ("Cargo.toml", "Cargo.lock")
    }
    roots = [("engine", workspace / "crates/switchyard-engine")]
    for package in metadata["packages"]:
        if package["name"] in {"candle-core", "candle-nn", "candle-kernels"}:
            roots.append((package["name"], Path(package["manifest_path"]).parent))
    for name, root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not any(
                part in {"target", ".git", "__pycache__"}
                for part in path.relative_to(root).parts
            ):
                inputs[f"{name}/{path.relative_to(root)}"] = file_info(path)["sha256"]
    return digest(inputs)


def cargo_configuration(workspace):
    paths = {
        parent / ".cargo" / name
        for parent in (workspace, *workspace.parents)
        for name in ("config", "config.toml")
    }
    home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    paths.update(home / name for name in ("config", "config.toml"))
    return {
        str(path): file_info(path)["sha256"] for path in sorted(paths) if path.is_file()
    }


def ptx_target(path, arch):
    text = path.read_text()
    target = re.search(r"^\s*\.target\s+(sm_[0-9]+[af]?)\b", text, re.MULTILINE)
    version = re.search(r"^\s*\.version\s+([0-9]+\.[0-9]+)\b", text, re.MULTILINE)
    if target is None or target[1] != f"sm_{arch}" or version is None:
        raise RuntimeError(f"PTX target differs from sm_{arch}: {path}")
    return version[1]


def inspect_device(path, arch, *, cwd, env, runner=run):
    output = runner(
        [env["NATIVE_CUOBJDUMP"], "--list-elf", str(path)], cwd=cwd, env=env
    )
    entries = re.findall(r"^ELF file\s+[0-9]+:\s+(.+)$", output, re.MULTILINE)
    found = {
        arch for entry in entries for arch in re.findall(r"\bsm_[0-9]+[af]?\b", entry)
    }
    if found != {f"sm_{arch}"}:
        raise RuntimeError(
            f"device objects differ from sm_{arch}: {path}: {sorted(found)}"
        )
    return output


def collect_kernels(out_dir, destination, arch, *, cwd, env, cubins, runner=run):
    ptx_files = sorted(out_dir.glob("*.ptx"))
    archive = out_dir / "libmoe.a"
    magic = b""
    if archive.is_file():
        with archive.open("rb") as stream:
            magic = stream.read(8)
    if not ptx_files or magic != b"!<arch>\n":
        raise RuntimeError("Candle build did not produce PTX and libmoe.a")
    ptx = destination / "ptx"
    ptx.mkdir(parents=True)
    versions = set()
    for source in ptx_files:
        versions.add(ptx_target(source, arch))
        shutil.copy2(source, ptx / source.name)
    shutil.copy2(archive, destination / archive.name)
    inspect_device(archive, arch, cwd=cwd, env=env, runner=runner)
    if cubins:
        binary = destination / "cubin"
        binary.mkdir()
        for source in ptx_files:
            output = binary / source.with_suffix(".cubin").name
            runner(
                [
                    env["NATIVE_PTXAS"],
                    f"--gpu-name=sm_{arch}",
                    str(source),
                    "--output-file",
                    str(output),
                ],
                cwd=cwd,
                env=env,
            )
            with output.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise RuntimeError(f"ptxas did not produce an ELF cubin: {output}")
            inspect_device(output, arch, cwd=cwd, env=env, runner=runner)
    return sorted(versions)


def build_target(args, arch, base, metadata, env, *, runner=run):
    compatibility = {**base, "gpu_architecture": f"sm_{arch}"}
    key = digest(compatibility)
    # Export policy and provenance changes need new receipts, but they do not
    # require recompiling identical sources with the same compiler inputs.
    compiler_inputs = {
        k: v
        for k, v in compatibility.items()
        if k
        not in {
            "source_commit",
            "build_job_sha256",
            "kernels_only",
            "standalone_cubins",
        }
    }
    compiler_key = digest(
        {
            "cargo_policy": "locked-release-cuda-explicit-target-v1",
            "inputs": compiler_inputs,
        }
    )
    env = {
        **env,
        "CUDA_COMPUTE_CAP": arch,
        "CARGO_TARGET_DIR": str(args.build_root / compiler_key),
    }
    target = args.out / "targets" / f"sm_{arch}"
    target.mkdir(parents=True)
    # Enable the optional dependency through its workspace owner. Cargo cannot
    # select candle-kernels by -p before the owner's cuda feature is resolved.
    command = [
        "cargo",
        "build",
        "--locked",
        "--release",
        "--target",
        base["rust_target"],
        "--message-format=json-render-diagnostics",
        "-p",
        "switchyard-engine",
        "--features",
        "cuda",
        "--lib" if args.kernels_only else "--bins",
    ]
    if args.offline:
        command.append("--offline")
    print(
        json.dumps(
            {
                "event": "build_started",
                "architecture": f"sm_{arch}",
                "compatibility_key": key,
            }
        ),
        flush=True,
    )
    output = runner(command, cwd=args.workspace, env=env, log=target / "cargo.log")
    messages = []
    for line in output.splitlines():
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    kernel_id = next(
        p["id"] for p in metadata["packages"] if p["name"] == "candle-kernels"
    )
    kernel_outputs = {
        m["out_dir"]
        for m in messages
        if m.get("reason") == "build-script-executed"
        and m.get("package_id") == kernel_id
    }
    if len(kernel_outputs) != 1:
        raise RuntimeError("expected exactly one Candle kernel build output")
    versions = collect_kernels(
        Path(kernel_outputs.pop()),
        target / "kernels",
        arch,
        cwd=args.workspace,
        env=env,
        cubins=not args.no_cubins,
        runner=runner,
    )
    runtime_requirements = {}
    excluded_binaries = []
    if not args.kernels_only:
        engine_id = next(
            p["id"] for p in metadata["packages"] if p["name"] == "switchyard-engine"
        )
        binaries = {
            m["target"]["name"]: Path(m["executable"])
            for m in messages
            if m.get("reason") == "compiler-artifact"
            and m.get("package_id") == engine_id
            and m.get("executable")
            and "bin" in m["target"]["kind"]
        }
        if not EXPORTED_BINARIES.issubset(binaries):
            raise RuntimeError("expected all three native engine binaries")
        excluded_binaries = sorted(set(binaries) - EXPORTED_BINARIES)
        (target / "bin").mkdir()
        for name in sorted(EXPORTED_BINARIES):
            source = binaries[name]
            shutil.copy2(source, target / "bin" / name)
            elf = runner(
                ["readelf", "--dynamic", "--version-info", str(source)],
                cwd=args.workspace,
                env=env,
                log=target / f"{name}.elf.txt",
            )
            runtime_requirements[name] = {
                "shared_libraries": sorted(
                    set(re.findall(r"Shared library: \[([^\]]+)\]", elf))
                ),
                "symbol_versions": sorted(
                    set(re.findall(r"Name: ((?:GLIBC|GLIBCXX|CXXABI)_[0-9.]+)", elf))
                ),
            }
    files = {
        str(path.relative_to(target)): file_info(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.parts[len(target.parts)] in {"bin", "kernels"}
    }
    receipt = {
        "format": "switchyard-native-gpu-target-v1",
        "compatibility": compatibility,
        "compatibility_key": key,
        "compiler_cache_key": compiler_key,
        "ptx_isa_versions": versions,
        "runtime_requirements": runtime_requirements,
        "excluded_binary_targets": excluded_binaries,
        "files": files,
        "gpu_execution_validated": False,
        "driver_jit_cache_included": False,
        "model_bake_included": False,
        "standalone_cubins_consumed_by_current_engine": False,
    }
    receipt["artifact_id"] = digest(receipt)
    write_json(target / "build.json", receipt)
    print(
        json.dumps(
            {
                "event": "build_complete",
                "architecture": f"sm_{arch}",
                "artifact_id": receipt["artifact_id"],
            }
        ),
        flush=True,
    )
    return {
        "architecture": f"sm_{arch}",
        "path": str(target.relative_to(args.out)),
        "artifact_id": receipt["artifact_id"],
        "compatibility_key": key,
    }


def build_matrix(args, base, metadata, env, *, runner=run):
    args.out.mkdir(parents=True)
    (args.out / "producer.py").write_bytes(BUILD_JOB_SOURCE)
    try:
        targets = [
            build_target(args, arch, base, metadata, env, runner=runner)
            for arch in args.arch
        ]
        matrix = {
            "format": "switchyard-native-gpu-build-v1",
            "targets": targets,
            "gpu_execution_validated": False,
        }
        matrix["artifact_id"] = digest(matrix)
        write_json(args.out / "build.json", matrix)
    except BaseException as error:
        write_json(
            args.out / "failure.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise


def build(args):
    args.workspace = args.workspace.resolve(strict=True)
    args.out = args.out.absolute()
    args.build_root = (
        args.build_root or args.workspace / "target/native-gpu-builds"
    ).absolute()
    if len(set(args.arch)) != len(args.arch):
        raise ValueError("duplicate architecture")
    if args.source_commit and not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.source_commit
    ):
        raise ValueError("--source-commit must be a complete Git object ID")
    if args.out.exists() or args.out.is_symlink():
        raise ValueError("a fresh output directory is required")
    env = dict(os.environ)
    cuda = args.cuda_root.resolve(strict=True)
    env.update(
        {
            "CUDA_ROOT": str(cuda),
            "CUDA_HOME": str(cuda),
            "CUDA_PATH": str(cuda),
            "CUDAFORGE_THREADS": str(args.jobs),
            "CARGO_BUILD_JOBS": str(args.jobs),
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "CUDA_CACHE_DISABLE": "1",
            "NATIVE_PTXAS": str(cuda / "bin/ptxas"),
            "NATIVE_CUOBJDUMP": str(cuda / "bin/cuobjdump"),
        }
    )
    env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
    env["LIBRARY_PATH"] = os.pathsep.join(
        [str(cuda / "lib64/stubs"), str(cuda / "lib64"), env.get("LIBRARY_PATH", "")]
    )
    for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        if env.get(name):
            raise ValueError(
                f"unset {name}; architecture flags must come from this job"
            )
    supported = run(
        [str(cuda / "bin/nvcc"), "--list-gpu-arch"], cwd=args.workspace, env=env
    ).split()
    # nvcc lists base architectures only, even when it accepts suffixed targets.
    # Actual PTX and device-object inspection below checks the selected suffix.
    if any(f"compute_{arch.rstrip('af')}" not in supported for arch in args.arch):
        raise ValueError("selected architecture is unsupported by this CUDA toolkit")
    metadata_command = [
        "cargo",
        "metadata",
        "--locked",
        "--format-version=1",
        "--features",
        "switchyard-engine/cuda",
    ]
    if args.offline:
        metadata_command.append("--offline")
    metadata = json.loads(run(metadata_command, cwd=args.workspace, env=env))
    candle = {
        p["name"]: {"version": p["version"], "source": p["source"]}
        for p in metadata["packages"]
        if p["name"]
        in {"candle-core", "candle-nn", "candle-kernels", "cudaforge", "cudarc"}
    }
    rust = run(["rustc", "-Vv"], cwd=args.workspace, env=env).strip()
    host = next(
        line.removeprefix("host: ")
        for line in rust.splitlines()
        if line.startswith("host: ")
    )
    if not host.endswith("linux-gnu"):
        raise ValueError(
            "this job supports Linux GNU targets; use a matching Linux builder"
        )
    base = {
        "backend": "candle-cuda",
        "source_commit": args.source_commit
        or run(["git", "rev-parse", "HEAD"], cwd=args.workspace, env=env).strip(),
        "source_inputs_sha256": source_inputs(args.workspace, metadata),
        "cargo_lock_sha256": file_info(args.workspace / "Cargo.lock")["sha256"],
        "cargo_configuration": cargo_configuration(args.workspace),
        "dependencies": candle,
        "rust_target": host,
        "rustc": rust,
        "builder_identity": args.builder_id,
        "host_libc": platform.libc_ver(),
        "toolchain": {
            name: tool(cuda / f"bin/{name}", args.workspace, env)
            for name in ("nvcc", "ptxas", "cuobjdump")
        },
        "host_cxx": tool(
            env.get("NVCC_CCBIN") or shutil.which("g++"), args.workspace, env
        ),
        "build_environment": {
            k: env.get(k, "")
            for k in (
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER",
                "NVCC_CCBIN",
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "LIBRARY_PATH",
            )
        },
        "kernels_only": args.kernels_only,
        "standalone_cubins": not args.no_cubins,
        "build_job_sha256": hashlib.sha256(BUILD_JOB_SOURCE).hexdigest(),
    }
    if args.plan:
        print(
            json.dumps(
                {
                    "format": "switchyard-native-gpu-plan-v1",
                    "compatibility": base,
                    "architectures": args.arch,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    build_matrix(args, base, metadata, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        type=architecture,
        action="append",
        required=True,
        help="repeat for each explicit target: 89, 90a, 120a",
    )
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--build-root", type=Path, help="separate reusable Cargo build cache"
    )
    parser.add_argument(
        "--builder-id", help="immutable builder image identity for release builds"
    )
    parser.add_argument(
        "--source-commit",
        help="original Git revision for a caller-owned source snapshot; content is still hashed",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--kernels-only",
        action="store_true",
        help="compile Candle kernels without engine binaries",
    )
    parser.add_argument(
        "--no-cubins",
        action="store_true",
        help="retain PTX and the static archive without assembling standalone cubins",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="resolve inputs and toolchain without compiling or writing output",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        build(args)
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as error:
        print(f"native GPU build: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
