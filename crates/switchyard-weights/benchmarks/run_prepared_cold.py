"""Run a prepared fixed-profile fixture through the actual static agent.

Inputs remain immutable. Every run gets a fresh installation and owned container;
phase summaries appear on stdout and detailed telemetry stays beneath output.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_artifacts import without_cpu_snapshot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "prepared",
    type=Path,
    help="Directory containing prepared.json, runtime.json and engine-args.json",
)
parser.add_argument(
    "runtime_set",
    type=Path,
    help="Partition containing boot/artifact and components/build/artifact",
)
parser.add_argument(
    "legacy_ticket",
    type=Path,
    help="Existing local fixture's legacy ticket, used only for fixture metadata",
)
parser.add_argument("--output", type=Path)
parser.add_argument(
    "--case", choices=["runtime", "artifacts", "serving"], default="serving"
)
parser.add_argument("--agent", type=Path)
parser.add_argument(
    "--image",
    help="Override the base image; ordinary startup defaults to the small Ubuntu serving image",
)
parser.add_argument(
    "--without-criu",
    action="store_true",
    help="Use ordinary CPU imports; omit CPU image transport and CRIU privileges",
)
parser.add_argument(
    "--without-native-profile",
    action="store_true",
    help="Control sample without Python import/operation instrumentation",
)
parser.add_argument(
    "--without-hydration-profile",
    action="store_true",
    help="Control sample with detailed shard instrumentation disabled",
)
args = parser.parse_args()
project = Path(__file__).resolve().parents[3]
prepared = json.loads((args.prepared / "prepared.json").read_text())
configuration = json.loads((args.prepared / "runtime.json").read_text())
runtime = configuration["bootstrap"]["runtime"]
output = args.output or project / "results/engine-startup" / (
    "direct-"
    + args.case
    + "-"
    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%f")
)
output.parent.mkdir(parents=True, exist_ok=True)
runtime_config = args.prepared / "runtime.json"
if args.without_criu:
    runtime = without_cpu_snapshot(runtime)
    configuration["bootstrap"]["runtime"] = runtime
    runtime_config = Path(str(output) + ".runtime.json")
    with runtime_config.open("x") as selected:
        json.dump(configuration, selected, indent=2)
        selected.write("\n")
cpu_snapshot_enabled = bool(runtime["environment"].get("VLLM_CPU_SNAPSHOT_ROOT"))
image = args.image or (
    "switchyard-local-criu"
    if cpu_snapshot_enabled
    else "ghcr.io/lxe/switchyard-runtime@sha256:a1b9a820d3a864628fce77c44182f0e7ce15b91d641ae88f243b298a810b9136"
)
print(
    "[rollout] CPU startup: "
    + (
        "CRIU native process tree"
        if cpu_snapshot_enabled
        else "ordinary imports; CRIU disabled"
    ),
    flush=True,
)
command = [
    sys.executable,
    "-u",
    str(Path(__file__).with_name("run_agent.py")),
    str(output),
    "--objects",
    prepared["model_objects"],
    "--legacy-ticket",
    str(args.legacy_ticket),
    "--runtime-objects",
    str(args.runtime_set / "boot/artifact"),
    "--runtime-config",
    str(runtime_config),
    "--engine-args",
    str(args.prepared / "engine-args.json"),
    "--agent",
    str(args.agent or project / "target/artifacts/switchyard-agent-x86_64-linux-musl"),
    "--image",
    image,
    "--memory-limit",
    "32g",
    "--mode",
    "extents",
    "--live-timeline",
]
if cpu_snapshot_enabled:
    command += [
        "--cpu-restore",
        str(
            project
            / "target/musl/x86_64-unknown-linux-musl/release/switchyard-cpu-snapshot"
        ),
    ]
for component in runtime.get("components", []):
    name = component["name"]
    if name == "cpu-process":
        path = prepared["cpu_objects"]
    elif name == "build":
        path = args.runtime_set / "components/build/artifact"
    else:
        parser.error("fixture has no local binding for component " + name)
    command += ["--component-objects", f"{name}={path}"]
command += ["--cache-objects", "compiled=" + prepared["cache_objects"]]
if not args.without_hydration_profile:
    command.append("--hydration-profile")
if args.case == "serving":
    if not args.without_native_profile:
        command.append("--profile-startup")
else:
    command += ["--isolation", args.case]
with Path(str(output) + ".console.log").open("w") as log, subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    code = process.wait()
sys.exit(code)
