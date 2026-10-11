"""Build agent tickets from explicitly selected local immutable artifacts."""

import copy
import json
import re
from pathlib import Path


def without_cpu_snapshot(configuration):
    """Select ordinary CPU startup without changing GPU or hydration policy."""
    runtime = copy.deepcopy(configuration)
    runtime["environment"] = {
        key: value
        for key, value in runtime["environment"].items()
        if not key.startswith("VLLM_CPU_SNAPSHOT_")
    }
    runtime["components"] = [
        item for item in runtime.get("components", []) if item["name"] != "cpu-process"
    ]
    return runtime


def named_directories(assignments):
    directories = {}
    for value in assignments:
        name, directory = value.split("=", 1)
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None
            or name in directories
        ):
            raise ValueError("invalid or duplicate local artifact name")
        directories[name] = Path(directory).resolve()
    return directories


def artifacts(configurations, directories, kind):
    if {item["name"] for item in configurations} != set(directories):
        raise ValueError(
            f"selected {kind} artifacts differ from local object directories"
        )
    selected = []
    for configuration in configurations:
        name = configuration["name"]
        manifest = json.loads((directories[name] / "manifest.json").read_text())
        if manifest["id"] != configuration["artifact"]["id"]:
            raise ValueError(f"selected {kind} artifact identity mismatch: {name}")
        selected.append(
            {
                "name": name,
                "manifest": manifest,
                "compatibility": copy.deepcopy(configuration["compatibility"]),
            }
        )
    return selected


def ticket_artifact(manifest, base, route):
    return {
        "manifest": manifest,
        "urls": {
            chunk["name"]: f"{base}/{route}/{chunk['name']}"
            for chunk in manifest["chunks"]
        },
    }


def runtime_ticket(configuration, manifest, components, caches, base):
    ticket = {
        key: copy.deepcopy(configuration[key])
        for key in ["program", "args", "environment", "compatibility"]
    }
    ticket.update(
        artifact=ticket_artifact(manifest, base, "runtime"),
        caches=[],
        hydration={"retries": 0},
    )
    for kind, selected in [("components", components), ("caches", caches)]:
        if selected:
            ticket[kind] = [
                {
                    "name": item["name"],
                    "artifact": ticket_artifact(
                        item["manifest"], base, f"{kind}-{item['name']}"
                    ),
                    "compatibility": copy.deepcopy(item["compatibility"]),
                }
                for item in selected
            ]
    return ticket
