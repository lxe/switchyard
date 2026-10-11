"""Compare the candidate with the pinned native CLI at its dispatch boundary.

The native parsers, validators and ServeSubcommand.cmd execute. Only the final
server launch functions are replaced, so these CPU checks create no GPU engine.
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from vllm.entrypoints.cli import main as general
from vllm.entrypoints.cli import serve
from vllm.entrypoints.launchers.api_server import entry as api_entry

spec = importlib.util.spec_from_file_location(
    "switchyard_serve_only", Path(__file__).with_name("serve_only.py")
)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def execute(entrypoint, arguments):
    dispatched = []

    def capture(name, args):
        values = vars(args).copy()
        values.pop("dispatch_function", None)
        dispatched.append((name, values))

    async def single(args):
        capture("single_api", args)

    os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
    with (
        patch.object(sys, "argv", ["vllm", *arguments]),
        patch.object(serve, "run_server", single),
        patch.object(serve, "run_headless", lambda args: capture("headless", args)),
        patch.object(
            serve, "run_multi_api_server", lambda args: capture("multi_api", args)
        ),
        patch.object(
            serve, "run_dp_supervisor", lambda args: capture("supervisor", args)
        ),
    ):
        try:
            entrypoint()
        except (SystemExit, TypeError, ValueError) as error:
            return (type(error).__name__, str(error)), dispatched
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    return None, dispatched


with tempfile.TemporaryDirectory(prefix="switchyard-serve-parser-") as directory:
    config = Path(directory) / "serve.yaml"
    config.write_text(
        "model: config-model\nmax-model-len: 128\nlanguage-model-only: true\n"
        "dtype: bfloat16\nserved-model-name: fixture\n"
    )
    cases = {
        "extent_language_model": [
            "serve",
            "metadata-model",
            "--served-model-name",
            "fixture",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--dtype",
            "bfloat16",
            "--language-model-only",
            "--max-model-len",
            "64",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "64",
            "--gpu-memory-utilization",
            "0.25",
            "--enforce-eager",
            "--attention-backend",
            "TRITON_ATTN",
            "--seed",
            "123",
            "--load-format",
            "switchyard_extents",
            "--model-loader-extra-config",
            '{"socket_path":"/private-model.sock","cuda_stage":true,"pinned_bytes":1048576}',
            "--default-chat-template-kwargs",
            '{"enable_thinking":false}',
        ],
        "config_model_cli_override": [
            "serve",
            "--config",
            str(config),
            "--max-model-len",
            "64",
        ],
        "positional_model_over_config": [
            "serve",
            "metadata-model",
            "--config",
            str(config),
        ],
        "deprecated_model_flag": [
            "serve",
            "--model",
            "metadata-model",
            "--port",
            "8001",
        ],
        "underscored_flags": [
            "serve",
            "metadata-model",
            "--language_model_only",
            "--max_model_len=64",
        ],
        "headless": ["serve", "metadata-model", "--headless"],
        "multiple_api_servers": ["serve", "metadata-model", "--api-server-count", "2"],
        "data_parallel_default": [
            "serve",
            "metadata-model",
            "--data-parallel-size",
            "2",
        ],
        "invalid_tool_parser": ["serve", "metadata-model", "--enable-auto-tool-choice"],
        "invalid_keepalive": [
            "serve",
            "metadata-model",
            "--sse-keep-alive-interval",
            "-1",
        ],
        "invalid_headless_count": [
            "serve",
            "metadata-model",
            "--headless",
            "--api-server-count",
            "1",
        ],
    }
    results = []
    for name, arguments in cases.items():
        before = execute(general.main, arguments)
        after = execute(candidate.main, arguments)
        assert before == after, name
        if name in ("extent_language_model", "positional_model_over_config"):
            assert after[1][0][1]["model"] == "metadata-model"
        if name == "config_model_cli_override":
            assert after[1][0][1]["model"] == "config-model"
            assert after[1][0][1]["max_model_len"] == 64
        if name == "extent_language_model":
            assert after[1][0][1]["language_model_only"] is True
            assert after[1][0][1]["load_format"] == "switchyard_extents"
        results.append(
            {
                "case": name,
                "equal": True,
                "error_type": after[0][0] if after[0] else None,
                "dispatch": after[1][0][0] if after[1] else None,
            }
        )

# Commands outside the adapter scope retain the unmodified native CLI source.
for arguments in (["collect-env"], ["serve", "metadata-model", "--omni"]):
    with (
        patch.object(sys, "argv", ["vllm", *arguments]),
        patch("runpy.run_path") as run,
    ):
        candidate.main()
        assert run.call_count == 1
        assert run.call_args.kwargs == {"run_name": "__main__"}
        assert Path(run.call_args.args[0]).resolve() == Path(general.__file__).resolve()

# The official single-server entrypoint is not a substitute for serve dispatch:
# its parser accepts these flags, but it calls run_server without cmd translation.
direct = []


async def direct_single(args):
    direct.append(vars(args).copy())


with (
    patch.object(
        sys,
        "argv",
        ["api-server", "metadata-model", "--language-model-only", "--headless"],
    ),
    patch.object(api_entry, "run_server", direct_single),
):
    api_entry.main()
assert len(direct) == 1
assert direct[0]["model_tag"] == "metadata-model"
assert direct[0]["model"] != "metadata-model"
assert direct[0]["headless"] is True
assert direct[0]["language_model_only"] is True

print(
    json.dumps(
        {
            "passed": True,
            "native_cases": results,
            "general_fallbacks": 2,
            "official_single_entrypoint_observation": {
                "model_tag": direct[0]["model_tag"],
                "model": direct[0]["model"],
                "headless_flag": direct[0]["headless"],
                "dispatch": "single_api",
            },
        }
    )
)
