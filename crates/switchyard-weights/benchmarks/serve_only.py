"""Experimental serve-only CLI adapter for the pinned vLLM runtime.

The native serve parser, validation and dispatch still run. This source is used
only by a benchmark overlay until total startup and parser equivalence pass.
"""

import importlib.metadata
import sys


def main():
    if sys.argv[1:2] != ["serve"] or "--omni" in sys.argv:
        # Preserve the general CLI for commands outside this adapter's scope.
        # During an overlay the normal module name resolves to this file, so
        # execute the original runtime source by its installed path instead.
        import runpy
        from pathlib import Path

        import vllm

        runpy.run_path(
            str(Path(vllm.__file__).parent / "entrypoints/cli/main.py"),
            run_name="__main__",
        )
        return

    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.entrypoints.serve.utils.api_utils import (
        VLLM_SUBCMD_PARSER_EPILOG,
        cli_env_setup,
    )
    from vllm.logger import configure_logging_from_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    command = ServeSubcommand()
    command.subparser_init(subparsers).set_defaults(dispatch_function=command.cmd)
    args = parser.parse_args()
    if args.subparser == command.name:
        configure_logging_from_args(args)
        command.validate(args)
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
