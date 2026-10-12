"""Agent-owned entrypoint; waits for SWITCHYARD_READY_FILE before serving."""

if __name__ == "__main__":
    import os
    from pathlib import Path
    import sys

    snapshot = os.environ.get("VLLM_CPU_SNAPSHOT_ROOT")
    if snapshot:
        root = Path(os.environ["SWITCHYARD_RUNTIME_ROOT"])
        program = root / os.environ.get("VLLM_CPU_SNAPSHOT_PROGRAM", "bin/switchyard-cpu-snapshot")
        work = Path(os.environ["SWITCHYARD_CACHE_ROOT"]) / "cpu-restore"
        os.execv(str(program), [str(program), "restore", "--snapshot", snapshot,
                               "--runtime-root", str(root), "--runtime-id",
                               os.environ["SWITCHYARD_RUNTIME_ARTIFACT_ID"],
                               "--work", str(work), "--", *sys.argv[1:]])

    from switchyard_vllm_startup import main

    main()
