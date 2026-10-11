"""Detached entrypoint for the native API/engine CPU process-tree checkpoint.

The Rust owner supplies one external control socket and an exact serving
configuration. The serving adapter checkpoints both processes before CUDA.
"""

import json
from pathlib import Path
import socket
import sys

from switchyard_cpu_tree import main

if __name__ == "__main__":
    control = socket.socket(fileno=int(sys.argv[1]))
    try:
        config = json.loads(Path(sys.argv[2]).read_text())
        main(control, config)
    except BaseException as error:
        if control.fileno() >= 0:
            from switchyard_cpu_tree import send

            send(control, {"error": str(error)})
        raise
