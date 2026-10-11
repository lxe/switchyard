"""Benchmark-only vLLM plugins; engine allocation/loading remains vLLM's own."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch


def fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(
            parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def register():
    import switchyard_extent_loader
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    switchyard_extent_loader.register()
    receipt_path = Path(os.environ["SWITCHYARD_BENCH_RECEIPT"])
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader

    extent_type = type(
        get_model_loader(
            LoadConfig(
                load_format="switchyard_extents",
                model_loader_extra_config={
                    "socket_path": os.environ["SWITCHYARD_BENCH_SOCKET"]
                },
            )
        )
    )

    @register_model_loader("benchmark_extents")
    class MeasuredExtents(extent_type):
        def load_weights(self, model, model_config):
            start = time.perf_counter()
            super().load_weights(model, model_config)
            receipt_path.write_text(
                json.dumps(
                    {
                        "weight_seconds": time.perf_counter() - start,
                        "parameters_sha256": fingerprint(model),
                        "tensors": self.completed.receipts,
                    }
                )
            )

    @register_model_loader("benchmark_legacy")
    class MeasuredLegacy(DefaultModelLoader):
        def __init__(self, load_config):
            import copy

            config = copy.copy(load_config)
            config.load_format = "safetensors"
            super().__init__(config)

        def download_model(self, model_config):
            pass

        def _prepare_weights(self, *args, **kwargs):
            start = time.perf_counter()
            output = subprocess.check_output(
                [
                    os.environ["SWITCHYARD_BENCH_LEGACY"],
                    os.environ["SWITCHYARD_BENCH_TICKET"],
                    os.environ["SWITCHYARD_BENCH_ROOT"],
                ],
                text=True,
            )
            self.network_receipt = json.loads(output)
            self.network_receipt["observed_seconds"] = time.perf_counter() - start
            args = (str(Path(os.environ["SWITCHYARD_BENCH_ROOT"]) / "model"),) + args[
                1:
            ]
            return super()._prepare_weights(*args, **kwargs)

        def load_weights(self, model, model_config):
            start = time.perf_counter()
            super().load_weights(model, model_config)
            torch.cuda.synchronize()
            receipt_path.write_text(
                json.dumps(
                    {
                        "weight_seconds": time.perf_counter() - start,
                        "parameters_sha256": fingerprint(model),
                        "network": self.network_receipt,
                    }
                )
            )
