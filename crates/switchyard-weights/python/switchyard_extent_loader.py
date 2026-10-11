"""Experimental local vLLM adapter for the Rust completion bridge.

The Rust process owns range validation, readiness, file descriptors and publication.
This module supplies completed CPU views to vLLM's ordinary model.load_weights path.
It does not open incomplete files through safetensors.safe_open.
"""

# A spawned worker opens its own lease; inherited API descriptors alone are
# insufficient because multiprocessing may close descriptors during spawn.
import fcntl
import json
import os
import socket
import time

_loader_lease = None
if lease_path := os.environ.get("SWITCHYARD_LOADER_LEASE"):
    _loader_lease = open(lease_path, "rb")  # noqa: SIM115 - lease lasts for the process
    fcntl.flock(_loader_lease, fcntl.LOCK_SH | fcntl.LOCK_NB)

import torch

_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


class CompletedWeights:
    """One checkpoint consumer. Mappings remain live through all assignments.

    Default CPU tensors retain vLLM's synchronous assignment semantics. Optional
    CUDA staging owns one stream, one bounded pinned buffer and one temporary
    device tensor. Events finish before yielding, so model-specific assignment,
    transposition and tensor-parallel slicing occur on a ready tensor.
    """

    def __init__(
        self,
        socket_path,
        *,
        timeout=120,
        cuda_device=None,
        pinned_bytes=16 * 1024 * 1024,
    ):
        if pinned_bytes <= 0:
            raise ValueError("pinned_bytes must be positive")
        self.mappings = {}
        self.device = None if cuda_device is None else torch.device(cuda_device)
        self.stream = (
            None if self.device is None else torch.cuda.Stream(device=self.device)
        )
        self.pinned = (
            None
            if self.device is None
            else torch.empty(pinned_bytes, dtype=torch.uint8, pin_memory=True)
        )
        # Prepare reusable host/CUDA resources before announcing consumer readiness.
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(timeout)
        try:
            self.socket.connect(socket_path)
        except Exception:
            self.socket.close()
            raise
        self.input = self.socket.makefile("rb")
        self.done = False
        self.receipts = []

    def __iter__(self):
        while True:
            self.socket.sendall(b"next\n")
            line = self.input.readline(1024 * 1024)
            if not line.endswith(b"\n"):
                raise RuntimeError(
                    "Rust weight bridge closed or exceeded message bound"
                )
            message = json.loads(line)
            if message.get("done"):
                self.done = True
                return
            received_monotonic = time.perf_counter()
            spec = message["tensor"]
            key = (spec["entry"], message["path"])
            if key not in self.mappings:
                self.mappings[key] = torch.from_file(
                    message["path"],
                    shared=False,
                    size=spec["file_size"],
                    dtype=torch.uint8,
                )
            raw = self.mappings[key].narrow(0, spec["offset"], spec["length"])
            tensor = raw.view(_DTYPES[spec["dtype"]]).reshape(spec["shape"])
            receipt = {
                "name": spec["name"],
                "ready_seconds": message["ready_seconds"],
                "bytes": spec["length"],
                "descriptor_received_monotonic": received_monotonic,
            }
            if self.device is not None:
                start = time.perf_counter()
                # Full temporary tensor is necessary for model-specific assignment;
                # pinned host staging is bounded independently of that tensor size.
                device_raw = torch.empty(
                    spec["length"], dtype=torch.uint8, device=self.device
                )
                self.stream.wait_stream(torch.cuda.current_stream(self.device))
                device_raw.record_stream(self.stream)
                event = torch.cuda.Event(enable_timing=True)
                copy_started = torch.cuda.Event(enable_timing=True)
                host_seconds = 0.0
                dma_seconds = 0.0
                for offset in range(0, spec["length"], self.pinned.numel()):
                    count = min(self.pinned.numel(), spec["length"] - offset)
                    host_began = time.perf_counter()
                    self.pinned[:count].copy_(raw[offset : offset + count])
                    host_seconds += time.perf_counter() - host_began
                    with torch.cuda.stream(self.stream):
                        copy_started.record(self.stream)
                        device_raw[offset : offset + count].copy_(
                            self.pinned[:count], non_blocking=True
                        )
                        event.record(self.stream)
                    # The same pinned buffer cannot be reused until DMA completes.
                    event.synchronize()
                    dma_seconds += copy_started.elapsed_time(event) / 1000
                tensor = device_raw.view(_DTYPES[spec["dtype"]]).reshape(spec["shape"])
                receipt["host_stage_seconds"] = host_seconds
                receipt["gpu_dma_seconds"] = dma_seconds
                receipt["h2d_seconds"] = time.perf_counter() - start
                receipt["copy_started_monotonic"] = start
                receipt["copy_finished_monotonic"] = time.perf_counter()
            self.receipts.append(receipt)
            yield spec["name"], tensor

    def finish(self):
        if not self.done:
            raise RuntimeError("Cannot acknowledge an incomplete checkpoint")
        if self.stream is not None:
            self.stream.synchronize()
        if self.device is None:
            self.socket.sendall(b"finish\n")
        else:
            mapped = [
                r["ready_seconds"]
                + r["copy_finished_monotonic"]
                - r["descriptor_received_monotonic"]
                for r in self.receipts
            ]
            report = {
                "copies": len(self.receipts),
                "bytes": sum(r["bytes"] for r in self.receipts),
                "first_copy_finished_seconds": min(mapped, default=None),
                "last_copy_finished_seconds": max(mapped, default=None),
                "staging_seconds": sum(r["host_stage_seconds"] for r in self.receipts),
                "dma_seconds": sum(r["gpu_dma_seconds"] for r in self.receipts),
            }
            self.socket.sendall(b"report\n" + json.dumps(report).encode() + b"\n")

    def close(self):
        if self.stream is not None:
            self.stream.synchronize()
        self.input.close()
        self.socket.close()
        self.mappings.clear()


def register():
    """Invoke through a vLLM general plugin in every engine worker process."""
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader("switchyard_extents")
    class ExtentModelLoader(DefaultModelLoader):
        def __init__(self, load_config):
            extra = dict(load_config.model_loader_extra_config)
            self.socket_path = extra.pop("socket_path")
            self.timeout = extra.pop("timeout_seconds", 120)
            self.cuda_stage = extra.pop("cuda_stage", False)
            self.pinned_bytes = extra.pop("pinned_bytes", 16 * 1024 * 1024)
            # Let the pinned base class validate only its ordinary configuration.
            import copy

            base_config = copy.copy(load_config)
            base_config.model_loader_extra_config = extra
            super().__init__(base_config)
            self.completed = None

        def download_model(self, model_config):
            # All config/tokenizer metadata must already be complete and local.
            return None

        def _get_weights_iterator(self, source):
            from vllm.config import get_current_vllm_config

            parallel = get_current_vllm_config().parallel_config
            if any(
                getattr(parallel, key) != 1
                for key in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                )
            ):
                raise ValueError(
                    "Extent bridge currently requires a single engine worker"
                )
            if (
                self.local_expert_ids is not None
                or self._encoder_only_lm_prefixes is not None
            ):
                raise ValueError(
                    "Extent bridge does not implement expert or encoder-only filtering"
                )
            if source.prefix or source.subfolder or self.completed is not None:
                raise ValueError(
                    "Extent adapter currently supports one unprefixed checkpoint"
                )
            device = f"cuda:{torch.cuda.current_device()}" if self.cuda_stage else None
            self.completed = CompletedWeights(
                self.socket_path,
                timeout=self.timeout,
                cuda_device=device,
                pinned_bytes=self.pinned_bytes,
            )
            if self.counter_before_loading_weights == 0.0:
                self.counter_before_loading_weights = time.perf_counter()
            return iter(self.completed)

        def load_weights(self, model, model_config):
            if model_config.quantization is not None or model_config.is_moe:
                raise ValueError(
                    "Extent bridge currently supports dense, unquantized checkpoints"
                )
            try:
                super().load_weights(model, model_config)
                # Preserve base model weight tracking. Consumers acknowledge only
                # after model-specific assignment has returned and CUDA has settled.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.completed.finish()
            finally:
                if self.completed is not None:
                    self.completed.close()
