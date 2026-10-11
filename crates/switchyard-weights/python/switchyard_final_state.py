"""Native vLLM final-state export and exact destination loading.

Native sharded-state export performs HF mapping/fusion during the bake. Serving
loads that resulting parameter/buffer state without calling model.load_weights.
The first supported layout is dense, unquantized, contiguous CUDA state at TP=1.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

FORMAT = "switchyard-vllm-final-state-v1"
RECEIPT = "switchyard-final-state.json"


def describe_state(state):
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
        }
        for name, tensor in state.items()
    }


def digest_layout(layout):
    return hashlib.sha256(
        json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def native_state(model):
    from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader

    return ShardedStateLoader._filter_subtensors(model.state_dict())


def validate_destinations(receipt, state, *, runtime_id, model_class):
    if receipt.get("format") != FORMAT or receipt.get("runtime_id") != runtime_id:
        raise ValueError("final-state format or runtime identity mismatch")
    if receipt.get("native_postload_completed") is not True:
        raise ValueError("final-state receipt has no completed native post-load bake")
    if receipt.get("model_class") != model_class or receipt.get(
        "vllm"
    ) != importlib.metadata.version("vllm"):
        raise ValueError("final-state model class or vLLM build mismatch")
    layout = describe_state(state)
    if receipt.get("layout") != layout or receipt.get("layout_sha256") != digest_layout(
        layout
    ):
        raise ValueError(
            "final-state names, shapes, dtype or strides differ from destination"
        )
    if receipt.get("tensor_parallel_size") != 1 or receipt.get("rank") != 0:
        raise ValueError(
            "final-state loader requires the prepared rank-zero TP=1 layout"
        )
    return state


def load_final_state(model, model_config, iterator_factory):
    receipt = json.loads((Path(model_config.model) / RECEIPT).read_text())
    state = validate_destinations(
        receipt,
        native_state(model),
        runtime_id=os.environ["SWITCHYARD_RUNTIME_ARTIFACT_ID"],
        model_class=type(model).__module__ + "." + type(model).__qualname__,
    )
    remaining = dict(state)
    for name, tensor in iterator_factory(state):
        if name not in remaining:
            raise ValueError("unknown or duplicate final-state tensor")
        destination = remaining.pop(name)
        if tensor.shape != destination.shape or tensor.dtype != destination.dtype:
            raise ValueError("final-state tensor does not match destination")
        if tensor.data_ptr() != destination.data_ptr():
            destination.copy_(tensor)
    if remaining:
        raise ValueError("final-state checkpoint is incomplete")
    return receipt


def export_model(model, output, context):
    """Call in a fully initialized native worker during preparation, never serve."""
    import torch
    from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = native_state(model)
    if not state or any(not t.is_contiguous() for t in state.values()):
        raise ValueError("final-state export requires nonempty contiguous state")
    ShardedStateLoader.save_model(model, str(output), max_size=512 * 1024 * 1024)
    layout = describe_state(state)
    receipt = {
        "format": FORMAT,
        "runtime_id": context["runtime_id"],
        "source_content_id": context["source_content_id"],
        "model_class": type(model).__module__ + "." + type(model).__qualname__,
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "tensor_parallel_size": 1,
        "rank": 0,
        "native_postload_completed": True,
        "layout": layout,
        "layout_sha256": digest_layout(layout),
        "tensor_count": len(state),
        "tensor_bytes": sum(t.numel() * t.element_size() for t in state.values()),
    }
    (output / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
    return {
        key: receipt[key]
        for key in ["layout_sha256", "tensor_count", "tensor_bytes", "model_class"]
    }
