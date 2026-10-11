"""Focused admission and final-state checks; CUDA proof runs in root's fixture."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import switchyard_final_state as final
import switchyard_serving_profile as serving


class ProfileTests(unittest.TestCase):
    def test_fixed_budget_rejects_mismatched_identity_config_and_headroom(self):
        engine = {"max_model_len": 64}
        gpu = {"name": "RTX5090"}
        profile = {
            "schema_version": 1,
            "purpose": "fixed_serving_budget",
            "runtime_id": "runtime",
            "model_content_id": "model",
            "engine": engine,
            "gpu": gpu,
            "kv_cache_memory_bytes": 100,
            "runtime_headroom_bytes": 50,
            "bake": {"native_warmup_completed": True},
        }
        args = {
            "runtime_id": "runtime",
            "model_content_id": "model",
            "engine": engine,
            "gpu": gpu,
            "free_bytes": 150,
            "kv_bytes": 100,
        }
        self.assertEqual(serving.check_profile(profile, **args), 100)
        for update in [
            {"runtime_id": "other"},
            {"model_content_id": "other"},
            {"engine": {"max_model_len": 65}},
            {"gpu": {"name": "L40S"}},
            {"free_bytes": 149},
            {"kv_bytes": 101},
        ]:
            with (
                self.subTest(update=update),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                serving.check_profile(profile, **{**args, **update})


class Tensor:
    def __init__(self, value):
        self.value, self.shape, self.dtype = value, (2,), "torch.bfloat16"

    def stride(self):
        return (1,)

    def data_ptr(self):
        return id(self)

    def copy_(self, source):
        self.value = source.value


class FinalStateTests(unittest.TestCase):
    def test_direct_named_copy_never_calls_hf_loader_and_rejects_missing_keys(self):
        class Model:
            def load_weights(self, unused):
                raise AssertionError("HF mapping must not run")

        model = Model()
        state = {"packed.weight": Tensor([0, 0]), "buffer": Tensor([0, 0])}
        layout = final.describe_state(state)
        receipt = {
            "format": final.FORMAT,
            "runtime_id": "runtime",
            "native_postload_completed": True,
            "model_class": type(model).__module__ + "." + type(model).__qualname__,
            "vllm": "0.31.0",
            "layout": layout,
            "layout_sha256": final.digest_layout(layout),
            "tensor_parallel_size": 1,
            "rank": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, final.RECEIPT).write_text(json.dumps(receipt))
            with (
                patch.dict(os.environ, {"SWITCHYARD_RUNTIME_ARTIFACT_ID": "runtime"}),
                patch.object(final, "native_state", return_value=state),
                patch.object(
                    final.importlib.metadata, "version", return_value="0.31.0"
                ),
            ):
                config = SimpleNamespace(model=directory)
                final.load_final_state(
                    model,
                    config,
                    lambda _: iter(
                        [("packed.weight", Tensor([1, 2])), ("buffer", Tensor([3, 4]))]
                    ),
                )
                self.assertEqual(state["packed.weight"].value, [1, 2])
                self.assertEqual(state["buffer"].value, [3, 4])
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    final.load_final_state(
                        model,
                        config,
                        lambda _: iter([("packed.weight", Tensor([1, 2]))]),
                    )

    def test_layout_dtype_shape_and_stride_must_match_before_copy(self):
        state = {"packed.weight": Tensor([1, 2])}
        layout = final.describe_state(state)
        receipt = {
            "format": final.FORMAT,
            "runtime_id": "runtime",
            "native_postload_completed": True,
            "model_class": "Model",
            "vllm": "0.31.0",
            "layout": layout,
            "layout_sha256": final.digest_layout(layout),
            "tensor_parallel_size": 1,
            "rank": 0,
        }
        state["packed.weight"].dtype = "torch.float32"
        with (
            patch.object(final.importlib.metadata, "version", return_value="0.31.0"),
            self.assertRaisesRegex(ValueError, "dtype"),
        ):
            final.validate_destinations(
                receipt, state, runtime_id="runtime", model_class="Model"
            )
