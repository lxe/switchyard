"""Native import-order checks for the pinned lazy fallback backport.

Run with the private patched runtime's Python, once for each --order.
"""

import argparse
import importlib
import json
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--order", choices=["vllm-first", "lowering-first", "graph-first"], required=True
)
args = parser.parse_args()
started = time.monotonic()
if args.order == "lowering-first":
    importlib.import_module("torch._inductor.lowering")
    assert "torch._inductor.lowering" in sys.modules
elif args.order == "graph-first":
    importlib.import_module("torch._inductor.graph")
    assert "torch._inductor.graph" in sys.modules

import vllm
from vllm import env_override

if args.order == "vllm-first":
    assert "torch._inductor.lowering" not in sys.modules
    assert "torch._inductor.graph" not in sys.modules

from torch._inductor import graph, lowering

proxy = lowering.FALLBACK_ALLOW_LIST
assert isinstance(proxy, env_override._VllmFallbackAllowList)
assert graph.FALLBACK_ALLOW_LIST is proxy
assert "vllm::regression_custom_op" in proxy
assert "vllm_aiter::regression_custom_op" in proxy
assert "other::regression_custom_op" not in proxy
baseline = set(proxy)
assert all(entry in proxy for entry in baseline)
assert len(proxy) == len(baseline)
proxy.add("other::temporary_op")
assert "other::temporary_op" in proxy
assert "other::temporary_op" in proxy._inner
proxy.discard("other::temporary_op")
assert set(proxy) == baseline
# Re-applying preserves the proxy and underlying set, rather than wrapping again.
env_override._patch_inductor_fallback_allow_list()
assert lowering.FALLBACK_ALLOW_LIST is proxy
assert graph.FALLBACK_ALLOW_LIST is proxy
from torch._inductor import pattern_matcher

assert (
    pattern_matcher.fallback_node_due_to_unsupported_type.__name__
    == "fallback_for_builtin"
)
print(
    json.dumps(
        {
            "order": args.order,
            "passed": True,
            "seconds": time.monotonic() - started,
            "native_lowering_entries": len(baseline),
            "vllm_version": vllm.__version__,
        }
    )
)
