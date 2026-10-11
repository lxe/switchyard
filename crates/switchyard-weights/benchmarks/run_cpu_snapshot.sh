#!/usr/bin/env bash
# Native CPU process-tree bake. Run in the existing pane; no tmux management.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${CPU_SNAPSHOT_PARTITION:?Set CPU_SNAPSHOT_PARTITION to a prepared runtime-set directory}"
: "${CPU_SNAPSHOT_OUT:?Set CPU_SNAPSHOT_OUT to a fresh output directory}"
: "${CPU_SNAPSHOT_PREPARED:?Set CPU_SNAPSHOT_PREPARED to the selected serving fixture}"
: "${CPU_SNAPSHOT_BAKE:?Set CPU_SNAPSHOT_BAKE to the native model/cache bake}"
: "${CPU_SNAPSHOT_LOADER_CONFIG:?Set CPU_SNAPSHOT_LOADER_CONFIG to the selected loader JSON}"
exec python3 -u "$repo/crates/switchyard-weights/benchmarks/prepare_cpu_tree.py" \
  "$CPU_SNAPSHOT_PREPARED" "$CPU_SNAPSHOT_PARTITION" "$CPU_SNAPSHOT_BAKE" "$CPU_SNAPSHOT_OUT" \
  --loader-config "$CPU_SNAPSHOT_LOADER_CONFIG"
