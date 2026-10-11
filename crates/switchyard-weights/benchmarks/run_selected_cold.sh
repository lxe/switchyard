#!/bin/bash
# Reproduce the selected local fixture using its prepared immutable inputs.
set -euo pipefail
selected_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
selected_partition="${COLD_START_PARTITION:?Set COLD_START_PARTITION to the prepared runtime-set directory}"
selected_output="${1:-$selected_project/results/engine-startup/cold-rollout-$(date -u +%Y%m%dT%H%M%S-%N)}"
selected_runtime="${COLD_START_RUNTIME_CONFIG:-$selected_project/scratch/engine-startup/selected-runtime.json}"
mkdir -p "$(dirname "$selected_output")"
printf '[rollout] Fresh local container; prepared startup requests guarded fork, with native safety fallback to spawn.\n'
printf '[rollout] Local HTTP hydration and native profiling are included; pre-bake and agent build are excluded.\n'
python -u "$selected_project/crates/switchyard-weights/benchmarks/run_agent.py" "$selected_output" \
  --objects "$selected_project/results/engine-startup/qwen-extents" \
  --legacy-ticket "$selected_project/scratch/engine-startup/legacy-ticket.json" \
  --runtime-objects "$selected_partition/boot/artifact" \
  --runtime-config "$selected_runtime" \
  --component-objects "build=$selected_partition/components/build/artifact" \
  --cache-objects "compiled=$selected_project/results/engine-startup/selected-native-cache/artifact-selected-identity" \
  --agent "$selected_project/target/artifacts/switchyard-agent-x86_64-linux-musl" \
  --mode extents --profile-startup --live-timeline --skip-tokenize-check \
  2>&1 | tee "$selected_output.console.log"
printf '[rollout] Completed successfully; receipt at %s/extents/receipt.json\n' "$selected_output"
