#!/usr/bin/env bash
# CPU-only CRIU proof. Run in the existing observation pane; no tmux management.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${CPU_SNAPSHOT_PARTITION:?Set CPU_SNAPSHOT_PARTITION to a prepared runtime-set directory}"
: "${CPU_SNAPSHOT_OUT:?Set CPU_SNAPSHOT_OUT to a fresh output directory}"
program=${CPU_SNAPSHOT_PROGRAM:-$repo/target/release/switchyard-cpu-snapshot}
root=$CPU_SNAPSHOT_PARTITION/boot/tree
manifest=$CPU_SNAPSHOT_PARTITION/boot/artifact/manifest.json
runtime_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$manifest")
args=()
if [[ -n ${CPU_SNAPSHOT_TOKENIZER:-} ]]; then
  args+=(--tokenizer "$CPU_SNAPSHOT_TOKENIZER")
fi
# Elevate the whole owner so dump, restore, stdio transfer, and waitpid share an owner.
sudo -n "$program" prepare --runtime-root "$root" --runtime-manifest "$manifest" \
  --out "$CPU_SNAPSHOT_OUT" "${args[@]}"
printf 'switchyard-metadata-ready-v1\n' | sudo -n "$program" restore \
  --snapshot "$CPU_SNAPSHOT_OUT/tree" --runtime-root "$root" --runtime-id "$runtime_id" \
  --work "$CPU_SNAPSHOT_OUT/restore-work" --probe
