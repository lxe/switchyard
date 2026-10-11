#!/bin/bash
# The agent validates family artifact bindings before hydrating its private seed.
set -euo pipefail
native_family_root="${NATIVE_ENGINE_CACHE_ROOT:?native driver cache family root is required}"
native_expected_gpu="${NATIVE_ENGINE_EXPECTED_GPU:?native cache GPU name is required}"
native_expected_cap="${NATIVE_ENGINE_EXPECTED_COMPUTE_CAP:?native cache capability is required}"
native_identity="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader -i 0)"
IFS=',' read -r native_gpu native_driver native_cap <<< "$native_identity"
native_gpu="$(printf '%s' "$native_gpu" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
native_driver="${native_driver//[[:space:]]/}"
native_cap="${native_cap//[[:space:]]/}"
[[ "$native_gpu" == "$native_expected_gpu" && "$native_cap" == "$native_expected_cap" ]] || {
  printf 'Native cache family GPU identity mismatch.\n' >&2; exit 78;
}
[[ "$native_driver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  printf 'Native cache family driver identity is invalid.\n' >&2; exit 78;
}
native_variant="$native_family_root/drivers/$native_driver"
[[ -d "$native_variant/cuda" && -f "$native_variant/identity.tsv" ]] || {
  printf 'Native cache family has no validated driver variant.\n' >&2; exit 78;
}
native_expected_identity="$(printf '%s\t%s\t%s' "$native_gpu" "$native_driver" "$native_cap")"
[[ "$(cat "$native_variant/identity.tsv")" == "$native_expected_identity" ]] || {
  printf 'Native cache family component identity mismatch.\n' >&2; exit 78;
}
export CUDA_CACHE_PATH="$native_variant/cuda"
printf 'native_cache_family_selected_driver=%s\n' "$native_driver"
exec "$(dirname "${BASH_SOURCE[0]}")/switchyard-engine" "$@"
