#!/bin/bash
# The agent validates family artifact bindings before hydrating its private seed.
set -euo pipefail
native_family_root="${NATIVE_ENGINE_CACHE_ROOT:?native driver cache family root is required}"
native_expected_gpu="${NATIVE_ENGINE_EXPECTED_GPU:?native cache GPU name is required}"
native_expected_cap="${NATIVE_ENGINE_EXPECTED_COMPUTE_CAP:?native cache capability is required}"
native_cache_policy="${NATIVE_ENGINE_CACHE_POLICY:-strict}"
case "$native_cache_policy" in
  strict|opportunistic) ;;
  *) printf 'Native cache policy is invalid.\n' >&2; exit 78;;
esac
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
# Follow no symlinks while reading seeds or creating a cache. The agent gives
# each runtime a private absolute root, so no alias or parent traversal is needed.
[[ "$native_family_root" == /* && "$native_family_root/" != */../* && "$native_family_root/" != */./* ]] || {
  printf 'Native private CUDA cache root must be an absolute normalized path.\n' >&2; exit 78;
}
for native_selected_path in "$native_variant/cuda" "$native_variant/identity.tsv"; do
  native_checked_path="$native_selected_path"
  while [[ "$native_checked_path" != / ]]; do
    [[ ! -L "$native_checked_path" ]] || {
      printf 'Native private CUDA cache path cannot be a symlink.\n' >&2; exit 78;
    }
    native_checked_path="$(dirname -- "$native_checked_path")"
  done
done
if [[ -e "$native_variant" || -L "$native_variant" ]]; then
  [[ -d "$native_variant/cuda" && -f "$native_variant/identity.tsv" ]] || {
    printf 'Native cache family component is incomplete.\n' >&2; exit 78;
  }
  native_expected_identity="$(printf '%s\t%s\t%s' "$native_gpu" "$native_driver" "$native_cap")"
  [[ "$(cat "$native_variant/identity.tsv")" == "$native_expected_identity" ]] || {
    printf 'Native cache family component identity mismatch.\n' >&2; exit 78;
  }
  [[ -z "$(find "$native_variant/cuda" -type l -print -quit)" ]] || {
    printf 'Native cache family component contains a symlink.\n' >&2; exit 78;
  }
  export CUDA_CACHE_PATH="$native_variant/cuda"
  native_cache_status='hit'
else
  [[ "$native_cache_policy" == opportunistic ]] || {
    printf 'Native cache family has no validated driver variant.\n' >&2; exit 78;
  }
  # The caller's private cache root retains runtime/model/GPU bindings. A miss
  # starts empty for this observed driver and never borrows another seed.
  mkdir -p -m 700 "$native_family_root/drivers"
  chmod 700 "$native_family_root" "$native_family_root/drivers"
  # Stage beside the export root, then expose a complete exact-driver component
  # in one rename. A terminated launch cannot leave a half-written seed behind.
  native_stage="$(mktemp -d "$native_family_root/../.native-cuda-stage.$native_driver.XXXXXXXX")"
  trap 'if [[ -n "${native_stage:-}" ]]; then rm -rf -- "$native_stage"; fi' EXIT
  mkdir -m 700 "$native_stage/cuda"
  printf '%s\t%s\t%s\n' "$native_gpu" "$native_driver" "$native_cap" > "$native_stage/identity.tsv"
  chmod 600 "$native_stage/identity.tsv"
  mv -T -n -- "$native_stage" "$native_variant"
  [[ ! -d "$native_stage" ]] || {
    printf 'Native private CUDA cache component appeared during initialization.\n' >&2; exit 78;
  }
  native_stage=''
  trap - EXIT
  export CUDA_CACHE_PATH="$native_variant/cuda"
  native_cache_status='miss'
fi
printf 'native_cache_family_policy=%s\n' "$native_cache_policy"
printf 'native_cache_family_status=%s\n' "$native_cache_status"
printf 'native_cache_family_observed_gpu=%s\n' "$native_gpu"
printf 'native_cache_family_selected_driver=%s\n' "$native_driver"
printf 'native_cache_family_compute_cap=%s\n' "$native_cap"
printf 'native_cache_family_cuda_path=%s\n' "$CUDA_CACHE_PATH"
exec "$(dirname "${BASH_SOURCE[0]}")/switchyard-engine" "$@"
