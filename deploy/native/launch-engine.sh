#!/bin/bash
# Cache selection is verified during preparation. Startup checks host identity only.
set -euo pipefail
if [[ -n "${NATIVE_ENGINE_EXPECTED_DRIVER:-}" ]]; then
  native_identity="$(nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader -i 0)"
  IFS=',' read -r native_driver native_cap <<< "$native_identity"
  native_driver="${native_driver//[[:space:]]/}"
  native_cap="${native_cap//[[:space:]]/}"
  [[ "$native_driver" == "$NATIVE_ENGINE_EXPECTED_DRIVER" ]] || {
    printf 'Native cache driver mismatch.\n' >&2; exit 78;
  }
  [[ "$native_cap" == "$NATIVE_ENGINE_EXPECTED_COMPUTE_CAP" ]] || {
    printf 'Native cache GPU architecture mismatch.\n' >&2; exit 78;
  }
fi
exec "$(dirname "${BASH_SOURCE[0]}")/switchyard-engine" "$@"
