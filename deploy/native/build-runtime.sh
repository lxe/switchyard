#!/bin/bash
# Build CUDA code without allocating a GPU; the exported runtime excludes tools.
set -euo pipefail
native_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
native_output="${1:?usage: build-runtime.sh OUTPUT [CUDA_COMPUTE_CAP]}"
native_cap="${2:-89}"
native_builder="${NATIVE_CUDA_BUILDER:-nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d}"
[[ "$native_cap" =~ ^[0-9]{2,3}$ ]] || {
  printf 'This build helper requires a numeric CUDA target, such as 89.\n' >&2; exit 64;
}
# Receipt inventories must never contain files from an earlier export.
mkdir -p "$(dirname "$native_output")" "$native_project/target/native-ubuntu"
mkdir "$native_output"
native_output="$(realpath "$native_output")"
docker run --rm --name switchyard-native-ubuntu-build \
  -v "$native_project:/source:ro" \
  -v "$native_project/target/native-ubuntu:/build" \
  -v "$native_output:/output" \
  -e "CUDA_COMPUTE_CAP=$native_cap" \
  -e CARGO_TARGET_DIR=/build/target -e CARGO_HOME=/build/cargo \
  -e RUSTUP_HOME=/build/rustup -e CARGO_BUILD_JOBS=6 \
  -w /source --entrypoint /bin/bash "$native_builder" \
  /source/deploy/native/build-in-container.sh
python "$native_project/deploy/native/runtime-receipt.py" \
  --tree "$native_output/tree" --build "$native_output/build.json" \
  --source-commit "$(git -C "$native_project" rev-parse HEAD)" \
  --builder "$native_builder" --compute-cap "$native_cap" \
  --evidence "$native_output"
