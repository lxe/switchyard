#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install --no-install-recommends -y curl ca-certificates build-essential cmake pkg-config python3
if [[ ! -x /build/cargo/bin/rustc ]]; then
  curl -fsSL https://sh.rustup.rs -o /build/rustup-init.sh
  bash /build/rustup-init.sh -y --profile minimal --default-toolchain 1.98.1 --no-modify-path
fi
export PATH="/build/cargo/bin:$PATH"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64"
export CUDA_ROOT=/usr/local/cuda
cargo build --locked --release -p switchyard-engine --features cuda --bins
mkdir -p /output/tree/bin /output/tree/lib
for native_bin in switchyard-engine switchyard-engine-bake; do
  cp "/build/target/release/$native_bin" "/output/tree/bin/$native_bin"
  ldd "/output/tree/bin/$native_bin" >"/output/$native_bin.ldd"
done
# The host driver supplies libcuda. Only the actual toolkit closure is bundled.
while read -r native_lib; do
  case "$native_lib" in
    /usr/local/cuda/*)
      native_name="$(basename "$native_lib")"
      [[ "$native_name" != libcuda.so* ]] && cp -L "$native_lib" "/output/tree/lib/$native_name"
      ;;
    */libstdc++.so.6|*/libgcc_s.so.1)
      cp -L "$native_lib" "/output/tree/lib/$(basename "$native_lib")"
      ;;
  esac
done < <(ldd /output/tree/bin/switchyard-engine | awk '$2 == "=>" && $3 ~ /^\// {print $3}')
readelf --version-info /output/tree/bin/switchyard-engine > /output/elf-versions.txt
nvcc --version > /output/nvcc.txt
rustc -Vv > /output/rustc.txt
chown -R "$(stat -c %u /output):$(stat -c %g /output)" /output
