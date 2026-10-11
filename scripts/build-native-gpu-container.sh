#!/bin/bash
# Runs only inside the owned, digest-pinned CPU builder container.
set -euo pipefail
native_jobs="${1:?CPU jobs required}"
native_commit="${2:?source commit required}"
native_image="${3:?builder image required}"
shift 3
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]]
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install --no-install-recommends -y \
  curl ca-certificates git build-essential binutils cmake pkg-config python3
export CARGO_HOME=/build/cargo RUSTUP_HOME=/build/rustup
if [[ ! -x /build/cargo/bin/rustup ]]; then
  curl -fsSL https://sh.rustup.rs -o /build/rustup-init.sh
  bash /build/rustup-init.sh -y --profile minimal \
    --default-toolchain 1.98.1 --no-modify-path
fi
export PATH="/build/cargo/bin:$PATH"
rustup toolchain install 1.98.1 --profile minimal
export RUSTUP_TOOLCHAIN=1.98.1
nvcc --version | grep -q 'release 13.0,'
mkdir /output/container-evidence
nvcc --list-gpu-arch > /output/container-evidence/nvcc-gpu-arch.txt
printf '%s\n' "${CUDA_VERSION:?CUDA builder must declare its toolkit release}" \
  > /output/container-evidence/cuda-environment-version.txt
dpkg-query -W -f='${binary:Package}\t${Version}\n' 'cuda-*' \
  > /output/container-evidence/cuda-packages.txt
nvcc --version > /output/container-evidence/nvcc.txt
cp /etc/os-release /output/container-evidence/os-release.txt
uname -m > /output/container-evidence/architecture.txt
cp /etc/os-release /output/ubuntu-os-release.txt
dpkg-query -W gcc g++ libc6 libstdc++6 binutils python3 > /output/ubuntu-packages.txt
ldd --version > /output/ubuntu-libc.txt
rustc -Vv > /output/ubuntu-rustc.txt
nvcc --version > /output/ubuntu-nvcc.txt
python3 /tools/build-native-gpu.py \
  --workspace /source --source-commit "$native_commit" \
  --cuda-root /usr/local/cuda --builder-id "$native_image" \
  --build-root /build/target --out /output/matrix --jobs "$native_jobs" "$@"
python3 /tools/build-native-gpu-container.py assemble /output/matrix \
  --source-root /source --source-commit "$native_commit" --image "$native_image" \
  --evidence /output/container-evidence
chown -R "$(stat -c %u /output):$(stat -c %g /output)" /output
