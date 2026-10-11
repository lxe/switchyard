# Source this file to use the fixed-prefix runtime outside the Rust agent.
export SWITCHYARD_RUNTIME_ROOT=/opt/switchyard/runtime
export PATH="$SWITCHYARD_RUNTIME_ROOT/venv/bin:$SWITCHYARD_RUNTIME_ROOT/toolchain/bin:$SWITCHYARD_RUNTIME_ROOT/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$SWITCHYARD_RUNTIME_ROOT/lib:$SWITCHYARD_RUNTIME_ROOT/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$SWITCHYARD_RUNTIME_ROOT/toolchain/bin/gcc"
export CXX="$SWITCHYARD_RUNTIME_ROOT/toolchain/bin/g++"
export CUDA_HOME="$SWITCHYARD_RUNTIME_ROOT/cuda"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export SSL_CERT_FILE="$SWITCHYARD_RUNTIME_ROOT/certs/ca-certificates.crt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# The verified runtime includes immutable hash-based bytecode for both library trees.
export PYTHONDONTWRITEBYTECODE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# A writable cache volume may override this value before sourcing the file.
export SWITCHYARD_CACHE_ROOT="${SWITCHYARD_CACHE_ROOT:-/opt/switchyard/cache}"
export HF_HOME="$SWITCHYARD_CACHE_ROOT/huggingface"
export TRITON_CACHE_DIR="$SWITCHYARD_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$SWITCHYARD_CACHE_ROOT/torchinductor"
export VLLM_CACHE_ROOT="$SWITCHYARD_CACHE_ROOT/vllm"
export CUDA_CACHE_PATH="$SWITCHYARD_CACHE_ROOT/cuda"
# FlashInfer appends .cache/flashinfer and its version/architecture workspace.
export FLASHINFER_WORKSPACE_BASE="$SWITCHYARD_CACHE_ROOT/flashinfer"
