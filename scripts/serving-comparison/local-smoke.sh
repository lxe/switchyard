#!/usr/bin/env bash
set -euo pipefail
engine=$1
runtime=$2
root=$(pwd)
cache="$root/results/local-$engine-cache"
mkdir -p "$cache"
timeout 500 docker run --rm --init --name "switchyard-serving-$engine-gpu" \
  --gpus all --network none --read-only --cpus 8 --memory 24g --shm-size 2g \
  --tmpfs /tmp:rw,exec,size=4g \
  --mount "type=bind,source=$root/$runtime,target=/opt/switchyard/runtime,readonly" \
  --mount "type=bind,source=$root/results/local-model/model,target=/model,readonly" \
  --mount "type=bind,source=$cache,target=/cache" \
  --mount "type=bind,source=$root/configs/runpod/serving-comparison/$engine.json,target=/config.json,readonly" \
  --mount "type=bind,source=$root/scripts/serving-comparison/local-smoke.py,target=/local-smoke.py,readonly" \
  --env SWITCHYARD_CACHE_ROOT=/cache --env MAX_JOBS=2 \
  switchyard-hydration-smoke:local /bin/bash -c \
  'source /opt/switchyard/runtime/runtime-env.sh && exec python /local-smoke.py'
