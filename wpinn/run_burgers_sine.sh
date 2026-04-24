#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export RELAXNN_DEFAULT_CUDA_DEVICE="${RELAXNN_DEFAULT_CUDA_DEVICE:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/burgers_sine.py}"
RUN_ID="${RUN_ID:-$(date +"%Y-%m-%dT%H-%M-%S")}"
TORCH_SEED="${TORCH_SEED:-1}"

uv run python -u -m wpinn.main \
    --config="${CONFIG_PATH}" \
    --config.timestamp="${RUN_ID}" \
    --config.torch_seed="${TORCH_SEED}" \
    --alsologtostderr="true" \
    "$@"
