#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/swe.py}"
RUN_ID="${RUN_ID:-$(date +"%Y-%m-%dT%H-%M-%S")}"
TORCH_SEED="${TORCH_SEED:-1}"
PYTHON_CMD=("${PYTHON_BIN:-python}")

"${PYTHON_CMD[@]}" -m pinn.main \
    --config="${CONFIG_PATH}" \
    --config.timestamp="${RUN_ID}" \
    --config.torch_seed="${TORCH_SEED}" \
    --alsologtostderr="true" \
    "$@"
