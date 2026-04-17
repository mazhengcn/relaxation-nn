#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/euler.py}"
RUN_ID="${RUN_ID:-$(date +"%Y-%m-%dT%H-%M-%S")}"
TORCH_SEED="${TORCH_SEED:-1}"

"${PYTHON_BIN}" -m relaxnn.main \
    --config="${CONFIG_PATH}" \
    --config.torch_seed="${TORCH_SEED}" \
    --config.timestamp="${RUN_ID}" \
    --alsologtostderr="true" \
    "$@"
