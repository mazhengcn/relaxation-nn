#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && cd .. && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/euler.py}"
RUN_ID="${RUN_ID:-$(date +"%Y-%m-%dT%H-%M-%S")}"
TORCH_SEED="${TORCH_SEED:-1}"

relaxnn --config="${CONFIG_PATH}" \
    --config.torch_seed="${TORCH_SEED}" \
    --config.timestamp="${RUN_ID}" \
    --alsologtostderr="true" \
    "$@"
