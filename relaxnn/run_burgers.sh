#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/burgers.py}"
SEEDS="${SEEDS:-42 123 3407 2024 1337}"

SEEDS="${SEEDS//,/ }"
read -r -a SEED_LIST <<< "${SEEDS}"

if [ "${#SEED_LIST[@]}" -eq 0 ]; then
    echo "No seeds specified via SEEDS." >&2
    exit 1
fi

for TORCH_SEED in "${SEED_LIST[@]}"; do
    RUN_ID="$(date +"%Y-%m-%dT%H-%M-%S")-seed${TORCH_SEED}"
    echo "Running RelaxNN Burgers with seed ${TORCH_SEED} -> ${RUN_ID}"
    "${PYTHON_BIN}" -m relaxnn.main \
        --config="${CONFIG_PATH}" \
        --config.torch_seed="${TORCH_SEED}" \
        --config.timestamp="${RUN_ID}" \
        --alsologtostderr="true" \
        "$@"
done
