#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/burgers_sine.py}"
SEEDS="${SEEDS:-42 123 3407 2024 1337}"
EPOCHS="${EPOCHS:-80000}"
COMPILE_TRAINING="${COMPILE_TRAINING:-true}"
COMPILE_FALLBACK_TO_EAGER="${COMPILE_FALLBACK_TO_EAGER:-false}"

SEEDS="${SEEDS//,/ }"
read -r -a SEED_LIST <<< "${SEEDS}"

if [ "${#SEED_LIST[@]}" -eq 0 ]; then
    echo "No seeds specified via SEEDS." >&2
    exit 1
fi

for TORCH_SEED in "${SEED_LIST[@]}"; do
    RUN_ID="$(date +"%Y-%m-%dT%H-%M-%S")"
    echo "Running WPINN with seed ${TORCH_SEED} -> ${RUN_ID}"
    uv run python -u -m wpinn.main \
        --config="${CONFIG_PATH}" \
        --alsologtostderr="true" \
        "$@" \
        --config.timestamp="${RUN_ID}" \
        --config.torch_seed="${TORCH_SEED}" \
        --config.TrainConfig.epochs="${EPOCHS}" \
        --config.TrainConfig.compile_training="${COMPILE_TRAINING}" \
        --config.TrainConfig.compile_fallback_to_eager="${COMPILE_FALLBACK_TO_EAGER}"
done
