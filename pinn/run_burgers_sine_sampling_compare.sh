#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +"%Y-%m-%dT%H-%M-%S")}"
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/_output/pinn/burgers/sine_sampling_compare/${RUN_ID}}"
TORCH_SEED="${TORCH_SEED:-1}"
SAMPLING_STRATEGIES="${SAMPLING_STRATEGIES:-monte_carlo fixed_grid}"

run_experiment() {
    local sampling_strategy="$1"
    local experiment_name="$2"
    shift 2

    "${PYTHON_BIN}" -m pinn.main \
        --config="${SCRIPT_DIR}/config/burgers_sine.py" \
        --config.root_dir="${ROOT_DIR}" \
        --config.timestamp="${experiment_name}" \
        --config.torch_seed="${TORCH_SEED}" \
        --config.DataConfig.sampling_strategy="${sampling_strategy}" \
        --alsologtostderr="true" \
        "$@"
}

for sampling_strategy in ${SAMPLING_STRATEGIES}; do
    run_experiment "${sampling_strategy}" "${sampling_strategy}" "$@"
done
