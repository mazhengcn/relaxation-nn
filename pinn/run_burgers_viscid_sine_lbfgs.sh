#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +"%Y-%m-%dT%H-%M-%S")}"
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/_output/pinn/burgers/viscid_sine_lbfgs}"
TORCH_SEED="${TORCH_SEED:-1}"

"${PYTHON_BIN}" -m pinn.main \
    --config="${SCRIPT_DIR}/config/burgers_viscid_sine_lbfgs.py" \
    --config.root_dir="${ROOT_DIR}" \
    --config.timestamp="${RUN_ID}" \
    --config.torch_seed="${TORCH_SEED}" \
    --alsologtostderr="true" \
    "$@"
