#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="$(date +"%Y-%m-%dT%H-%M-%S")"

"${PYTHON_BIN}" -m relaxnn.main \
    --config="${SCRIPT_DIR}/config/burgers.py" \
    --config.model="burgers" \
    --config.root_dir="${REPO_ROOT}/_output/relaxnn/burgers/riemann" \
    --config.torch_seed=1 \
    --config.timestamp="${TIMESTAMP}" \
    --alsologtostderr="true"
