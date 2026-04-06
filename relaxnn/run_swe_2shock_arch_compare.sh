#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

uv run --extra train python -m relaxnn.swe_2shock_arch_experiments \
    --root_dir="${REPO_ROOT}/_output/relaxnn/swe/2shock_arch_compare/$(date +"%Y-%m-%dT%H-%M-%S")" \
    "$@"
