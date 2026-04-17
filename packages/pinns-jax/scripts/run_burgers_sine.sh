#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname -- "${SCRIPT_DIR}")"

CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/default.py}"
RUN_ID="${RUN_ID:-$(date +"%Y-%m-%dT%H-%M-%S")}"
WORKDIR="${WORKDIR:-${REPO_DIR}/_output/pinns_jax/burgers/experiments/${RUN_ID}}"

uv run python -m pinns_jax.main \
    --config="${CONFIG_PATH}" \
    --workdir="${WORKDIR}" \
    --alsologtostderr="true" \
    "$@"
