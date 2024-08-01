#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="0"

TIMESTAMP="$(date +"%Y-%m-%dT%H-%M-%S")"


python main.py \
    --config=./config/burgers.py \
    --config.model="burgers" \
    --config.root_dir="./_output/burgers/riemann" \
    --config.torch_seed=1 \
    --config.timestamp="${TIMESTAMP}" \
    --alsologtostderr="true"