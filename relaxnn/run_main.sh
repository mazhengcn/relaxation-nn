#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="0"

TIMESTAMP="$(date +"%Y-%m-%dT%H-%M-%S")"


python main.py \
    --config=./config/euler_2d.py \
    --config.model="2d_euler" \
    --config.root_dir="./_output/euler/riemann" \
    --config.DataConfig.seed=1 \
    --config.torch_seed=1 \
    --config.timestamp="${TIMESTAMP}" \
    --alsologtostderr="true"