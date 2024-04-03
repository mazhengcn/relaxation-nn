#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="0"

TIMESTAMP="$(date +"%Y-%m-%dT%H-%M-%S")"


python main.py \
    --config=./config/euler_v1.py \
    --config.model="euler_v1" \
    --config.root_dir="./_output/euler/sod" \
    --config.DataConfig.seed=1 \
    --config.torch_seed=1 \
    --config.timestamp="${TIMESTAMP}" \
    --alsologtostderr="true"