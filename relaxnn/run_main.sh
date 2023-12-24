#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="0"

TIMESTAMP="$(date +"%Y-%m-%dT%H-%M-%S")"


python ./relaxnn/main.py \
    --config=./relaxnn/config/euler.py \
    --config.model="euler_v1" \
    --config.root_dir="./relaxnn/_output/euler_v1/sod" \
    --config.DataConfig.seed=1 \
    --config.torch_seed=1 \
    --config.timestamp="${TIMESTAMP}" \
    --alsologtostderr="true"