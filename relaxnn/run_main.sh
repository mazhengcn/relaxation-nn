#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="0"

TIMESTAMP="$(date +"%Y-%m-%dT%H-%M-%S")"


python main.py \
    --config=cf_euler.py \
    --config.model="euler_v2" \
    --config.root_dir="./_output/euler_v2/osher" \
    --config.DataConfig.seed=1 \
    --config.torch_seed=1 \
    --config.timestamp="${TIMESTAMP}" \
    --alsologtostderr="true"