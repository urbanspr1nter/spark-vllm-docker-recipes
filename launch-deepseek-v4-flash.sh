#!/bin/bash

HOST_1=192.168.100.21
HOST_2=192.168.100.22
BASE_PATH="$HOME/code/spark-vllm-docker"

"$BASE_PATH/run-recipe.sh" deepseek-v4-flash \
    --no-ray \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    --max-model-len 1000000 \
    -n $HOST_1,$HOST_2 