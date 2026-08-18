#!/bin/bash

HOST_1=192.168.100.27
HOST_2=192.168.100.28
BASE_PATH="$HOME/code/spark-vllm-docker"

"$BASE_PATH/run-recipe.sh" deepseek-v4-flash-0731-abliterated \
    --max-model-len 524288 \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    -n $HOST_1,$HOST_2 