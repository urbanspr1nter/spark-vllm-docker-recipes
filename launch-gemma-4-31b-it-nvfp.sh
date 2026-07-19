#!/bin/bash

HOST_1=192.168.100.23
HOST_2=192.168.100.24
BASE_PATH="$HOME/code/spark-vllm-docker"

"$BASE_PATH/run-recipe.sh" gemma4-31b-it-nvfp4 \
    --no-ray \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    -n $HOST_1,$HOST_2 