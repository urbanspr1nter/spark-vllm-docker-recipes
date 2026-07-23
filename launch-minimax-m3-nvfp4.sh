#!/bin/bash

BASE_PATH="$HOME/code/spark-vllm-docker"
HOST_1=192.168.100.21
HOST_2=192.168.100.22
HOST_3=192.168.100.23
HOST_4=192.168.100.24

"$BASE_PATH/run-recipe.sh" minimax-m3-nvfp4 \
    --ib-if rocep1s0f0,roceP2p1s0f0 \
    --eth-if enp1s0f0np0 \
    -n $HOST_1,$HOST_2,$HOST_3,$HOST_4
