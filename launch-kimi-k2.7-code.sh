#!/bin/bash

BASE_PATH="$HOME/code/spark-vllm-docker"
HOST_1=192.168.100.11
HOST_2=192.168.100.12
HOST_3=192.168.100.13
HOST_4=192.168.100.14
HOST_5=192.168.100.15
HOST_6=192.168.100.16
HOST_7=192.168.100.17
HOST_8=192.168.100.18

"$BASE_PATH/run-recipe.sh" kimi-k2.7-code \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    -n $HOST_1,$HOST_2,$HOST_3,$HOST_4,$HOST_5,$HOST_6,$HOST_7,$HOST_8
