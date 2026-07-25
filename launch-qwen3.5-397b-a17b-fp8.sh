#!/bin/bash

HOST_1=192.168.100.21
HOST_2=192.168.100.22
HOST_3=192.168.100.23
HOST_4=192.168.100.24
BASE_PATH="$HOME/code/spark-vllm-docker"

"$BASE_PATH/run-recipe.sh" $BASE_PATH/recipes/4x-spark-cluster/qwen3.5-397b-a17B-fp8 \
    --no-ray \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    --earlyoom --earlyoom-args "-M 2097152,524288 -s 100 -r 60" \
    --gpu-memory-utilization 0.82 \
    -n $HOST_1,$HOST_2,$HOST_3,$HOST_4