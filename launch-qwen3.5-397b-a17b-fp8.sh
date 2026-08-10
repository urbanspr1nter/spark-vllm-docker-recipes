#!/bin/bash

HOST_1=192.168.100.23
HOST_2=192.168.100.24
HOST_3=192.168.100.25
HOST_4=192.168.100.26
BASE_PATH="$HOME/code/spark-vllm-docker"

"$BASE_PATH/run-recipe.sh" $BASE_PATH/recipes/4x-spark-cluster/qwen3.5-397b-a17B-fp8 \
    --no-ray \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.88 \
    --earlyoom --earlyoom-args "-M 262144,102400 -s 100 -r 60" \
    -n $HOST_1,$HOST_2,$HOST_3,$HOST_4