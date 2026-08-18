#!/bin/bash

BASE_PATH="$HOME/code/spark-vllm-docker"

"$BASE_PATH/run-recipe.sh" qwen3.6-35b-a3b-nvfp4 \
    --tensor-parallel-size 1 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --tool-call-parser qwen3_coder