#!/bin/bash
# Stops the current model being served in whole Spark cluster, workers first (.18 -> .12), head (.11) last.
set -u

BASE_PATH="$HOME/code/spark-vllm-docker"

read -r -p "Stop the cluster on all 8 nodes? [y/N] " ans
[ "${ans,,}" = "y" ] || { echo "aborted"; exit 1; }

# Stop the serving cluster cleanly first (ignore errors if nothing is running)
if [ -x "$BASE_PATH/launch-cluster.sh" ]; then
  echo "stopping vLLM cluster..."
  ("$BASE_PATH/launch-cluster.sh" stop) || true
fi

