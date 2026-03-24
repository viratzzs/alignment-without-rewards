#!/bin/bash
set -e

echo "Starting OPSD Training..."
python3 custom_kd_trainer.py \
  --config-dir="${PWD}/config" \
  --config-name='opsd_trainer.yaml' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node=1

echo "Training finished. Locating last checkpoint..."
LATEST_CKPT=$(ls -td subnetwork-alignment/opsd-qwen3-4b/global_step_* 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo "Latest checkpoint found at: $LATEST_CKPT"
    python3 scripts/push_to_hub.py \
        --ckpt-dir "$LATEST_CKPT" \
        --repo-name "ViratChauhan/Qwen3-4B-opsd"
else
    echo "Warning: Could not find any saved checkpoint in subnetwork-alignment/opsd-qwen3-4b/"
fi
