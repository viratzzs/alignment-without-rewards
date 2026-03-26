#!/bin/bash
set -e
set -x

# ──────────────────────────────────────────────────────────────────────────────
#  On-Policy Self-Distillation (OPSD) via KDFlow
#  Teacher == Student: Qwen/Qwen3-4B  (self-distillation with privileged info)
#
#  Key difference from OPD:
#    - Uses the custom train_opsd.py entry point (not kdflow CLI)
#    - Teacher receives the correct answer as privileged information
#    - Uses JSD loss with pointwise KL clipping for stability
#
#  Prerequisite:
#    1. pip install -e /path/to/KDFlow
#    2. python data/prepare_logiqa_kdflow.py
#    3. ray start --head --node-ip-address 0.0.0.0 --num-gpus <N>
# ──────────────────────────────────────────────────────────────────────────────

# ============ Training ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 1"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 32"
OPTS+=" --micro_train_batch_size 4"
OPTS+=" --learning_rate 2e-5"
OPTS+=" --lr_warmup_ratio 0.1"
OPTS+=" --num_epochs 3"
OPTS+=" --save_path ./outputs/opsd-qwen3-4b"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --enable_sleep True"

# ============ Model (self-distillation: teacher == student) ============
OPTS+=" --student_name_or_path Qwen/Qwen3-4B"
OPTS+=" --teacher_name_or_path Qwen/Qwen3-4B"
OPTS+=" --enable_thinking False"

# ============ Rollout ============
OPTS+=" --rollout_batch_size 32"
OPTS+=" --rollout_num_engines 1"
OPTS+=" --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.5"
OPTS+=" --rollout_enable_sleep True"
OPTS+=" --n_samples_per_prompt 1"
OPTS+=" --generate_max_len 512"

# ============ Data ============
OPTS+=" --train_dataset_path ./data/logiqa_train_kdflow.parquet"
OPTS+=" --max_len 2560"
OPTS+=" --prompt_max_len 2048"
OPTS+=" --input_key messages"
OPTS+=" --label_key label"
OPTS+=" --apply_chat_template True"

# ============ Distillation (OPSD-specific settings) ============
OPTS+=" --kd_ratio 1.0"
OPTS+=" --kd_loss_fn jsd"
OPTS+=" --jsd_beta 0.5"
OPTS+=" --kd_algorithm opsd"
OPTS+=" --teacher_tp_size 1"
OPTS+=" --teacher_dp_size 1"
OPTS+=" --teacher_mem_fraction_static 0.4"
OPTS+=" --teacher_enable_sleep True"

# ============ Logging ============
OPTS+=" --logging_steps 1"
OPTS+=" --use_wandb True"
OPTS+=" --wandb_project subnetwork-alignment"
OPTS+=" --wandb_group on_policy_self_distillation"
OPTS+=" --wandb_run_name opsd-qwen3-4b"
OPTS+=" --wandb_mode online"

echo "Starting OPSD Training..."
python opsd/train_opsd.py $OPTS

echo "Training finished."
LATEST_CKPT=$(ls -td outputs/opsd-qwen3-4b/epoch_* 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Latest checkpoint: $LATEST_CKPT"
    python scripts/push_to_hub.py \
        --ckpt-dir "$LATEST_CKPT" \
        --repo-name "ViratChauhan/Qwen3-4B-opsd"
fi
