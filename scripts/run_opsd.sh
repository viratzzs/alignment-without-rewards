#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  On-Policy Self-Distillation (OPSD) via KDFlow
#  Teacher == Student: Qwen/Qwen3-4B (self-distillation with privileged info)
#
#  Key difference from OPD:
#    - Teacher sees the correct answer as privileged information
#    - Uses JSD loss with pointwise KL clipping for training stability
#    - Custom entry point: opsd/train_opsd.py
#
#  Prerequisites:
#    1. pip install -e /home/KDFlow
#    2. python data/prepare_logiqa_kdflow.py
#    3. ray start --head --node-ip-address 0.0.0.0 --num-gpus 1
# ──────────────────────────────────────────────────────────────────────────────
set -e
set -x

# Load env vars (HF_TOKEN, WANDB_API_KEY, etc.)
set -a; source "$(dirname "$0")/../.env"; set +a

# Ensure KDFlow is in path
export PYTHONPATH=$PYTHONPATH:/home/KDFlow

#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ============ Training ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 1"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 32"
OPTS+=" --micro_train_batch_size 2"
OPTS+=" --learning_rate 2e-6"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 3"
OPTS+=" --save_path ./outputs/opsd-qwen3-4b"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --train_enable_sleep True"
OPTS+=" --packing_samples True"
OPTS+=" --teacher_update_freq 1000000"

# ============ Model (self-distillation: teacher == student) ============
OPTS+=" --student_name_or_path Qwen/Qwen3-4B"
OPTS+=" --teacher_name_or_path Qwen/Qwen3-4B"
OPTS+=" --enable_thinking True"

# ============ Rollout ============
OPTS+=" --rollout_batch_size 256"
OPTS+=" --rollout_num_engines 1"
OPTS+=" --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.15"
OPTS+=" --rollout_enable_sleep False"
OPTS+=" --n_samples_per_prompt 1"
OPTS+=" --generate_max_len 8192"

# ============ Data ============
OPTS+=" --train_dataset_path ./data/logiqa_train_kdflow.parquet"
OPTS+=" --max_len 8192"
OPTS+=" --prompt_max_len 2048"
OPTS+=" --input_key messages"
OPTS+=" --label_key label"
OPTS+=" --apply_chat_template True"

# ============ Distillation (OPSD-specific) ============
# JSD loss with beta=0.5 (symmetric blend of forward+reverse KL)
# Pointwise KL clip prevents stylistic tokens from dominating gradients
OPTS+=" --kd_ratio 1.0"
OPTS+=" --kd_loss_fn jsd"
OPTS+=" --jsd_beta 0.5"
OPTS+=" --kd_algorithm opsd"
OPTS+=" --pointwise_kl_clip 10.0"
OPTS+=" --teacher_tp_size 1"
OPTS+=" --teacher_dp_size 1"
OPTS+=" --teacher_mem_fraction_static 0.15"
OPTS+=" --teacher_enable_sleep True"

# ============ Logging ============
OPTS+=" --logging_steps 1"
OPTS+=" --use_wandb True"
OPTS+=" --wandb_project alignment-opsd"
OPTS+=" --wandb_group on_policy_self_distillation"
OPTS+=" --wandb_run_name opsd-qwen3-4b"
OPTS+=" --wandb_mode online"

echo "Starting OPSD Training via KDFlow..."
/home/alignment-without-rewards/.venv/bin/python scripts/opsd/train_opsd.py $OPTS

echo "Training finished."
LATEST_CKPT=$(ls -td outputs/opsd-qwen3-4b/epoch_* 2>/dev/null | head -1)
PUSH_DIR="${LATEST_CKPT:-outputs/opsd-qwen3-4b}"
echo "Pushing checkpoint from: $PUSH_DIR"
python scripts/push_to_hub.py \
    --ckpt-dir "$PUSH_DIR" \
    --repo-name "ViratChauhan/Qwen3-4B-OPSD"
