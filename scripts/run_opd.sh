#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  On-Policy Distillation (OPD) via KDFlow
#  Teacher: Qwen/Qwen3-30B-A3B   Student: Qwen/Qwen3-4B
#
#  Prerequisites:
#    1. pip install -e /home/KDFlow
#    2. python data/prepare_logiqa_kdflow.py   (regenerates *_kdflow.parquet)
#    3. ray start --head --node-ip-address 0.0.0.0 --num-gpus 1
# ──────────────────────────────────────────────────────────────────────────────
set -e
set -x

# Load env vars (HF_TOKEN, WANDB_API_KEY, etc.)
set -a; source "$(dirname "$0")/../.env"; set +a

# ============ Training ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 1"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 32" # edited
OPTS+=" --micro_train_batch_size 4"
OPTS+=" --learning_rate 1e-6"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 1"
OPTS+=" --save_path ./outputs/opd-qwen3-4b"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --train_enable_sleep True"
OPTS+=" --packing_samples True" # edited

# ============ Model ============
OPTS+=" --student_name_or_path Qwen/Qwen3-4B"
OPTS+=" --teacher_name_or_path Qwen/Qwen3-30B-A3B"
OPTS+=" --enable_thinking False"

# ============ Rollout ============
OPTS+=" --rollout_batch_size 64" # edited
OPTS+=" --rollout_num_engines 1"
OPTS+=" --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.5"
OPTS+=" --rollout_enable_sleep True"
OPTS+=" --n_samples_per_prompt 1"
OPTS+=" --generate_max_len 2048"

# ============ Data ============
OPTS+=" --train_dataset_path ./data/logiqa_train_kdflow.parquet"
OPTS+=" --max_len 4096"
OPTS+=" --prompt_max_len 2048"
OPTS+=" --input_key messages"
OPTS+=" --label_key label"
OPTS+=" --apply_chat_template True"

# ============ Distillation ============
OPTS+=" --kd_ratio 1.0"
OPTS+=" --kd_loss_fn rkl"
OPTS+=" --kd_algorithm vanilla_kd"
OPTS+=" --teacher_tp_size 1"
OPTS+=" --teacher_dp_size 1"
OPTS+=" --teacher_mem_fraction_static 0.4"
OPTS+=" --teacher_enable_sleep True"

# ============ Logging ============
OPTS+=" --logging_steps 1"
OPTS+=" --use_wandb True"
OPTS+=" --wandb_project alignment-opd"
OPTS+=" --wandb_group on_policy_distillation"
OPTS+=" --wandb_run_name opd-qwen3-4b"
OPTS+=" --wandb_mode online"

echo "Starting OPD Training via KDFlow..."
python -m kdflow.cli.train_kd_on_policy $OPTS

echo "Training finished."
LATEST_CKPT=$(ls -td outputs/opd-qwen3-4b/epoch_* 2>/dev/null | head -1)
PUSH_DIR="${LATEST_CKPT:-outputs/opd-qwen3-4b}"
echo "Pushing checkpoint from: $PUSH_DIR"
python scripts/push_to_hub.py \
    --ckpt-dir "$PUSH_DIR" \
    --repo-name "ViratChauhan/Qwen3-4B-OPD"
