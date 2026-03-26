import os

#os.environ["VLLM_MAX_MODEL_LEN"] = "19000"

# Launch script with accelerate instead
# CUDA_VISIBLE_DEVICES=0 accelerate launch grpo_run.py 
 
import torch
import pandas as pd

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer
from loguru import logger
from dotenv import load_dotenv

from rewards import *

load_dotenv()

SYSTEM_PROMPT = """\
You are a logical assistant who is good at critical thinking and problem solving. \
Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.
Rationalize your answer step by step, then provide only the final choice letter at the very end after \
'#### (Correct option number out of all 4 options)' 
For example, #### 1"""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLIT_FILES = {
    "train":      os.path.join(ROOT, "data/logiqa_train_kdflow.parquet"),
    "validation": os.path.join(ROOT, "data/logiqa_validation_kdflow.parquet"),
    "test":       os.path.join(ROOT, "data/logiqa_test_kdflow.parquet"),
}

def get_data(split="train") -> Dataset:
    """Load LogiQA from local parquet. Returns message lists so TRL formats completions
    as dicts for the reward functions."""
    df = pd.read_parquet(SPLIT_FILES[split])

    records = []
    for _, row in df.iterrows():
        answer_letter = str(row["label"]).strip().upper()
        correct_option = ord(answer_letter) - ord('A') + 1  # A→1, B→2, C→3, D→4
        records.append({
            "prompt": row["messages"],   # list of dicts — TRL applies template
            "correct_option": correct_option,
        })

    return Dataset.from_list(records)

dataset = get_data()


model_name = "Qwen/Qwen3-4B"

output_dir="outputs/Qwen3-4B-RL"
run_name="Qwen3-4B-RL"

if __name__ == "__main__":
    # 45k samples * 2 epochs * 2 generations = 180k samples
    # 180k / (2 batch size * 8 gradient accumulation * 2 workers) = 5624 steps (for 2 gpus)
    # 180k / (2 batch size * 8 gradient accumulation * 8 workers) = 1406 steps (for 8 gpus)
    # 180k / (2 batch size * 2 gradient accumulation * 8 workers) = 5624 steps (for 8 gpus)
    # UPDATED: 40k samples * 1 epoch * 2 generations = 80k samples
    # 80k / (2 batch size * 4 gradient accumulation * 8 workers) = 1250 steps (for 8 gpus)
    # 18k * 2 epochs * 2 generations = 72k samples
    # 72k / (2 batch size * 8 gradient accumulation * 4 workers) = 1125 steps (for 4 gpus)
    training_args = GRPOConfig(
        #importance_sampling_level="sequence", # GSPO implementation by Qwen team
        #loss_type="bnpo",
        #beta=0.1,
        #epsilon=0.2,
        output_dir=output_dir,
        run_name=run_name,
        use_vllm=True,
        vllm_mode="colocate",
        #vllm_mode="server",
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_length=4096,
        chat_template_kwargs={"enable_thinking": False},
        learning_rate=1e-6,
        temperature=0.7,
        top_p=0.9, # do 0.8 if nonsense generations happen
        top_k=-1,
        min_p=0.0,
        #presence_penalty=1.5,
        repetition_penalty=1.0,
        #weight_decay = 0.05, #exp
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        #use_liger_loss=True,  # liger only supports token level sampling, so incompatible with GSPO(sequence level sampling)
        logging_steps=1,
        per_device_train_batch_size=8,
        #gradient_accumulation_steps=4, # TODO: Add grad accumul. again if training too slow.
        num_generations=4,
        # max_prompt_length removed in TRL 0.29.1 — enforce via tokenizer.model_max_length=2048 below
        max_completion_length=3072,
        num_train_epochs=1,
        save_steps=500,
        # to prevent RTE: expected to mark a variable ready only once, add this in startup: --ddp_find_unused_parameters False
        #gradient_checkpointing=True, # TODO: add this back if OOM, else don't cus it makes training slow.
        ddp_find_unused_parameters=False,
        report_to="wandb",
        push_to_hub=True,
        #generation_kwargs={},
        log_unique_prompts=True,         # renamed from wandb_log_unique_prompts
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )#.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 2048  # replaces removed GRPOConfig.max_prompt_length

    trainer = GRPOTrainer(
        #model=model_name, # for vllm server mode
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            format_reward_func,
            int_answer_reward_func,
            no_repetition_reward_func,
            reasoning_quality_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,

    )
    
    #trainer.train(resume_from_checkpoint="/workspace/training-mvp/src/code-rl/outputs/Qwen8B-RL/checkpoint-600")
    trainer.train()

    repo_name = "ViratChauhan/Qwen3-4B-GRPO"

    logger.info("Saving full model...")
    trainer.save_model(f"{output_dir}/final_model")

    try:
        trainer.model.push_to_hub(
            repo_name,
            commit_message="push model",
            private=True,
        )

        trainer.tokenizer.push_to_hub(
            repo_name,
            commit_message="push tokenizer",
            private=True,
        )

        logger.info(f"Model pushed to: https://huggingface.co/{repo_name}")
    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")
        logger.info(f"Model saved locally in: {output_dir}/final_model")
