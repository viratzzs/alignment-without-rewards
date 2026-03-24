import os

#os.environ["VLLM_MAX_MODEL_LEN"] = "19000"

# Launch script with accelerate instead
# CUDA_VISIBLE_DEVICES=0 accelerate launch grpo_run.py 
 
import torch

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer
from loguru import logger

from rewards import *


SYSTEM_PROMPT = """
You are a logical assistant that is good at critical thinking and problem solving. Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.
Rationalize your answer step-by-step, then provide the final choice letter at the very end after '#### (Correct option number from 1 to 4)' (eg., #### 1)
"""

def get_data(split = "train") -> Dataset:
    data = load_dataset('lucasmccabe/logiqa', split=split)#.select(range(1000))
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Context: {x['context']}\\nQuestion: {x['query']}\nOptions:\n{x['options']}"}
        ],
    })
    return data

dataset = get_data()

model_name = "Qwen/Qwen3.5-4B"

output_dir="outputs/Qwen3.5-4B-RL"
run_name="Qwen3.5-4B-RL"

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
        vllm_gpu_memory_utilization=0.7,
        learning_rate=6e-5,
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        #weight_decay = 0.05, #exp
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        use_liger_loss=True,  # liger only supports token level sampling, so incompatible with GSPO(sequence level sampling)
        logging_steps=1,
        per_device_train_batch_size=32,
        #gradient_accumulation_steps=4, # TODO: Add grad accumul. again if training too slow.
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=512,
        num_train_epochs=2,
        save_steps=500,
        # to prevent RTE: expected to mark a variable ready only once, add this in startup: --ddp_find_unused_parameters False
        #gradient_checkpointing=True, # TODO: add this back if OOM, else don't cus it makes training slow.
        ddp_find_unused_parameters=False,
        report_to="wandb",
        push_to_hub=True,
        #generation_kwargs={},
        wandb_log_unique_prompts=True,
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )#.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    trainer = GRPOTrainer(
        #model=model_name, # for vllm server mode
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            format_reward_func,
            int_answer_reward_func,
            #reasoning_quality_reward_func,
            #no_repetition_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,

    )
    
    #trainer.train(resume_from_checkpoint="/workspace/training-mvp/src/code-rl/outputs/Qwen8B-RL/checkpoint-600")
    trainer.train()

    repo_name = "Viratzzs/Qwen3.5-4B-RL"

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

        logger.success(f"Model successfully pushed to: https://huggingface.co/{repo_name}")
    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")
        logger.info(f"Model saved locally in: {output_dir}/final_model")
