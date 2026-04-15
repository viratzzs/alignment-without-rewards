"""
Custom OPSD training script for KDFlow.

Key difference from standard on-policy KD:
  - The teacher prompt INCLUDES the correct answer (privileged info)
  - Teacher and student use the SAME model (self-distillation)
  - Registers the custom 'opsd' algorithm before training

This modifies the PromptDataset to inject the label into the teacher prompt,
creating the privileged-information conditioning described in the OPSD paper.
"""
import math
import sys
import os

# Register the custom OPSD algorithm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opsd_algorithm  # noqa: F401  — side effect: registers @register_algorithm("opsd")

import ray

from kdflow.ray.train.teacher_group import TeacherActorGroup
from kdflow.ray.train.student_group import StudentActorGroup
from kdflow.ray.rollout.rollout_group_vllm import RolloutActorGroupVLLM as RolloutActorGroup
from kdflow.ray.placement_group import create_placement_group
from kdflow.trainer import OnPolicyKDTrainer
from kdflow.datasets import PromptDataset
from kdflow.datasets.utils import blending_datasets, convert_to_openai_messages
from kdflow.models.utils import check_tokenizer_identical
from kdflow.backend import get_strategy
from kdflow.arguments import init_args
from kdflow.utils.distributed_sampler import DistributedSampler
from kdflow.utils.utils import get_tokenizer


class OPSDPromptDataset(PromptDataset):
    """
    Modified PromptDataset for On-Policy Self-Distillation.
    
    The key modification: the teacher prompt includes the correct answer
    (label) as privileged information. This creates the asymmetry between
    teacher and student that drives the self-distillation signal.
    
    Teacher sees: [system] + [user question] + [assistant: "The correct answer is {label}"]
    Student sees: [system] + [user question]  (no answer)
    """

    def process_data(self, data):
        """Override to inject privileged info into teacher prompt."""
        # Build student prompt normally
        if self.apply_chat_template:
            chat = convert_to_openai_messages(data[self.input_key])
            stu_prompt = self.student_processor.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking, # Follow global flag for student
            )
        else:
            stu_prompt = self._build_prompt(data, self.student_processor, self.input_key)
        
        # Build teacher prompt WITH privileged information (TM-on)
        label = data.get(self.label_key, "") if self.label_key else ""
        tea_prompt = self._build_teacher_prompt_with_privilege(
            data, self.teacher_processor or self.student_processor, label
        )
        
        # Compute prompt token length for filtering
        tokenizer = (
            self.student_processor.tokenizer 
            if hasattr(self.student_processor, "tokenizer") 
            else self.student_processor
        )
        prompt_len = len(tokenizer.encode(stu_prompt))

        result = {
            "stu_prompt": stu_prompt,
            "tea_prompt": tea_prompt,
            "prompt": stu_prompt,
            "prompt_len": prompt_len,
            "label": str(label),
            "datasource": data.get("datasource", "default"),
        }
        return result
    
    def _build_teacher_prompt_with_privilege(self, data, processor_or_tokenizer, label):
        """
        Build teacher prompt that includes the correct answer as privileged info.
        
        The teacher is conditioned on the correct answer and uses Thinking Mode
        to rationalize the solution.
        """
        if self.apply_chat_template:
            chat = convert_to_openai_messages(data[self.input_key])
            
            # Append the correct answer as privileged assistant context
            if label:
                privilege_msg = f"The correct answer is {label}. Now let me reason through this step by step."
                chat = chat + [{"role": "assistant", "content": privilege_msg}]
            
            return processor_or_tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True if not label else False,
                enable_thinking=True, # Thinking ALWAYS enabled for teacher rationalization
            )
        
        # Fallback for non-chat-template format
        prompt = data[self.input_key]
        if self.input_template:
            prompt = self.input_template.format(prompt)
        if label:
            prompt += f"\nThe correct answer is {label}."
        return prompt


def train(args):
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                },
            }
        )

    strategy = get_strategy(args)
    strategy.print(args)

    # Create placement group
    num_gpus = args.train.num_nodes * args.train.num_gpus_per_node
    pg, reordered_bundle_indices, reordered_gpu_ids = create_placement_group(num_gpus)

    rollout_group = RolloutActorGroup(
        model_path=args.model.student_name_or_path,
        num_actors=args.rollout.rollout_num_engines,
        tp_size=args.rollout.rollout_tp_size,
        num_gpus_per_node=args.train.num_gpus_per_node,
        enable_memory_saver=args.train.enable_sleep,
        mem_fraction_static=args.rollout.rollout_mem_fraction_static,
        num_gpus_per_actor=0.3,
        pg=(pg, reordered_bundle_indices, reordered_gpu_ids),
        max_model_len=args.data.max_len,
    )
    rollout_group.sleep()

    # For OPSD: teacher == student (same model path)
    teacher_model = TeacherActorGroup(
        strategy,
        num_gpus,
        num_gpus_per_node=args.train.num_gpus_per_node,
        num_gpus_per_actor=0.2,
        pg=(pg, reordered_bundle_indices, reordered_gpu_ids),
    )
    student_model = StudentActorGroup(
        args,
        args.train.num_nodes,
        args.train.num_gpus_per_node,
        pg=(pg, reordered_bundle_indices),
        num_gpus_per_actor=0.5,
    )

    # Removed the hack that bypassed vLLM off-policy rollout weight-syncing.
    # The RolloutActorGroupVLLM now supports synchronous sync-from-disk
    # seamlessly without deadlocking FSDP.

    # Initialize tokenizers
    student_tokenizer = get_tokenizer(
        args.model.student_name_or_path,
        use_fast=not args.model.disable_fast_tokenizer,
    )
    teacher_tokenizer = get_tokenizer(
        args.model.teacher_name_or_path,
        use_fast=not args.model.disable_fast_tokenizer,
    )
    tokenizer_info = check_tokenizer_identical(student_tokenizer, teacher_tokenizer)
    strategy.print(f"Tokenizers {tokenizer_info}")

    # Load training data
    train_data = blending_datasets(
        args.data.train_dataset_path,
        args.data.train_dataset_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.train_split,
    )
    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))

    # Use our custom OPSD dataset that injects privileged info into teacher prompt
    train_dataset = OPSDPromptDataset(
        train_data,
        strategy,
        tokenizer_info=tokenizer_info,
        max_data_num=args.data.max_samples,
        input_template=args.data.input_template,
        num_processors=None, # Force None to avoid multiprocessing completely
    )

    sampler = DistributedSampler(
        train_dataset, num_replicas=1, rank=0, shuffle=True, seed=args.train.seed, drop_last=True
    )
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.rollout.rollout_batch_size,
        True,
        False,
        collate_fn=train_dataset.collate_fn,
        sampler=sampler,
    )

    # Calculate max training steps
    num_rollout_iters_per_epoch = (
        len(train_dataset) * args.rollout.n_samples_per_prompt // args.rollout.rollout_batch_size
    )
    num_update_steps_per_rollout = (
        args.rollout.rollout_batch_size * args.rollout.n_samples_per_prompt // args.train.train_batch_size
    )
    max_rollout_iters = math.ceil(args.train.num_epochs * num_rollout_iters_per_epoch)
    strategy.log(f"Max training iterations: {max_rollout_iters}")

    # Initialize student model
    ray.get(
        student_model.async_init_model_from_pretrained(
            strategy,
            (max_rollout_iters * num_update_steps_per_rollout),
            teacher_tokenizer=teacher_tokenizer,
            tokenizer_info=tokenizer_info,
        )
    )
    strategy.log("Models initialized on all student actors")

    generate_kwargs = {
        "max_new_tokens": args.rollout.generate_max_len,
        "temperature": args.rollout.temperature,
        "top_p": args.rollout.top_p,
    }

    trainer = OnPolicyKDTrainer(
        strategy=strategy,
        student_model=student_model,
        teacher_model=teacher_model,
        rollout_group=rollout_group,
        is_same_tokenizer=tokenizer_info.is_identical,
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        max_rollout_iters=max_rollout_iters,
        num_rollout_iters_per_epoch=num_rollout_iters_per_epoch,
        generate_kwargs=generate_kwargs,
    )

    try:
        trainer.fit()
        ray.get(student_model.async_save_model())
        strategy.log("OPSD training completed and model saved.")
    finally:
        teacher_model.shutdown()
        rollout_group.shutdown()


if __name__ == "__main__":
    args = init_args()
    train(args)
