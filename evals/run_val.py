"""
Evaluation script for the validation split of lucasmccabe/logiqa.

Usage:
    python run_val.py <model_name_or_path> [--max-samples N] [--max-tokens 750] [--batch-size 4]

Example:
    python run_val.py Qwen/Qwen3.5-4B
    python run_val.py ./outputs/Qwen3.5-4B-RL/final_model --max-samples 100
"""

import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


SYSTEM_PROMPT = """\
You are a logical assistant who is good at critical thinking and problem solving. \
Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.
Rationalize your answer, then provide the final choice letter at the very end after \
'#### (Correct option number out of all 4 options)' (eg., #### 1)"""


def extract_hash_answer(text: str) -> str | None:
    """Extract the answer after #### from model output."""
    if "####" not in text:
        return None
    raw = text.split("####")[-1].strip()
    match = re.match(r"(\d+)", raw)
    return match.group(1) if match else None


def format_options(options: list[str]) -> str:
    """Format options as 1-indexed numbered list."""
    return "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))


def build_prompt(sample: dict) -> str:
    """Build the user prompt from a dataset sample."""
    return (
        f"Context: {sample['context']}\n"
        f"Question: {sample['query']}\n"
        f"Options:\n{format_options(sample['options'])}"
    )


def generate_response(model, tokenizer, sample: dict, max_tokens: int, device: str) -> str:
    """Generate a single response for a dataset sample."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(sample)},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy for deterministic eval
            temperature=None,
            top_p=None,
        )

    # decode only the new tokens (strip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a HuggingFace model on LogiQA validation split")
    parser.add_argument("model", type=str, help="HuggingFace model name or local path")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate (default: all)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens to generate (default: 1024)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (default: bfloat16)")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading validation split of lucasmccabe/logiqa...")
    data = load_dataset("lucasmccabe/logiqa", split="validation")
    if args.max_samples is not None:
        data = data.select(range(min(args.max_samples, len(data))))
    print(f"Evaluating on {len(data)} samples")

    # ── Load model & tokenizer ────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    # ── Evaluation loop ───────────────────────────────────────────────────
    correct = 0
    total = 0
    no_answer = 0

    for idx in tqdm(range(len(data)), desc="Evaluating"):
        sample = data[idx]

        # Ground truth: dataset uses 0-indexed, we use 1-indexed
        gt = sample["correct_option"] + 1

        response = generate_response(model, tokenizer, sample, args.max_tokens, device)
        extracted = extract_hash_answer(response)

        if extracted is None:
            no_answer += 1
            is_correct = False
        else:
            is_correct = (int(extracted) == gt)

        if is_correct:
            correct += 1
        total += 1

        # Print first few and any noteworthy examples
        if idx < 3 or (idx < 50 and not is_correct):
            print(f"\n{'─' * 60}")
            print(f"Sample {idx}")
            print(f"Question: {sample['query'][:200]}...")
            print(f"Ground truth: {gt}")
            print(f"Response (last 200 chars): ...{response[-200:]}")
            print(f"Extracted: {extracted}")
            print(f"Correct: {is_correct}")

    # ── Results ───────────────────────────────────────────────────────────
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n{'═' * 60}")
    print(f"MODEL: {args.model}")
    print(f"SPLIT: validation")
    print(f"TOTAL: {total}")
    print(f"CORRECT: {correct}")
    print(f"NO ANSWER: {no_answer}")
    print(f"ACCURACY: {accuracy:.2f}%")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()