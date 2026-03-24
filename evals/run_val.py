"""
Evaluation script for the validation split of lucasmccabe/logiqa.

Uses vLLM for fast batched generation.

Usage:
    python run_val.py <model_name_or_path> [--max-samples N] [--max-tokens 1024] [--batch-size 50]

Example:
    python run_val.py Qwen/Qwen3.5-4B
    python run_val.py ./outputs/Qwen3.5-4B-RL/final_model --max-samples 100
"""

import argparse
import os
import re
import urllib.request
from loguru import logger
from tqdm import tqdm
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """\
You are a logical assistant who is good at critical thinking and problem solving. \
Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.
Rationalize your answer step by step, then provide only the final choice letter at the very end after \
'#### (Correct option number out of all 4 options)' 
For example, #### 1"""


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


def build_messages(sample: dict) -> list[dict]:
    """Build chat messages for a dataset sample."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(sample)},
    ]

def load_logiqa(url: str) -> list[dict]:
    """Download and parse a raw LogiQA text file into a list of sample dicts."""
    def process_sent(text: str) -> str:
        text = text.replace("\n", "")
        sents = text.split(".")
        out = ""
        for sent in sents:
            if not sent: continue
            if not out: out += sent
            elif sent[0].isnumeric(): out += "." + sent
            else: out += ". " + sent
        out = out.replace("  ", " ").replace("\\'", "'").rstrip()
        if re.match(r'^[A-Z][\w\s]+[?.!]$', out) is None:
            out += "."
        return out.replace("?.", "?").replace("!.", "!").replace("..", ".")

    def process_answer(a: str) -> str:
        return a[3:] if a[:1] in "ABCD" else a

    with urllib.request.urlopen(url) as resp:
        lines = [process_sent(l.decode("utf-8")) for l in resp.readlines()]

    samples = []
    for i in range(len(lines) // 8):
        base = i * 8
        correct_option = "abcd".index(lines[base + 1].replace(".", ""))
        samples.append({
            "context": lines[base + 2],
            "query": lines[base + 3],
            "options": [process_answer(lines[base + 4 + j]) for j in range(4)],
            "correct_option": correct_option,
            "options_len": 4
        })
    return samples

def main():
    parser = argparse.ArgumentParser(description="Evaluate a HuggingFace model on LogiQA validation split")
    parser.add_argument("model", type=str, help="HuggingFace model name or local path")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate (default: all)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens to generate (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of samples per vLLM batch (default: 50)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (default: bfloat16)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size for multi-GPU (default: 1)")
    args = parser.parse_args()

    # ── Setup Logging ──────────────────────────────────────────────────────
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    safe_model_name = args.model.replace("/", "_")
    log_file = os.path.join(log_dir, f"{safe_model_name}_val.log")
    
    # Optional: configure logger to output to both console and file
    # logger.add(log_file, mode="w") will add the file sink
    logger.add(log_file, mode="w")
    logger.info(f"Logging evaluation results to {log_file}")

    # ── Load dataset ──────────────────────────────────────────────────────
    logger.info(f"Loading validation split of lucasmccabe/logiqa...")
    url = "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt"
    data = load_logiqa(url)

    if args.max_samples is not None:
        data = data[:args.max_samples]
    logger.info(f"Evaluating on {len(data)} samples")

    # ── Load model via vLLM ───────────────────────────────────────────────
    logger.info(f"Loading model via vLLM: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        max_tokens=args.max_tokens,
    )
    tokenizer = llm.get_tokenizer()

    # ── Build all prompts ─────────────────────────────────────────────────
    logger.info("Building prompts...")
    all_prompts = []
    for idx in range(len(data)):
        messages = build_messages(data[idx])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        all_prompts.append(prompt)

    # ── Batched generation ────────────────────────────────────────────────
    correct = 0
    total = 0
    no_answer = 0
    wrong_answer = 0
    all_outputs: list[str] = []

    num_batches = (len(all_prompts) + args.batch_size - 1) // args.batch_size
    logger.info(f"Running {num_batches} batches of up to {args.batch_size} samples each...")

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(all_prompts))
        batch_prompts = all_prompts[start:end]

        outputs = llm.generate(batch_prompts, sampling_params)
        for output in outputs:
            all_outputs.append(output.outputs[0].text)

    # ── Scoring ───────────────────────────────────────────────────────────
    for idx in range(len(data)):
        sample = data[idx]
        gt = sample["correct_option"] + 1  # dataset is 0-indexed, we use 1-indexed
        response = all_outputs[idx]
        extracted = extract_hash_answer(response)

        if extracted is None:
            no_answer += 1
            is_correct = False
        else:
            is_correct = (int(extracted) == gt)
            if not is_correct:
                wrong_answer += 1

        if is_correct:
            correct += 1
        total += 1

        # Print first few and broadly any examples that were actively guessed wrong
        is_extracted_but_wrong = (extracted is not None and not is_correct)
        if idx < 3 or is_extracted_but_wrong:
            logger.info(f"\n{'─' * 60}")
            logger.info(f"Sample {idx}")
            logger.info(f"Context: {sample['context`']}...")
            logger.info(f"Question: {sample['query']}...")
            logger.info(f"Ground truth: {gt}")
            logger.info(f"Response : ...{response[-200:]}")
            #logger.info(f"Response : {response}")
            logger.info(f"Extracted: {extracted}")
            logger.info(f"Correct: {is_correct}")

    # ── Results ───────────────────────────────────────────────────────────
    accuracy = correct / total * 100 if total > 0 else 0.0
    logger.info(f"\n{'═' * 60}")
    logger.info(f"MODEL: {args.model}")
    logger.info(f"SPLIT: validation")
    logger.info(f"TOTAL: {total}")
    logger.info(f"CORRECT: {correct}")
    logger.info(f"WRONG ANSWER: {wrong_answer}")
    logger.info(f"NO ANSWER: {no_answer}")
    logger.info(f"ACCURACY: {accuracy:.2f}%")
    logger.info(f"{'═' * 60}")


if __name__ == "__main__":
    main()