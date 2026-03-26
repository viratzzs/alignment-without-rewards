"""
Unified evaluation script for LogiQA (validation + test combined, ~1300 samples).

Uses vLLM for fast batched generation.

Usage:
    python evaluate.py <model_name_or_path> [--max-tokens 2048] [--batch-size 50]

Example:
    python evaluate.py Qwen/Qwen3-4B
    python evaluate.py ./outputs/opd-qwen3-4b/epoch_2 --batch-size 100
"""

import argparse
import os
import re
import urllib.request
from loguru import logger
from tqdm import tqdm
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """\
Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.
Rationalize your answer and then provide only the final choice number (out of 1-4) at the very end after ####. For example, #### 1
"""

URLS = {
    "validation": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt",
    "test":       "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt",
}


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    raw = text.split("####")[-1].strip()
    match = re.match(r"(\d+)", raw)
    return match.group(1) if match else None


def format_options(options: list[str]) -> str:
    return "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))


def build_messages(sample: dict) -> list[dict]:
    user_content = (
        f"Context: {sample['context']}\n"
        f"Question: {sample['query']}\n"
        f"Options:\n{format_options(sample['options'])}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def load_logiqa(url: str) -> list[dict]:
    def process_sent(text: str) -> str:
        text = text.replace("\n", "")
        sents = text.split(".")
        out = ""
        for sent in sents:
            if not sent:
                continue
            if not out:
                out += sent
            elif sent[0].isnumeric():
                out += "." + sent
            else:
                out += ". " + sent
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
            "context":        lines[base + 2],
            "query":          lines[base + 3],
            "options":        [process_answer(lines[base + 4 + j]) for j in range(4)],
            "correct_option": correct_option,
        })
    return samples


def score(data: list[dict], outputs: list[str], split_name: str) -> dict:
    correct = wrong = no_answer = 0
    for idx, (sample, response) in enumerate(zip(data, outputs)):
        gt = sample["correct_option"] + 1  # 0-indexed → 1-indexed
        extracted = extract_hash_answer(response)

        if extracted is None:
            no_answer += 1
            is_correct = False
        else:
            is_correct = (int(extracted) == gt)
            if not is_correct:
                wrong += 1

        if is_correct:
            correct += 1

        # Log first 3 samples + any actively-wrong predictions for debugging
        if idx < 3 or (extracted is not None and not is_correct):
            logger.debug(
                f"\n{'─'*60}\nSample {idx}\n"
                f"Question: {sample['query'][:200]}\n"
                f"Ground truth: {gt} | Extracted: {extracted} | Correct: {is_correct}\n"
                f"Response: ...{response[-300:]}"
            )

    total = len(data)
    accuracy = correct / total * 100 if total > 0 else 0.0
    pct_passed = correct / total * 100 if total > 0 else 0.0
    return {
        "split":      split_name,
        "total":      total,
        "correct":    correct,
        "wrong":      wrong,
        "no_answer":  no_answer,
        "accuracy":   accuracy,
        "pct_passed": pct_passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on LogiQA (val + test combined)")
    parser.add_argument("model", type=str, help="HuggingFace model name or local path")
    parser.add_argument("--max-samples",        type=int, default=None,      help="Cap samples per split (default: all)")
    parser.add_argument("--max-tokens",         type=int, default=2048,      help="Max new tokens (default: 2048)")
    parser.add_argument("--batch-size",         type=int, default=50,        help="vLLM batch size (default: 50)")
    parser.add_argument("--dtype",              type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of GPU memory vLLM may use (default: 0.9)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs to save GPU memory")
    parser.add_argument("--splits",             type=str, default="validation,test",
                        help="Comma-separated splits to evaluate (default: validation,test)")
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    safe_name = args.model.replace("/", "_")
    log_file = os.path.join(log_dir, f"{safe_name}.log")
    logger.add(log_file, mode="w")
    logger.info(f"Logging to {log_file}")

    # ── Load & merge splits ────────────────────────────────────────────────
    splits_to_run = [s.strip() for s in args.splits.split(",")]
    all_data: list[dict] = []
    split_boundaries: dict[str, tuple[int, int]] = {}

    for split in splits_to_run:
        logger.info(f"Loading '{split}' split...")
        samples = load_logiqa(URLS[split])
        start = len(all_data)
        all_data.extend(samples)
        split_boundaries[split] = (start, len(all_data))
        logger.info(f"  {len(samples)} samples loaded")

    # Cap TOTAL samples after merging all splits
    if args.max_samples is not None and len(all_data) > args.max_samples:
        all_data = all_data[:args.max_samples]
        for split in list(split_boundaries.keys()):
            s, e = split_boundaries[split]
            if s >= args.max_samples:
                del split_boundaries[split]
            else:
                split_boundaries[split] = (s, min(e, args.max_samples))

    logger.info(f"Total samples: {len(all_data)}")

    # ── Load model ─────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
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

    # ── Build prompts ──────────────────────────────────────────────────────
    logger.info("Building prompts...")
    all_prompts = [
        tokenizer.apply_chat_template(
            build_messages(s),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for s in all_data
    ]

    # ── Generate ───────────────────────────────────────────────────────────
    all_outputs: list[str] = []
    num_batches = (len(all_prompts) + args.batch_size - 1) // args.batch_size
    logger.info(f"Running {num_batches} batches...")

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        start = batch_idx * args.batch_size
        end   = min(start + args.batch_size, len(all_prompts))
        outputs = llm.generate(all_prompts[start:end], sampling_params)
        for o in outputs:
            all_outputs.append(o.outputs[0].text)

    # ── Score per split ────────────────────────────────────────────────────
    results = []
    for split, (s, e) in split_boundaries.items():
        r = score(all_data[s:e], all_outputs[s:e], split)
        results.append(r)

    # Combined totals
    total_correct = sum(r["correct"]   for r in results)
    total_samples = sum(r["total"]     for r in results)
    total_wrong   = sum(r["wrong"]     for r in results)
    total_na      = sum(r["no_answer"] for r in results)
    overall_acc   = total_correct / total_samples * 100 if total_samples else 0.0

    # ── Print results ──────────────────────────────────────────────────────
    sep = "═" * 60
    logger.info(f"\n{sep}")
    logger.info(f"MODEL:  {args.model}")

    if len(results) == 1:
        r = results[0]
        logger.info(f"  SPLIT         : {r['split'].upper()}")
        logger.info(f"  Total samples : {r['total']}")
        logger.info(f"  Correct       : {r['correct']}")
        logger.info(f"  Wrong answer  : {r['wrong']}")
        logger.info(f"  No answer     : {r['no_answer']}")
    #    logger.info(f"  Accuracy      : {r['accuracy']:.2f}%")
    #    logger.info(f"  % Passed      : {r['pct_passed']:.1f}%   ({r['correct']}/{r['total']})")
        logger.info(f"  Accuracy      : {r['accuracy']:.1f}%   ({r['correct']}/{r['total']})")
    else:
        logger.info(f"  Total samples : {total_samples}")
        logger.info(f"  Correct       : {total_correct}")
        logger.info(f"  Wrong answer  : {total_wrong}")
        logger.info(f"  No answer     : {total_na}")
        logger.info(f"  Accuracy      : {overall_acc:.2f}%")
        logger.info(f"  % Passed      : {overall_acc:.1f}%   ({total_correct}/{total_samples})")
    logger.info(f"{sep}")


if __name__ == "__main__":
    main()
