import re
from loguru import logger


def extract_hash_answer(text: str) -> str | None:
    """Extract the answer after #### from model output."""
    if "####" not in text:
        return None
    raw = text.split("####")[-1].strip()
    # grab just the first token (the number)
    match = re.match(r"(\d+)", raw)
    return match.group(1) if match else None


# ── Reward functions ──────────────────────────────────────────────────────────
def correctness_reward_func(prompts, completions, correct_option, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_hash_answer(r) for r in responses]

    rewards = []
    for ext, gt in zip(extracted, correct_option):
        if ext is not None and ext == str(gt):
            rewards.append(2.0)
        else:
            rewards.append(0.0)

    # debug logging for the first sample
    q = prompts[0][-1]["content"]
    logger.debug(
        f"\n{'─'*40}\n"
        f"Question:\n{q}\n"
        f"Answer: {correct_option[0]}\n"
        f"Response:\n{responses[0][-300:]}\n"
        f"Extracted: {extracted[0]}\n"
        f"Reward: {rewards[0]}"
    )
    return rewards


def format_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        if re.search(r"####\s*\d\s*$", r.strip()):
            # perfect: ends with #### <digit>
            rewards.append(1.0)
        elif "####" in r:
            # has the delimiter but not at the end or not followed by a single digit
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards


def int_answer_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        ans = extract_hash_answer(r)
        if ans is not None and ans in ("1", "2", "3", "4"):
            rewards.append(0.5)
        elif ans is not None and ans.isdigit():
            # it's a digit but out of range — partial credit
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def reasoning_quality_reward_func(completions, **kwargs) -> list[float]:
    """
    Encourages longer, step-by-step reasoning before the final answer.
    Rewards based on:
      - Having multiple sentences of reasoning
      - Having reasoning BEFORE the #### answer (not after)
      - Penalises very short or empty completions
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        score = 0.0
        # split at #### to get the reasoning portion
        parts = r.split("####")
        reasoning = parts[0].strip() if parts else r.strip()

        # reward based on reasoning length (in sentences)
        sentences = [s.strip() for s in re.split(r'[.!?\n]', reasoning) if len(s.strip()) > 10]
        num_sentences = len(sentences)

        if num_sentences >= 5:
            score += 0.5
        elif num_sentences >= 3:
            score += 0.3
        elif num_sentences >= 1:
            score += 0.1

        # bonus: penalise content after #### (should be just the number)
        if len(parts) > 1:
            trailing = parts[-1].strip()
            # ideal trailing is just a single digit
            if re.match(r"^\d$", trailing):
                score += 0.25
            elif len(trailing) > 10:
                # too much text after the answer delimiter
                score -= 0.1

        # penalise extremely short responses
        if len(r.strip()) < 30:
            score = 0.0

        rewards.append(max(score, 0.0))
    return rewards


def no_repetition_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        words = r.split()
        if len(words) < 10:
            rewards.append(0.0)
            continue

        # check trigram repetition rate
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        unique_trigrams = set(trigrams)

        if len(trigrams) == 0:
            rewards.append(0.0)
            continue

        uniqueness_ratio = len(unique_trigrams) / len(trigrams)

        if uniqueness_ratio > 0.7:
            rewards.append(0.5)   # healthy diversity
        elif uniqueness_ratio > 0.4:
            rewards.append(0.2)   # some repetition
        else:
            rewards.append(-0.5)  # degenerate repetition, penalise

    return rewards
