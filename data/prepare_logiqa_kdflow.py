"""
Reads existing _kdflow parquets and replaces sys message in every row's `messages` list with the current SYSTEM_PROMPT, then writes back.
"""
import copy
import pandas as pd


SYSTEM_PROMPT = (
    "You are a logical assistant who is good at critical thinking and problem solving. "
    "Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.\n"
    "Rationalize your answer step by step, then provide only the final choice number (out of 1-4) at the very end after ####. "
    "For example, #### 1"
)


def patch_system_prompt(messages: list) -> list:
    """Replace the system message content in a messages list."""
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if m.get("role") == "system":
            m["content"] = SYSTEM_PROMPT
            return msgs
    # No system message found — prepend one
    msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return msgs


def regenerate_split(path: str):
    df = pd.read_parquet(path)
    df["messages"] = df["messages"].apply(patch_system_prompt)
    df.to_parquet(path)
    print(f"Updated {len(df)} rows in {path}")


def main():
    splits = [
        "data/logiqa_train_kdflow.parquet",
        "data/logiqa_validation_kdflow.parquet",
        "data/logiqa_test_kdflow.parquet",
    ]
    for path in splits:
        regenerate_split(path)


if __name__ == "__main__":
    main()
