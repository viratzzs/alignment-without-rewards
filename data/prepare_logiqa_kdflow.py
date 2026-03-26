"""
Convert the existing LogiQA parquet files into KDFlow-compatible format.

KDFlow's PromptDataset expects a `messages` column with OpenAI-style chat
messages (list of dicts with role/content) and optionally a `label` column.
"""
import pandas as pd


SYSTEM_PROMPT = (
    "You are a logical assistant who is good at critical thinking and problem solving. "
    "Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.\n"
    "Rationalize your answer step by step, then provide only the final choice letter at the very end after "
    "'#### (Correct option number out of all 4 options)' \n"
    "For example, #### 1"
)


def convert_split(input_path: str, output_path: str):
    df = pd.read_parquet(input_path)
    
    records = []
    for _, row in df.iterrows():
        # row["prompt"] is a list of dicts [{"role": "system", ...}, {"role": "user", ...}]
        messages = row["prompt"]
        label = row.get("answer", "")
        records.append({
            "messages": messages,
            "label": str(label),
        })
    
    out_df = pd.DataFrame(records)
    out_df.to_parquet(output_path)
    print(f"Saved {len(out_df)} samples to {output_path}")


def main():
    splits = {
        "data/logiqa_train.parquet": "data/logiqa_train_kdflow.parquet",
        "data/logiqa_validation.parquet": "data/logiqa_validation_kdflow.parquet",
        "data/logiqa_test.parquet": "data/logiqa_test_kdflow.parquet",
    }
    for src, dst in splits.items():
        convert_split(src, dst)


if __name__ == "__main__":
    main()
