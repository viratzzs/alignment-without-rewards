import os
import re
import urllib.request
import pandas as pd

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

    print(f"Downloading from {url}...")
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

def format_prompt(sample: dict) -> dict:
    sys_prompt = (
        "You are a logical assistant who is good at critical thinking and problem solving. "
        "Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.\n"
        "Rationalize your answer step by step, then provide only the final choice letter at the very end after "
        "'#### (Correct option number out of all 4 options)' \n"
        "For example, #### 1"
    )
    
    prompt = f"Context: {sample['context']}\n\nQuestion: {sample['query']}\n\nOptions:\n"
    for i, opt in enumerate(sample['options']):
        prompt += f"{chr(65+i)}. {opt}\n"
    
    return {
        "prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        "answer": chr(65 + sample['correct_option'])
    }

def main():
    os.makedirs("data", exist_ok=True)
    
    splits = {
        "train": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt",
        "validation": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt",
        "test": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt"
    }
    
    for split_name, url in splits.items():
        raw_samples = load_logiqa(url)
        formatted = [format_prompt(s) for s in raw_samples]
        
        df = pd.DataFrame(formatted)
        out_path = f"data/logiqa_{split_name}.parquet"
        df.to_parquet(out_path)
        print(f"Saved {len(df)} samples to {out_path}")

if __name__ == "__main__":
    main()
