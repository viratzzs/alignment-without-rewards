from transformers import AutoTokenizer
import pandas as pd
import glob
import warnings

# Suppress HF warnings
warnings.filterwarnings('ignore')

print("Loading tokenizer Qwen/Qwen3-4B...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', trust_remote_code=True)

sys_msg = {
    'role': 'system', 
    'content': (
        "You are a logical assistant who is good at critical thinking and problem solving. "
        "Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.\n"
        "Rationalize your answer step by step, then provide only the final choice letter at the very end after "
        "'#### (Correct option number out of all 4 options)' \n"
        "For example, #### 1"
    )
}

max_len = 0
for f in glob.glob('data/*.parquet'):
    df = pd.read_parquet(f)
    print(f"Scanning {f} with {len(df)} samples...")
    for p in df['prompt']:
        # p is a list of dicts: [{'role': 'user', 'content': '...'}]
        msg = [sys_msg] + list(p)
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        length = len(tokenizer.encode(text))
        if length > max_len:
            max_len = length

print(f"HIGHEST PROMPT TOKEN COUNT: {max_len}")
