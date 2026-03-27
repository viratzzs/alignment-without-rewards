import requests
from datasets import load_dataset

SYSTEM_PROMPT = """
You are a logical assistant who is good at critical thinking and problem solving. Given a question out of a provided context, you will be given multiple options out of which you have to pick the right answer.
Rationalize your answer, then provide the final choice letter at the very end after '#### (Correct option number out of all 4 options)' (eg., #### 1)
"""
#Rationalize your answer step-by-step, then provide the final choice letter at the very end after '#### (Correct option number out of all 4 options)' (eg., #### 1)
data = load_dataset('lucasmccabe/logiqa', split="train")
x = data[69]
print(x)
response = requests.post(
    "https://shara-erring-marcy.ngrok-free.dev/v1/chat/completions",
    json={
        #"model": "Qwen/Qwen3-4B-Instruct-2507",
        "model": "Qwen/Qwen3.5-4B",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Context: {x['context']}\nQuestion: {x['query']}\nOptions:\n{x['options']}"}
            #{"role": "user", "content": "Hello, who are you?"}
            ],
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "max_tokens": 750
    }
)
#print(response.json())
print(response.json()['choices'][0]['message']['content'])