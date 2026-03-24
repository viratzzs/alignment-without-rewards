import os
#os.environ['HF_HOME'] = "/projects/alignment-without-rewards/hf_cache/"

import pandas as pd
from datasets import load_dataset

dataset = load_dataset("lucasmccabe/logiqa", trust_remote_code=True)["train"]

print(len(dataset))
print(dataset)

print("-------------------------------")
for count, item in enumerate(dataset):
    #print(item)
    if count >= 0:
        break
    print("CONTEXT: ", item["context"])
    print("QUERY: ", item["query"])
    print("OPTIONS:\n" + "\n".join(f'{i+1}. {o}' for i, o in enumerate(item['options'])))
    #for i, o in enumerate(item["options"]):
    #    print(i+1, o)
    print("CORRECT OPTION: ", item["options"][item["correct_option"]])
    #print("choice number: ", item["correct_option"] + 1)
    print("-------------------------------")

print(dataset[0])