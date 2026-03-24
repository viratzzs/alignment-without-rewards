import os
import sys
import subprocess
from pyngrok import ngrok
from dotenv import load_dotenv

load_dotenv()
ngrok.set_auth_token(os.environ["NGROK_TOKEN"])

public_url = ngrok.connect(8000).public_url
print(f"🚀 TEACHER SERVER PUBLIC URL: {public_url}")

# enable logprobs and set the max-model-len for training needs
model_path = "Qwen/Qwen3.5-4B"

vllm_command = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", model_path,
    "--port", "8000",
    "--max-model-len", "2048",
    "--trust-remote-code",
    "--dtype", "bfloat16",
    "--gpu-memory-utilization", "0.8",
    "--enforce-eager"
]

print(f"Waiting for {model_path} to initialize on port 8000...")
try:
    subprocess.run(vllm_command)
except KeyboardInterrupt:
    print("\nShutting down teacher tunnel...")
    ngrok.kill()