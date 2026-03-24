import os
import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Path to the saved checkpoint directory")
    parser.add_argument("--repo-name", type=str, required=True, help="HF Repo name like ViratChauhan/Qwen3-4B-opd")
    args = parser.parse_args()

    api = HfApi()
    print(f"Pushing {args.ckpt_dir} to HuggingFace Hub at {args.repo_name}...")
    try:
        api.create_repo(repo_id=args.repo_name, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=args.ckpt_dir,
            repo_id=args.repo_name,
            commit_message="Verl distillation run complete"
        )
        print("Successfully uploaded to Hub!")
    except Exception as e:
        print(f"Failed to push to Hub: {e}")

if __name__ == "__main__":
    main()
