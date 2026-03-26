"""Push a checkpoint directory to HuggingFace Hub."""
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir",   required=True,  help="Path to the saved checkpoint directory")
    parser.add_argument("--repo-name",  required=True,  help="HF repo, e.g. ViratChauhan/Qwen3-4B-OPD")
    parser.add_argument("--private",    action="store_true", default=True, help="Make repo private (default: True)")
    parser.add_argument("--commit-msg", default="Training run complete", help="Commit message")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"Pushing {args.ckpt_dir} → {args.repo_name} ...")
    try:
        api.create_repo(repo_id=args.repo_name, exist_ok=True, private=args.private)
        api.upload_folder(
            folder_path=args.ckpt_dir,
            repo_id=args.repo_name,
            commit_message=args.commit_msg,
        )
        print(f"✓ Uploaded to https://huggingface.co/{args.repo_name}")
    except Exception as e:
        print(f"✗ Failed to push to Hub: {e}")
        raise


if __name__ == "__main__":
    main()
