#python evals/evaluate.py Qwen/Qwen3-4B --max-tokens 3072 --batch-size 100 #--max-samples 200
#python evals/evaluate.py Qwen/Qwen3-30B-A3B --max-tokens 3072 --batch-size 100 #--max-samples 200

accelerate launch rl/grpo_run.py
python evals/evaluate.py ViratChauhan/Qwen3-4B-GRPO --max-tokens 3072 --batch-size 100 #--max-samples 200
bash scripts/run_opd.sh
python evals/evaluate.py ViratChauhan/Qwen3-4B-OPD --max-tokens 3072 --batch-size 100 #--max-samples 200