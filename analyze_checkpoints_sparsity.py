import torch
import argparse
import os
import gc
from transformers import AutoModelForCausalLM
from loguru import logger
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def get_state_dict(model_name):
    logger.info(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda", 
        trust_remote_code=True
    )
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return sd

def analyze_models(base_name, tuned_names, threshold=1e-5):
    base_sd = get_state_dict(base_name)
    tuned_sds = {name: get_state_dict(name) for name in tuned_names}
    
    results = {}
    
    # Analyze sparsity and rank per tuned model
    for name, sd in tuned_sds.items():
        logger.info(f"{'='*50}")
        logger.info(f"Analyzing: {name} vs Base")
        total_params = 0
        total_sparse = 0
        layer_sparsities = []
        
        for k in base_sd.keys():
            if k not in sd: continue
            
            diff = sd[k].float() - base_sd[k].float()
            
            # 1. Parameter Update Sparsity (RQ1)
            sparse_mask = diff.abs() < threshold
            sparse_count = sparse_mask.sum().item()
            num_params = diff.numel()
            
            total_params += num_params
            total_sparse += sparse_count
            
            # Print layer-wise samples (just looking at standard linear layers for brevity)
            if "mlp.down_proj.weight" in k or "self_attn.v_proj.weight" in k:
                layer_sparsity = sparse_count / num_params
                
                # 2. Matrix Rank (RQ2)
                # The paper states updates are nearly full-rank despite being sparse
                # We calculate the rank of the diff matrix
                rank = -1
                if diff.ndim == 2:
                    try:
                        # PyTorch GPU backend uses rocSOLVER (extremely fast)
                        rank = torch.linalg.matrix_rank(diff).item()
                    except RuntimeError:
                        # Fallback for CPU
                        rank = np.linalg.matrix_rank(diff.cpu().numpy()).item()
                    max_rank = min(diff.shape)
                    
                logger.info(f"  {k}: Sparsity: {layer_sparsity:.2%} | Rank: {rank}/{max_rank if diff.ndim==2 else 'N/A'}")
                
        overall_sparsity = total_sparse / total_params if total_params else 0
        logger.info(f"--> Overall Update Sparsity for {name}: {overall_sparsity:.2%} (Paper claims RL induces 70-95% sparsity)")
        
        # Save exact boolean mask of updated parameters for Overlap calculation
        results[name] = {
            k: (sd[k].float() - base_sd[k].float()).abs() >= threshold 
            for k in base_sd.keys() if k in sd
        }
        
    # 3. Subnetwork Overlap (RQ4)
    # The paper tests if different RL algos update the SAME subnetwork
    if len(tuned_sds) >= 2:
        logger.info(f"{'='*50}")
        logger.info("Subnetwork Overlap Analysis (RQ4)")
        names = list(tuned_sds.keys())
        
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name1, name2 = names[i], names[j]
                mask1 = results[name1]
                mask2 = results[name2]
                
                overlap_total = 0
                union_total = 0
                
                for k in mask1.keys():
                    if k in mask2:
                        overlap_total += (mask1[k] & mask2[k]).sum().item()
                        union_total += (mask1[k] | mask2[k]).sum().item()
                
                overlap_ratio = overlap_total / union_total if union_total else 0
                logger.info(f"Overlap between {name1} and {name2}: {overlap_ratio:.2%}")
                logger.info(f"  (If this is high, it confirms the paper's claim that intrinsic subnetworks are consistent across algorithms)")
                logger.info(f"If not, then the overlap is more than just on-policy data distribution as we might think. It can be the difference in updates performed by GRPO and OPD with a teacher model (OPSD todo.)")

if __name__ == "__main__":
    logger.add("sparsity_analysis.log", mode="w")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--tuned", type=str, nargs="+", default=[
        "ViratChauhan/Qwen3-4B-OPD", 
        #"ViratChauhan/Qwen3-4B-OPSD",
        "ViratChauhan/Qwen3-4B-GRPO"
    ])
    parser.add_argument("--threshold", type=float, default=1e-5)
    args = parser.parse_args()
    
    analyze_models(args.base, args.tuned, args.threshold)
