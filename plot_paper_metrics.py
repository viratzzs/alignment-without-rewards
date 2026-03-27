import wandb
import matplotlib.pyplot as plt
import re
import os
from dotenv import load_dotenv

load_dotenv("/home/alignment-without-rewards/.env")
api = wandb.Api(timeout=30)

# We will try the known projects
projects = {
    "OPD": "alignment-opd",
    "OPSD": "alignment-opsd",
    "GRPO": "huggingface" # GRPO uses trls default which is sometimes huggingface or the script's project config
}

# Also try looking for the runs across the whole user workspace if the project names differ
username = "virat-codes"

data = {}
for algo, proj in projects.items():
    print(f"Fetching {algo}...")
    try:
        # Get the runs in the project
        runs = api.runs(f"{username}/{proj}")
        
        target_run = None
        for r in runs:
            if "grad_sparsity" in str(r.summary.keys()):
                target_run = r
                break
                
        if not target_run and algo == "GRPO":
             # GRPO might be in a different proj, let's search broadly
             runs = api.runs(f"{username}/alignment-without-rewards")
             for r in runs:
                 if "grad_sparsity" in str(r.summary.keys()):
                     target_run = r
                     break

        if target_run:
            print(f"  Found run: {target_run.name}")
            summary = target_run.summary
            
            sparsity_data = {}
            for k, v in summary.items():
                if "grad_sparsity" in k and isinstance(v, (int, float)):
                    # k looks like "grad_sparsity/model.layers.10.mlp.down_proj.weight"
                    match = re.search(r'layers\.(\d+)', k)
                    if match:
                        layer_idx = int(match.group(1))
                        if layer_idx not in sparsity_data:
                            sparsity_data[layer_idx] = []
                        sparsity_data[layer_idx].append(v)
            
            if sparsity_data:
                # average across all matrices in the layer (mlp, attn, etc.)
                avg_per_layer = {k: sum(v)/len(v) for k, v in sparsity_data.items()}
                data[algo] = avg_per_layer
                overall_avg = sum(avg_per_layer.values()) / len(avg_per_layer)
                print(f"  --> Overall Sparsity for {algo}: {overall_avg:.2%}")
                
                # Check densest/sparsest layers for overlap reporting
                sorted_l = sorted(avg_per_layer.items(), key=lambda x: x[1])
                print(f"  --> Densest layers (most updated): {[l[0] for l in sorted_l[:3]]}")
                print(f"  --> Sparsest layers (least updated): {[l[0] for l in sorted_l[-3:]]}")
        else:
            print(f"  No grad_sparsity runs found in {proj}.")
            
    except Exception as e:
        print(f"  Error fetching {algo}: {e}")

if data:
    plt.figure(figsize=(12, 6))
    
    colors = {"OPD": "tab:blue", "OPSD": "tab:orange", "GRPO": "tab:green"}
    
    for algo, layer_dict in data.items():
        layers = sorted(layer_dict.keys())
        sparsities = [layer_dict[l] * 100 for l in layers]
        
        plt.plot(layers, sparsities, marker='o', linewidth=2, markersize=6, 
                 label=f"{algo} (Avg: {sum(sparsities)/len(sparsities):.1f}%)",
                 color=colors.get(algo, "tab:red"))
        
    plt.title("Intrinsic RL Subnetworks: Consistency Across Algorithms (RQ4)", fontsize=14, fontweight='bold')
    plt.suptitle("Tracing the 'Parameter Update Sparsity' from arXiv:2505.11711", fontsize=10, color='gray')
    plt.xlabel("Transformer Layer Index", fontsize=12)
    plt.ylabel("Gradient Sparsity % (Parameters NOT Updated)", fontsize=12)
    plt.ylim(0, 100)
    plt.xlim(min(layers)-1, max(layers)+1)
    
    # Highlight the 70-95% sparsity band identified by the paper
    plt.axhspan(70, 95, color='gray', alpha=0.1, label='Paper\'s Observed Sparsity Range (70-95%)')
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    
    out_path = "/root/.gemini/antigravity/brain/05c9bfb3-3d19-44d8-bfb3-d6bfe39868a8/sparsity_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"SUCCESS: Plot generated at {out_path}")
else:
    print("FAILED: No data was plotted.")
