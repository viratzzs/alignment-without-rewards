# Hypotheses and Expected Conclusions: The "Intrinsic RL Subnetwork"

Based on the findings from *"Reinforcement Learning Finetunes Small Subnetworks in Large Language Models"* (arXiv:2505.11711), we are conducting an offline checkpoint analysis comparing your trained models (`Qwen3-4B-OPD`, `Qwen3-4B-OPSD`, and `Qwen3-4B-GRPO`) against the base model (`Qwen3-4B`). 

Here are the core hypotheses we are aiming to prove, and the exact results we expect to see from the [analyze_checkpoints_sparsity.py](file:///home/alignment-without-rewards/analyze_checkpoints_sparsity.py) script.

---

## Hypothesis 1: Intrinsic Parameter Update Sparsity
**The Claim (RQ1):** Alignment algorithms (like GRPO, or on-policy KD like OPD/OPSD) implicitly behave like mask-learning algorithms. They freeze the vast majority of the network's knowledge and only update a tiny, sparse subnetwork.
**How We Test:** Calculating the fraction of parameters where the absolute difference $|\Delta W| < 1\times10^{-5}$ between the Finetuned and Base models.
**Expected Result:** 
The script should output an `Overall Update Sparsity` between **70% and 95%** across all three of your tuned models. This will definitively prove that your models only needed to update 5%–30% of their parameters to achieve alignment, preserving the pretrained capabilities perfectly.

---

## Hypothesis 2: Alignment is a "Full-Rank" Problem
**The Claim (RQ2):** Even though the updates are highly sparse, the parameters that *do* change are scattered in a highly complex structure. This directly contradicts the assumption made by Parameter-Efficient approaches like LoRA, which assumes updates can be compressed into low-rank bottleneck matrices.
**How We Test:** Performing NumPy SVD Rank calculation (`np.linalg.matrix_rank`) on the delta matrices of the attention and MLP linear layers.
**Expected Result:** 
The script should output `Rank: 4096 / 4096` (or whatever the maximum dimension of the subset is). If the matrix rank is consistently maximized despite 90% parameters being untouched, we can officially conclude that **Full Finetuning is mathematically superior/required** over LoRA for your specific advanced alignment recipes.

---

## Hypothesis 3: Subnetwork Consistency Across Algorithms
**The Claim (RQ4):** The underlying structure that needs to be updated for alignment is intrinsic to the *base model itself*, not the algorithm used. Whether you use PPO, GRPO, supervised distillation (OPD), or self-learning (OPSD), the optimizer will naturally gravitate towards updating the exact same subset of weights.
**How We Test:** Taking the boolean sparsity masks of the parameters updated in `Qwen3-4B-OPD` and intersecting them with the updated parameters in `Qwen3-4B-OPSD` and `Qwen3-4B-GRPO`.
**Expected Result:** 
We expect the script to output an exceedingly high `Overlap Ratio` (e.g. > 50-70%) between the models. If GRPO (using PPO-style clipping rewards) and OPD (using logit-matching KL divergence) both decide to independently update the exact same `layers.31.mlp.down_proj.weight`, it proves that the "Alignment Subnetwork" is universally consistent. 

---

### Final Implementation Checklist for the Report
Once [analyze_checkpoints_sparsity.py](file:///home/alignment-without-rewards/analyze_checkpoints_sparsity.py) finishes executing on your local machine:
- [ ] Save the **Sparsity %** to validate H1.
- [ ] Note the **Matrix Rankings** to validate H2.
- [ ] Record the **Overlap Ratios** between OPD, OPSD, and GRPO to validate H3.
