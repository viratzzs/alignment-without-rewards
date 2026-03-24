import os
import torch
import wandb

class SubnetworkLogger:
    def __init__(self, model):
        """
        We only track live gradients, so we don't need to save init_weights 
        or track delta norms live according to user's updated specification.
        """
        pass

    def log_step(self, model, step):
        metrics = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                metrics[f"grad_norm/{name}"] = param.grad.norm().item()
                metrics[f"grad_sparsity/{name}"] = (
                    param.grad.abs() < 1e-6
                ).float().mean().item()
        
        if metrics:
            wandb.log(metrics, step=step)
