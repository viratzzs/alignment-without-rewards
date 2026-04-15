"""
On-Policy Self-Distillation (OPSD) algorithm for KDFlow.

Implements the OPSD method from:
  "Self-Distilled Reasoner: On-Policy Self-Distillation"
  (https://siyan-zhao.github.io/blog/2026/opsd/)

Key differences from vanilla KD:
  - Teacher == Student (same model weights, but teacher is frozen at init)
  - Teacher is conditioned on privileged info (correct answer appended to prompt)
  - Per-token pointwise KL clipping to prevent style tokens from dominating
  - Full-vocabulary JSD divergence (not sparse top-k)
"""
import torch
import torch.nn.functional as F
import ray
import numpy as np

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm


@register_algorithm("opsd")
class OPSD:
    """
    On-Policy Self-Distillation.
    
    The teacher and student share the same architecture (self-distillation).
    The teacher is conditioned on the correct answer (privileged information)
    while the student only sees the prompt + its own rollout response.
    
    Training signal: per-token distribution matching between teacher's
    privileged distribution and student's distribution over the student's
    own generated trajectory.
    """

    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
        
        # OPSD-specific: per-token pointwise KL clipping threshold
        self.pointwise_kl_clip = getattr(self.args.kd, "pointwise_kl_clip", 10.0)
        
        # Explicit masking for thinking tokens to prevent style leakage
        self.tokenizer = student_model.tokenizer if hasattr(student_model, "tokenizer") else None
        self.thinking_token_ids = []
        if self.tokenizer:
            for token in ["<think>", "</think>"]:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and token_id != self.tokenizer.unk_token_id:
                    self.thinking_token_ids.append(token_id)
        
        if self.thinking_token_ids:
            strategy.print(f"[OPSD] Masking reasoning tokens {self.thinking_token_ids} from loss.")

    def training_step(self, micro_batch):
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, (
            "micro_batch must contain `teacher_hiddens` for OPSD. "
            "Ensure the teacher model (same arch as student) is providing hidden states."
        )

        # multimodal kwargs
        mm_kwargs = {k[3:]: v for k, v in micro_batch.items() if k.startswith("mm_")}

        # Student forward pass
        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits = output["logits"]

        # We will project teacher hiddens to logits chunk-by-chunk in _compute_clipped_loss
        # to save tens of gigabytes of VRAM.

        # Apply loss mask (only on response tokens)
        student_logits = student_logits[student_loss_mask]
        
        # Projecting teacher hiddens to logits on-the-fly saves tens of GBs of VRAM.
        # Note: teacher_hiddens is ALREADY masked by the teacher engine.
        if isinstance(teacher_hiddens, ray.ObjectRef):
            teacher_hiddens = ray.get(teacher_hiddens)
        if isinstance(teacher_hiddens, np.ndarray):
            teacher_hiddens = torch.from_numpy(teacher_hiddens)

        # ── Compute loss with optional pointwise KL clipping ──
        # DISTILLATION LOSS (full vocabulary)
        # Clear cache once before heavy distillation loss calculation to ensure maximum headroom
        torch.cuda.empty_cache()
        if self.pointwise_kl_clip > 0:
            kd_loss = self._compute_clipped_loss(
                student_logits, teacher_hiddens, avg_token_num
            )
        else:
            kd_loss = self.loss_fn(
                student_logits, teacher_hiddens, reduction="none"
            )
            kd_loss = kd_loss.sum() / avg_token_num

        return {"loss": kd_loss, "kd_loss": kd_loss}

    def _compute_clipped_loss(self, student_logits, teacher_hiddens, avg_token_num):
        """
        Compute per-token loss with JSD and pointwise clipping using CHUNKING and 
        ON-THE-FLY projection for teacher logits to save massive VRAM.
        
        teacher_hiddens: [num_tokens, hidden_dim] (~100MB for 10k seq)
        student_logits: [num_tokens, vocab_size] (already allocated by model)
        """
        temperature = self.args.kd.kd_temperature
        beta = getattr(self.args.kd, "jsd_beta", 0.5)
        chunk_size = 512 
        
        num_tokens = student_logits.shape[0]
        total_loss = 0.0
        
        log_beta = torch.tensor(beta, dtype=torch.float32, device=student_logits.device).log()
        log_1_minus_beta = torch.tensor(1.0 - beta, dtype=torch.float32, device=student_logits.device).log()
        
        for i in range(0, num_tokens, chunk_size):
            end = min(i + chunk_size, num_tokens)
            
            # Project teacher hiddens to logits ONLY for this chunk
            with torch.no_grad():
                t_h_chunk = teacher_hiddens[i:end].to(
                    device=self.teacher_lm_head.weight.device, 
                    dtype=self.teacher_lm_head.weight.dtype
                )
                t_logits_chunk = self.teacher_lm_head(t_h_chunk) / temperature
                t_probs = F.softmax(t_logits_chunk, dim=-1, dtype=torch.float32)
                t_log_probs = F.log_softmax(t_logits_chunk, dim=-1, dtype=torch.float32)
            
            s_logits_chunk = student_logits[i:end] / temperature
            s_probs = F.softmax(s_logits_chunk, dim=-1, dtype=torch.float32)
            s_log_probs = F.log_softmax(s_logits_chunk, dim=-1, dtype=torch.float32)
            
            # log(m) = log(beta*p_T + (1-beta)*p_S)
            log_m = torch.logaddexp(t_log_probs + log_beta, s_log_probs + log_1_minus_beta)
            
            # KL(p_T || m) = p_T * (log p_T - log m)
            pointwise_kl_tm = t_probs * (t_log_probs - log_m)
            if self.pointwise_kl_clip > 0:
                pointwise_kl_tm = pointwise_kl_tm.clamp(max=self.pointwise_kl_clip)
                
            # KL(p_S || m) = p_S * (log p_S - log m)
            pointwise_kl_sm = s_probs * (s_log_probs - log_m)
            if self.pointwise_kl_clip > 0:
                pointwise_kl_sm = pointwise_kl_sm.clamp(max=self.pointwise_kl_clip)
                
            # Pointwise JSD
            pointwise_jsd = beta * pointwise_kl_tm + (1.0 - beta) * pointwise_kl_sm
            
            # Thinking-token mask
            if self.thinking_token_ids:
                pointwise_jsd[:, self.thinking_token_ids] = 0.0
                
            total_loss += pointwise_jsd.sum()
            
        return total_loss / avg_token_num
