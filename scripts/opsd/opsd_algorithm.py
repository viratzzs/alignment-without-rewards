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
        # This caps the maximum per-vocabulary-entry divergence contribution
        # at each position to prevent stylistic tokens from dominating.
        self.pointwise_kl_clip = getattr(self.args.kd, "pointwise_kl_clip", 10.0)

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

        # Teacher logits from hidden states + lm_head
        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits = self.teacher_lm_head(teacher_hiddens)

        # Apply loss mask (only on response tokens)
        student_logits = student_logits[student_loss_mask]
        
        # Handle potential vocab size mismatch (shouldn't happen for self-distillation
        # but kept for robustness)
        min_vocab = min(teacher_logits.shape[-1], student_logits.shape[-1])
        teacher_logits = teacher_logits[:, :min_vocab]
        student_logits = student_logits[:, :min_vocab]

        assert teacher_logits.shape == student_logits.shape, (
            f"Shape mismatch: teacher {teacher_logits.shape} vs student {student_logits.shape}"
        )

        # ── Compute loss with optional pointwise KL clipping ──
        if self.pointwise_kl_clip > 0:
            kd_loss = self._compute_clipped_loss(
                student_logits, teacher_logits, avg_token_num
            )
        else:
            kd_loss = self.loss_fn(
                student_logits, teacher_logits, reduction="none"
            )
            kd_loss = kd_loss.sum() / avg_token_num

        return {"loss": kd_loss, "kd_loss": kd_loss}

    def _compute_clipped_loss(self, student_logits, teacher_logits, avg_token_num):
        """
        Compute per-token loss with pointwise KL clipping.
        
        Per-token pointwise KL clipping caps the per-vocabulary-entry 
        divergence contribution at each position. This prevents stylistic
        tokens (e.g., "wait", "think", "therefore") from dominating the
        training signal over mathematically meaningful tokens.
        
        The clipping is applied to the per-vocab-entry KL before summing
        across the vocabulary dimension.
        """
        temperature = self.args.kd.kd_temperature
        
        student_log_probs = F.log_softmax(
            student_logits / temperature, dim=-1, dtype=torch.float32
        )
        teacher_probs = F.softmax(
            teacher_logits / temperature, dim=-1, dtype=torch.float32
        )
        teacher_log_probs = F.log_softmax(
            teacher_logits / temperature, dim=-1, dtype=torch.float32
        )
        
        # Per-vocab-entry KL(P_T || P_S): teacher_probs * (log(teacher) - log(student))
        # Shape: [num_valid_tokens, vocab_size]
        pointwise_kl = teacher_probs * (teacher_log_probs - student_log_probs)
        
        # Clip per-vocab-entry contribution
        pointwise_kl = pointwise_kl.clamp(max=self.pointwise_kl_clip)
        
        # Sum across vocab dimension → per-token loss, then average across tokens
        per_token_loss = pointwise_kl.sum(dim=-1)
        loss = per_token_loss.sum() / avg_token_num
        
        return loss
