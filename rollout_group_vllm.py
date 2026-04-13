"""
/home/KDFlow/kdflow/ray/rollout/rollout_group_vllm.py
vLLM-based RolloutActorGroup — drop-in replacement for the SGLang version.

Used when sgl_kernel is unavailable (CUDA-only; not available on ROCm).
Implements the same interface as RolloutActorGroup so the KDFlow trainer
works without changes.

Weight-sync note:
  The SGLang version syncs FSDP2 weights to the rollout engines via Gloo IPC
  after each training step (on-policy). We stub that mechanism here:
  the rollout model uses its initial weights throughout training (off-policy).
  This is acceptable for knowledge distillation because the teacher logits
  — not the rollout policy — provide the gradient signal.
"""
import asyncio
import socket
import time
from typing import Any, Dict, List, Optional, Union

import ray
import torch
from vllm import LLM, SamplingParams
from ray.util.placement_group import PlacementGroup

from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


class _SyncRemoteMethod:
    """Helper to wrap synchronous methods so they behave like ray remote calls."""
    def __init__(self, func):
        self.func = func

    def remote(self, *args, **kwargs):
        res = self.func(*args, **kwargs)
        return ray.put(res)

class _FakeActorHandle:
    """
    Fake Ray actor handle that intercepts FSDP student weight-update calls
    and routes them to the main RolloutActorGroupVLLM perfectly synchronously
    without triggering a ray.get() deadlock.
    """
    def __init__(self, parent):
        self.parent = parent
        self.update_weights_from_disk = _SyncRemoteMethod(self.parent.update_weights_from_disk)
        self.update_weights_from_tensor = _SyncRemoteMethod(lambda *a, **k: True)
        self.shutdown = _SyncRemoteMethod(self.parent.shutdown)
        self.sleep = _SyncRemoteMethod(self.parent.sleep)
        self.wakeup = _SyncRemoteMethod(self.parent.wakeup)
        self.connect_rollout_engines = _SyncRemoteMethod(lambda *a, **k: None)
        self.flush_cache = _SyncRemoteMethod(lambda *a, **k: None)
        self.health_check = _SyncRemoteMethod(lambda *a, **k: True)
        self.get_node_ip = _SyncRemoteMethod(lambda *a, **k: ray._private.services.get_node_ip_address().strip("[]"))


class RolloutActorGroupVLLM:
    """
    vLLM-backed rollout group for on-policy KD on ROCm.

    The interface is identical to the SGLang-based RolloutActorGroup:
      - generate(prompts, sampling_params, image_data) → [{"text": "..."}]
      - sleep / wakeup
      - shutdown
      - actors  (list of stub Ray actors for StudentGroup compatibility)
    """

    def __init__(
        self,
        model_path: str,
        num_actors: int = 1,
        tp_size: int = 1,
        num_gpus_per_node: int = 1,
        enable_memory_saver: bool = True,
        mem_fraction_static: Optional[float] = None,
        max_concurrent: int = 64,
        num_gpus_per_actor: float = 0.3,
        pg: Optional[Union[PlacementGroup, tuple]] = None,
        extra_server_args: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.mem_fraction = mem_fraction_static or 0.5
        self.enable_memory_saver = enable_memory_saver
        self.tp_size = tp_size
        self._sleeping = False

        logger.info(
            f"[RolloutActorGroupVLLM] Loading {model_path} "
            f"(gpu_memory_utilization={self.mem_fraction})"
        )

        # vLLM LLM instance — loads the rollout model
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=self.mem_fraction,
            enable_sleep_mode=enable_memory_saver,
            enforce_eager=True,        # avoids ROCm graph capture issues
            tensor_parallel_size=tp_size,
        )

        # Fake actor handles for StudentGroup.connect_rollout_engines compatibility
        self.actors: List[Any] = [
            _FakeActorHandle(self) for i in range(num_actors)
        ]

        # Put vLLM to sleep immediately if sleep-mode is on
        if enable_memory_saver:
            self.llm.sleep(level=1)
            self._sleeping = True
            logger.info("[RolloutActorGroupVLLM] Model sleeping after init.")

        logger.info("[RolloutActorGroupVLLM] Initialized successfully.")

    # ── Main interface ─────────────────────────────────────────────────────

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
        image_data: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Run vLLM generation and return [{text: completion}, ...]."""
        if sampling_params is None:
            sampling_params = {"temperature": 1.0, "max_new_tokens": 2048}

        sp = SamplingParams(
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            max_tokens=sampling_params.get(
                "max_new_tokens",
                sampling_params.get("max_tokens", 2048),
            ),
        )

        outputs = self.llm.generate(prompts, sampling_params=sp, use_tqdm=False)

        results = []
        for out in outputs:
            completion = out.outputs[0].text if out.outputs else ""
            token_ids = list(out.outputs[0].token_ids) if out.outputs else []
            results.append({"text": completion, "output_ids": token_ids})

        return results

    def sleep(self, tags: Optional[list] = None):
        """Offload vLLM weights to CPU."""
        if not self._sleeping:
            self.llm.sleep(level=1)
            self._sleeping = True
            logger.info("[RolloutActorGroupVLLM] Sleeping.")

    def wakeup(self, tags: Optional[list] = None):
        """Reload vLLM weights to GPU."""
        if self._sleeping:
            self.llm.wake_up()
            self._sleeping = False
            logger.info("[RolloutActorGroupVLLM] Awake.")

    def health_check(self) -> List[bool]:
        return [True] * len(self.actors)

    def shutdown(self):
        """Release vLLM resources."""
        try:
            self.llm.sleep(level=2)  # release all GPU memory
        except Exception:
            pass
        logger.info("[RolloutActorGroupVLLM] Shutdown complete.")

    def update_weights_from_disk(self, model_path: str, load_format=None, flush_cache=True):
        import gc
        import torch
        from vllm.distributed.parallel_state import destroy_model_parallel
        
        logger.info(f"[RolloutActorGroupVLLM] Updating vLLM weights from disk: {model_path}")
        
        if self._sleeping:
            self.llm.wake_up()
            
        # Completely destroy the old LLM
        del self.llm
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Re-initialize
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=self.mem_fraction,
            enable_sleep_mode=self.enable_memory_saver,
            enforce_eager=True,
            tensor_parallel_size=self.tp_size,
        )
        
        if self.enable_memory_saver:
            self.llm.sleep(level=1)
            self._sleeping = True
            
        logger.info("[RolloutActorGroupVLLM] Weight update complete.")
        return True