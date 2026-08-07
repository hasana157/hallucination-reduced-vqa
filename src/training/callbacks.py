"""HuggingFace Trainer Callbacks for Checkpointing and GPU Memory Logging.

Example:
    >>> from src.training.callbacks import CheckpointCallback, MemoryCallback
"""

import logging
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class CheckpointCallback(TrainerCallback):
    """Custom callback for saving model checkpoints and logging state."""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        logger.info(f"Checkpoint saved at step {state.global_step}: {checkpoint_dir}")


class MemoryCallback(TrainerCallback):
    """Log CUDA VRAM allocation periodically during training."""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % 100 == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / (1024**3)
                    peak = torch.cuda.max_memory_allocated() / (1024**3)
                    logger.info(
                        f"Step {state.global_step} — CUDA VRAM Allocated: {alloc:.2f} GB | Peak: {peak:.2f} GB"
                    )
            except Exception:
                pass
