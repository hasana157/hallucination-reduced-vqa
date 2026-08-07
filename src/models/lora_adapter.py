"""PEFT LoRA adapter configuration and wrapper for Qwen2-VL.

Applies parameter-efficient Low-Rank Adaptation (LoRA) to linear projection
layers (q_proj, v_proj, up_proj, down_proj, gate_proj), adding <0.5% (~1M)
trainable parameters.

Example:
    >>> from src.models.lora_adapter import LoRAAdapter
    >>> model = LoRAAdapter.apply(model, r=16, alpha=32)
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class LoRAAdapter:
    """Helper for configuring and applying PEFT LoRA adapters to base model."""

    DEFAULT_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]

    @classmethod
    def get_config(
        cls,
        r: int = 16,
        alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.05,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
    ):
        """Create LoraConfig instance.

        Args:
            r: LoRA rank (default: 16).
            alpha: LoRA scaling factor (default: 32).
            target_modules: List of projection layer names.
            dropout: Dropout probability.
            bias: Bias handling ("none").
            task_type: PEFT task type ("CAUSAL_LM").

        Returns:
            peft.LoraConfig instance.
        """
        from peft import LoraConfig, TaskType

        targets = target_modules or cls.DEFAULT_TARGET_MODULES
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=targets,
            lora_dropout=dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )
        return config

    @classmethod
    def apply(
        cls,
        model: Any,
        r: int = 16,
        alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.05,
    ):
        """Apply LoRA adapters to a quantized base model.

        Args:
            model: PyTorch model instance.
            r: LoRA rank.
            alpha: LoRA alpha scaling.
            target_modules: Target projection module names.
            dropout: Dropout rate.

        Returns:
            PEFT model with LoRA adapters attached.
        """
        from peft import get_peft_model, prepare_model_for_kbit_training

        lora_config = cls.get_config(r=r, alpha=alpha, target_modules=target_modules, dropout=dropout)

        logger.info(f"Preparing k-bit model for training...")
        model = prepare_model_for_kbit_training(model)

        logger.info(f"Applying LoRA adapters (r={r}, alpha={alpha}, targets={lora_config.target_modules})...")
        peft_model = get_peft_model(model, lora_config)

        trainable_params, all_params = peft_model.get_nb_trainable_parameters()
        pct = (trainable_params / all_params) * 100
        logger.info(
            f"LoRA setup complete — Trainable params: {trainable_params:,} / {all_params:,} ({pct:.3f}%)"
        )

        return peft_model
