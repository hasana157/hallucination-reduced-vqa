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

    DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    DEFAULT_MODULES_TO_SAVE = ["visual.merger"]

    @classmethod
    def get_config(
        cls,
        r: int = 32,
        alpha: int = 64,
        target_modules: Optional[List[str]] = None,
        modules_to_save: Optional[List[str]] = None,
        dropout: float = 0.05,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
    ):
        """Create LoraConfig instance.

        Args:
            r: LoRA rank (default: 32).
            alpha: LoRA scaling factor (default: 64).
            target_modules: List of projection layer names.
            modules_to_save: Additional non-LoRA modules to train (e.g. ["visual.merger"]).
            dropout: Dropout probability.
            bias: Bias handling ("none").
            task_type: PEFT task type ("CAUSAL_LM").

        Returns:
            peft.LoraConfig instance.
        """
        from peft import LoraConfig, TaskType

        targets = target_modules or cls.DEFAULT_TARGET_MODULES
        # Do NOT pass modules_to_save here — visual.merger is unfrozen manually
        # in apply() after prepare_model_for_kbit_training to avoid the
        # 'only Tensors of floating point dtype can require gradients' crash.

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=targets,
            lora_dropout=dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )
        return config

    @staticmethod
    def _dequantize_4bit_linears(module: Any) -> int:
        """Recursively replace all bnb.Linear4bit layers with plain float32 nn.Linear.

        llm_int8_skip_modules in BitsAndBytesConfig does not reliably prevent
        4-bit quantization of MLP weights inside visual.merger. This method
        walks the module tree and swaps every quantized Linear with a trainable
        fp32 copy before LoRA is applied.

        Returns:
            Number of layers dequantized.
        """
        import torch
        import torch.nn as nn

        try:
            import bitsandbytes.nn as bnb
            quant_types = (bnb.Linear4bit, bnb.Linear8bitLt)
        except ImportError:
            return 0

        count = 0
        for child_name, child in list(module.named_children()):
            if isinstance(child, quant_types):
                w = child.weight
                # Dequantize weight tensor to fp32
                if hasattr(w, 'dequantize'):
                    weight_data = w.dequantize().to(torch.float32)
                else:
                    weight_data = w.data.float()

                has_bias = child.bias is not None
                new_lin = nn.Linear(
                    child.in_features, child.out_features,
                    bias=has_bias,
                    device=weight_data.device,
                )
                new_lin.weight = nn.Parameter(weight_data)
                if has_bias:
                    new_lin.bias = nn.Parameter(child.bias.data.float())

                setattr(module, child_name, new_lin)
                count += 1
                logger.info(f"  Dequantized {child_name}: {child.__class__.__name__} → nn.Linear (float32)")
            else:
                # Recurse into sub-modules (e.g. Sequential)
                count += LoRAAdapter._dequantize_4bit_linears(child)
        return count


    @classmethod
    def apply(
        cls,
        model: Any,
        r: int = 32,
        alpha: int = 64,
        target_modules: Optional[List[str]] = None,
        modules_to_save: Optional[List[str]] = None,
        dropout: float = 0.05,
    ):
        """Apply LoRA adapters to a quantized base model.

        Args:
            model: PyTorch model instance.
            r: LoRA rank.
            alpha: LoRA alpha scaling.
            target_modules: Target projection module names.
            modules_to_save: Non-LoRA modules to manually unfreeze (e.g. ["visual.merger"]).
                These are cast to float32 and trained alongside LoRA adapters.
                NOT passed to LoraConfig.modules_to_save to avoid the uint8
                gradient crash when the module is loaded with 4-bit quantization.
            dropout: Dropout rate.

        Returns:
            PEFT model with LoRA adapters attached.
        """
        from peft import get_peft_model, prepare_model_for_kbit_training

        to_unfreeze = modules_to_save if modules_to_save is not None else cls.DEFAULT_MODULES_TO_SAVE

        # --- Step 1: Dequantize visual connector layers to float32 ---
        # llm_int8_skip_modules in BitsAndBytesConfig does not reliably prevent
        # 4-bit quantization of MLP weights (they show up as torch.uint8).
        # We must replace them with plain nn.Linear before LoRA setup.
        for module_path in to_unfreeze:
            target = model
            for part in module_path.split('.'):
                target = getattr(target, part, None)
                if target is None:
                    logger.warning(f"Module '{module_path}' not found in model, skipping.")
                    break
            if target is not None:
                n = cls._dequantize_4bit_linears(target)
                logger.info(f"Dequantized {n} layer(s) in '{module_path}' to float32")

        lora_config = cls.get_config(
            r=r,
            alpha=alpha,
            target_modules=target_modules,
            dropout=dropout,
        )

        # --- Step 2: Prepare model for k-bit training ---
        logger.info("Preparing k-bit model for training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

        # --- Step 3: Apply LoRA adapters ---
        logger.info(
            f"Applying LoRA adapters (r={r}, alpha={alpha}, "
            f"targets={lora_config.target_modules})..."
        )
        peft_model = get_peft_model(model, lora_config)

        # --- Step 4: Re-enable gradients on dequantized connector modules ---
        # After prepare_model_for_kbit_training, all non-LoRA params are frozen.
        # Re-enable requires_grad for the now-float32 merger params.
        unfrozen_count = 0
        for name, param in peft_model.named_parameters():
            for module_name in to_unfreeze:
                if module_name in name:
                    if param.data.is_floating_point():
                        param.requires_grad_(True)
                        unfrozen_count += 1
                    else:
                        logger.warning(
                            f"Cannot enable grad on '{name}': dtype={param.data.dtype}"
                        )
        logger.info(f"Re-enabled gradients on {unfrozen_count} param(s) in: {to_unfreeze}")

        trainable_params, all_params = peft_model.get_nb_trainable_parameters()
        pct = (trainable_params / all_params) * 100
        logger.info(
            f"LoRA setup complete — Trainable: {trainable_params:,} / {all_params:,} ({pct:.3f}%)"
        )

        return peft_model
