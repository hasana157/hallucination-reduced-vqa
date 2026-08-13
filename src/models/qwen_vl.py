"""Qwen2-VL-2B-Instruct 4-bit quantization and model loading.

Loads Qwen2-VL vision-language model using BitsAndBytes 4-bit NormalFloat4 (NF4)
quantization to fit within Google Colab T4 memory limits (~2.8GB model VRAM).

Example:
    >>> from src.models.qwen_vl import QwenVLQuantizer
    >>> quantizer = QwenVLQuantizer()
    >>> model, processor = quantizer.load("Qwen/Qwen2-VL-2B-Instruct")
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class QwenVLQuantizer:
    """Load Qwen2-VL model with 4-bit BitsAndBytes quantization.

    Args:
        model_name: HuggingFace model identifier.
        quantization_bits: Bit precision (default: 4).
        device_map: Device placement strategy (default: "auto").
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        quantization_bits: int = 4,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.quantization_bits = quantization_bits
        self.device_map = device_map

    def get_bnb_config(self):
        """Create BitsAndBytesConfig for 4-bit NF4 double quantization."""
        from transformers import BitsAndBytesConfig

        if self.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                # Keep visual.merger in fp16 — required so requires_grad_(True)
                # works later. Quantized (uint8) tensors cannot have gradients.
                llm_int8_skip_modules=["visual.merger"],
            )
        elif self.quantization_bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None

    def load(self, model_name: Optional[str] = None) -> Tuple[Any, Any]:
        """Load 4-bit quantized model and AutoProcessor.

        Args:
            model_name: Optional override model identifier.

        Returns:
            Tuple[model, processor].
        """
        target_model = model_name or self.model_name
        logger.info(f"Loading {target_model} with {self.quantization_bits}-bit quantization...")

        bnb_config = self.get_bnb_config()

        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                target_model,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            processor = AutoProcessor.from_pretrained(target_model, trust_remote_code=True)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"Loaded {target_model} successfully. CUDA memory allocated: {allocated:.2f} GB")

            return model, processor
        except Exception as e:
            logger.error(f"Failed to load {target_model}: {e}")
            raise
