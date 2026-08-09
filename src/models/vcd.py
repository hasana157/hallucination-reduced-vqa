"""Visual Contrastive Decoding (VCD) for hallucination suppression.

Implements inference-time hallucination reduction by contrasting model
predictions on the original image vs. a deliberately blurred (distorted)
image. The difference isolates hallucinations caused by language-prior
co-occurrence bias rather than actual visual evidence.

Reference: Leng et al. (2024) "Mitigating Object Hallucinations in Large
Vision-Language Models through Visual Contrastive Decoding"

Math:
    final_logits = alpha * logits_A - (1 - alpha) * logits_B
    where logits_A = model(original_image, prompt)
          logits_B = model(blurred_image, prompt)

Example:
    >>> from src.models.vcd import GaussianBlurDistorter, VisualContrastiveDecoder
    >>> distorter = GaussianBlurDistorter(blur_radius=15)
    >>> blurred = distorter.distort(image)
    >>> vcd = VisualContrastiveDecoder(model, processor, alpha=0.5)
    >>> answer, latency = vcd.generate(image, question, input_ids)
"""

import logging
import time
from typing import Any, Optional, Tuple

import torch
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
GAUSSIAN_BLUR_RADIUS: int = 15          # SRS "distorted image" spec
VCD_ALPHA: float = 0.5                   # Logit combination weight (SRS spec)
MAX_VCD_OVERHEAD_MS: int = 500          # Max acceptable VCD overhead (SRS NFG)
DEFAULT_MAX_NEW_TOKENS: int = 20         # Token-by-token loop limit


class GaussianBlurDistorter:
    """Apply deterministic Gaussian blur to an input image.

    Used by VisualContrastiveDecoder to create the distorted image
    whose logits are subtracted from the original to suppress hallucinations.

    Args:
        blur_radius: Gaussian blur kernel radius (default: 15).
        seed: Random seed for determinism (default: 42).
    """

    def __init__(self, blur_radius: int = GAUSSIAN_BLUR_RADIUS, seed: int = 42) -> None:
        self.blur_radius = blur_radius
        self.seed = seed

    def distort(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur to produce distorted image.

        Args:
            image: Input PIL Image (RGB).

        Returns:
            Blurred PIL Image of the same size and mode.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

        blurred = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        logger.debug(
            f"GaussianBlurDistorter: applied blur radius={self.blur_radius} "
            f"to image size={image.size}"
        )
        return blurred

    def __repr__(self) -> str:
        return f"GaussianBlurDistorter(blur_radius={self.blur_radius}, seed={self.seed})"


class VisualContrastiveDecoder:
    """Token-by-token Visual Contrastive Decoding for hallucination suppression.

    For each decoding step:
      1. Run forward pass on (original_image, current_ids) → logits_A
      2. Run forward pass on (blurred_image, current_ids)  → logits_B
      3. Combine: final_logits = alpha * logits_A - (1 - alpha) * logits_B
      4. Pick next token via argmax (greedy)
      5. Repeat until EOS or max_new_tokens

    Args:
        model: Qwen2-VL model (with or without LoRA).
        processor: Qwen2-VL AutoProcessor for image/text preprocessing.
        alpha: Logit combination weight (default: 0.5, SRS spec).
        blur_radius: Gaussian blur radius for distorted image (default: 15).
        max_new_tokens: Maximum tokens to generate (default: 20).
        seed: Reproducibility seed.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        alpha: float = VCD_ALPHA,
        blur_radius: int = GAUSSIAN_BLUR_RADIUS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.processor = processor
        self.alpha = alpha
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.distorter = GaussianBlurDistorter(blur_radius=blur_radius, seed=seed)

    def _prepare_inputs(
        self, image: Image.Image, prompt_text: str, device: str
    ) -> dict:
        """Prepare model inputs using the processor.

        Args:
            image: PIL Image.
            prompt_text: Full formatted prompt string.
            device: Target device string.

        Returns:
            Dict of model input tensors on the specified device.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            # Qwen2-VL uses apply_chat_template for prompt formatting
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            # Fallback: plain tokenizer without image tokens
            inputs = self.processor(
                text=[prompt_text],
                padding=True,
                return_tensors="pt",
            )

        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _get_device(self) -> str:
        """Resolve model device string."""
        try:
            p = next(self.model.parameters())
            dev = getattr(p, "device", "cpu")
            if isinstance(dev, torch.device):
                return dev.type
            if isinstance(dev, str):
                return dev
            if hasattr(dev, "type") and isinstance(dev.type, str):
                return dev.type
            return "cpu"
        except Exception:
            return "cpu"

    def generate(
        self,
        original_image: Image.Image,
        question: str,
        evidence: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Run token-by-token VCD generation.

        Args:
            original_image: The original PIL Image.
            question: The VQA question string.
            evidence: Optional RAG evidence string to prepend to prompt.

        Returns:
            Tuple of (generated_answer: str, vcd_overhead_seconds: float)
            where vcd_overhead_seconds is the extra time for blurred-image passes.
        """
        torch.manual_seed(self.seed)

        device = self._get_device()
        prompt = question if evidence is None else f"{evidence}\n\n{question}"

        blurred_image = self.distorter.distort(original_image)

        # Prepare initial inputs for both images
        orig_inputs = self._prepare_inputs(original_image, prompt, device)
        blur_inputs = self._prepare_inputs(blurred_image, prompt, device)

        # Resolve EOS token
        eos_token_id = getattr(self.processor, "eos_token_id", None)
        if eos_token_id is None and hasattr(self.processor, "tokenizer"):
            eos_token_id = self.processor.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = 2  # common fallback

        generated_ids: list[int] = []
        vcd_overhead_seconds: float = 0.0

        # Get base input_ids to extend token-by-token
        input_ids = orig_inputs.get("input_ids")
        if input_ids is None:
            logger.error("No input_ids in processor output — cannot run VCD loop.")
            raise ValueError("Processor did not return input_ids.")

        current_orig_ids = input_ids.clone()
        current_blur_ids = blur_inputs.get("input_ids", input_ids).clone()

        # Pixel values (images) are fixed throughout the loop
        orig_pixel_values = orig_inputs.get("pixel_values")
        blur_pixel_values = blur_inputs.get("pixel_values")

        logger.debug(
            f"VCD: starting token-by-token loop "
            f"(max_new_tokens={self.max_new_tokens}, alpha={self.alpha})"
        )

        for step in range(self.max_new_tokens):
            # --- Original image forward pass ---
            orig_forward_kwargs = {k: v for k, v in orig_inputs.items()}
            orig_forward_kwargs["input_ids"] = current_orig_ids
            orig_forward_kwargs["attention_mask"] = torch.ones_like(current_orig_ids)
            if orig_pixel_values is not None:
                orig_forward_kwargs["pixel_values"] = orig_pixel_values

            with torch.no_grad():
                orig_out = self.model(**orig_forward_kwargs)
            logits_a = orig_out.logits[:, -1, :]  # (1, vocab_size)

            # --- Blurred image forward pass (VCD overhead) ---
            blur_start = time.perf_counter()
            blur_forward_kwargs = {k: v for k, v in blur_inputs.items()}
            blur_forward_kwargs["input_ids"] = current_blur_ids
            blur_forward_kwargs["attention_mask"] = torch.ones_like(current_blur_ids)
            if blur_pixel_values is not None:
                blur_forward_kwargs["pixel_values"] = blur_pixel_values

            with torch.no_grad():
                blur_out = self.model(**blur_forward_kwargs)
            logits_b = blur_out.logits[:, -1, :]  # (1, vocab_size)
            vcd_overhead_seconds += time.perf_counter() - blur_start

            # --- Combine logits: alpha * A - (1 - alpha) * B ---
            combined_logits = self.alpha * logits_a - (1.0 - self.alpha) * logits_b  # (1, vocab_size)

            next_token_id = combined_logits.argmax(dim=-1)  # (1,)
            token_val = next_token_id.item()

            if token_val == eos_token_id:
                logger.debug(f"VCD: EOS reached at step {step}")
                break

            generated_ids.append(token_val)

            # Extend both sequences with the newly chosen token
            next_token_tensor = next_token_id.unsqueeze(-1)  # (1, 1)
            current_orig_ids = torch.cat([current_orig_ids, next_token_tensor], dim=-1)
            current_blur_ids = torch.cat([current_blur_ids, next_token_tensor], dim=-1)

        # Decode generated token IDs to string
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        overhead_ms = vcd_overhead_seconds * 1000
        logger.info(
            f"VCD: generated {len(generated_ids)} tokens in {self.max_new_tokens} steps. "
            f"VCD overhead (blurred passes): {overhead_ms:.1f}ms "
            f"({'OK' if overhead_ms < MAX_VCD_OVERHEAD_MS else 'EXCEEDS TARGET'})"
        )

        return answer, vcd_overhead_seconds

    def __repr__(self) -> str:
        return (
            f"VisualContrastiveDecoder("
            f"alpha={self.alpha}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"distorter={self.distorter})"
        )
