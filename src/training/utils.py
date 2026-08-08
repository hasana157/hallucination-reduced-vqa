"""Training Utilities, Datasets, Data Collators, and Metric functions.

Example:
    >>> from src.training.utils import TrainingConfig, compute_vqa_accuracy
    >>> acc = compute_vqa_accuracy(["cat"], [["cat", "tabby"]])
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Dataclass holding all QLoRA fine-tuning hyperparameters."""

    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    quantization_bits: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
    )
    learning_rate: float = 1e-4
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 500
    save_steps: int = 500
    eval_steps: int = 1000
    save_total_limit: int = 3
    seed: int = 42
    output_dir: str = "checkpoints/lora_weights"
    logging_dir: str = "logs/training"
    use_unsloth: bool = False  # Set True to use Unsloth 2x speedup (requires: pip install unsloth)


class VQADataset(Dataset):
    """PyTorch Dataset wrapping VQAv2 loader outputs.

    Args:
        data_dict: Dictionary returned by VQAv2Loader.load().
    """

    def __init__(self, data_dict: Dict[int, Dict[str, Any]]):
        self.data_items = list(data_dict.values())

    def __len__(self) -> int:
        return len(self.data_items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data_items[idx]
        return {
            "image_path": item["image_path"],
            "question": item["question"],
            "answers": item["answers"],
            "question_id": item["question_id"],
        }


class VQACollator:
    """Data collator formatting VQA image and text samples into Qwen2-VL inputs.

    Args:
        processor: Qwen2VL processor instance.
    """

    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch items into tensors.

        Label masking strategy: prompt tokens are masked with -100 so the
        cross-entropy loss is computed only over the answer tokens. This
        prevents the model from wasting capacity re-generating its own prompt.
        """
        from PIL import Image

        images = []
        texts = []
        answer_texts = []

        for item in batch:
            img_p = Path(item["image_path"])
            if img_p.exists():
                try:
                    img = Image.open(img_p).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), color="black")
            else:
                img = Image.new("RGB", (224, 224), color="black")

            images.append(img)

            # Format Qwen2-VL conversational prompt (full sequence)
            answer_text = item["answers"][0] if item["answers"] else ""
            answer_texts.append(answer_text)
            full_prompt = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer_text}<|im_end|>"
            )
            texts.append(full_prompt)

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # --- Label masking: mask all prompt tokens, keep only answer tokens ---
        labels = inputs["input_ids"].clone()
        for i, answer_text in enumerate(answer_texts):
            # Build the prompt-only prefix (without the answer) to find its token length
            prefix = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{batch[i]['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                # Tokenize the prefix to find where the answer starts
                prefix_ids = self.processor.tokenizer(
                    prefix,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids
                prompt_len = prefix_ids.shape[-1]
            except Exception:
                # Fallback: mask first 80% of tokens as prompt
                prompt_len = int(labels[i].shape[-1] * 0.8)

            # Mask padding tokens and prompt tokens
            labels[i, :prompt_len] = -100
            # Also mask padding
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", 0) or 0
            labels[i, labels[i] == pad_token_id] = -100

        inputs["labels"] = labels
        return inputs


def compute_vqa_accuracy(predictions: List[str], ground_truths: List[List[str]]) -> float:
    """Compute VQA accuracy (exact match, case-insensitive).

    VQA official accuracy rule: min(1.0, count(matching_answers) / 3)

    Args:
        predictions: Model generated answer strings.
        ground_truths: List of valid ground truth answer strings per question.

    Returns:
        Accuracy float between 0.0 and 1.0.
    """
    if not predictions or len(predictions) != len(ground_truths):
        return 0.0

    scores = []
    for pred, gt_list in zip(predictions, ground_truths):
        pred_norm = pred.strip().lower()
        gt_norm = [gt.strip().lower() for gt in gt_list]
        matches = sum(1 for gt in gt_norm if gt == pred_norm)
        acc = min(1.0, matches / 3.0)
        scores.append(acc)

    return float(sum(scores) / len(scores)) if scores else 0.0
