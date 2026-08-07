"""QLoRATrainer class orchestrating QLoRA fine-tuning loop for Qwen2-VL-2B.

Supports HuggingFace `Trainer` and optional Unsloth 2x speedup backend.

Example:
    >>> from src.training.trainer import QLoRATrainer
    >>> trainer = QLoRATrainer(model, processor, train_dataset, val_dataset, config)
    >>> trainer.train()
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import Trainer, TrainingArguments

from src.training.callbacks import CheckpointCallback, MemoryCallback
from src.training.utils import TrainingConfig, VQACollator, compute_vqa_accuracy
from src.utils.helpers import set_seed

logger = logging.getLogger(__name__)


class QLoRATrainer:
    """Trainer orchestrator for QLoRA fine-tuning on VQAv2.

    Args:
        model: PEFT LoRA wrapped model instance.
        processor: Qwen2-VL processor.
        train_dataset: PyTorch Dataset for training split.
        val_dataset: PyTorch Dataset for validation split.
        config: TrainingConfig hyperparameters.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()

        set_seed(self.config.seed)

        self.output_dir = Path(self.config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collator = VQACollator(processor=self.processor)
        self._hf_trainer = None

    def setup(self):
        """Prepare HuggingFace TrainingArguments and Trainer."""
        logger.info("Setting up TrainingArguments for QLoRA fine-tuning...")

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=1.0,
            weight_decay=0.01,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps" if self.val_dataset else "no",
            eval_steps=self.config.eval_steps if self.val_dataset else None,
            fp16=True,
            logging_steps=50,
            seed=self.config.seed,
            report_to="none",
            remove_unused_columns=False,
        )

        self._hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collator,
            callbacks=[CheckpointCallback(), MemoryCallback()],
        )

        logger.info("QLoRATrainer setup complete.")

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Execute QLoRA fine-tuning loop.

        Args:
            resume_from_checkpoint: Optional checkpoint directory path to resume from.
        """
        if self._hf_trainer is None:
            self.setup()

        logger.info(f"Starting QLoRA fine-tuning (Epochs={self.config.num_epochs}, LR={self.config.learning_rate})...")
        
        train_result = self._hf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final LoRA adapter
        final_adapter_dir = self.output_dir / "adapter_model"
        self.model.save_pretrained(str(final_adapter_dir))
        logger.info(f"Saved final LoRA adapter to {final_adapter_dir}")

        return train_result
