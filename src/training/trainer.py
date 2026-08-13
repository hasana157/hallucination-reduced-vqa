"""QLoRATrainer class orchestrating QLoRA fine-tuning loop for Qwen2-VL-2B.

Supports HuggingFace ``Trainer`` backend (default) and optional Unsloth 2x
speedup backend. Set ``use_unsloth=True`` in TrainingConfig and install
``unsloth`` to enable the fast path.

Example:
    >>> from src.training.trainer import QLoRATrainer
    >>> trainer = QLoRATrainer(model, processor, train_dataset, val_dataset, config)
    >>> trainer.train()
"""

import logging
from pathlib import Path
from typing import Any, Optional

from transformers import Trainer, TrainingArguments

from src.training.callbacks import CheckpointCallback, MemoryCallback
from src.training.utils import TrainingConfig, VQACollator
from src.utils.helpers import set_seed

logger = logging.getLogger(__name__)

# Detect Unsloth availability once at import time
try:
    import unsloth  # noqa: F401
    _UNSLOTH_AVAILABLE = True
    logger.info("Unsloth detected — fast training path available.")
except ImportError:
    _UNSLOTH_AVAILABLE = False
    logger.info("Unsloth not installed — using standard HuggingFace Trainer.")


class QLoRATrainer:
    """Trainer orchestrator for QLoRA fine-tuning on VQAv2.

    Supports two backends:
      - **Standard** (default): HuggingFace ``Trainer`` with fp16 and gradient
        accumulation. Works everywhere.
      - **Unsloth** (optional): ``FastVisionModel`` + ``SFTTrainer`` for ~2x
        training speed and ~40% lower memory. Requires ``pip install unsloth``.
        Enabled via ``config.use_unsloth = True``.

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

        # Determine which backend to use
        self._use_unsloth = getattr(self.config, "use_unsloth", False) and _UNSLOTH_AVAILABLE
        if getattr(self.config, "use_unsloth", False) and not _UNSLOTH_AVAILABLE:
            logger.warning(
                "config.use_unsloth=True but unsloth is not installed. "
                "Falling back to standard HuggingFace Trainer. "
                "Install with: pip install unsloth"
            )
        backend = "Unsloth" if self._use_unsloth else "HuggingFace Trainer"
        logger.info(f"QLoRATrainer initialized with backend: {backend}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self):
        """Prepare TrainingArguments and Trainer (standard or Unsloth)."""
        if self._use_unsloth:
            self._setup_unsloth()
        else:
            self._setup_standard()

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Execute QLoRA fine-tuning loop.

        Args:
            resume_from_checkpoint: Optional checkpoint directory path to resume from.
        """
        if self._hf_trainer is None:
            self.setup()

        backend = "Unsloth" if self._use_unsloth else "HuggingFace Trainer"
        logger.info(
            f"Starting QLoRA fine-tuning via {backend} "
            f"(Epochs={self.config.num_epochs}, LR={self.config.learning_rate})..."
        )

        train_result = self._hf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final LoRA adapter
        final_adapter_dir = self.output_dir / "adapter_model"
        self.model.save_pretrained(str(final_adapter_dir))
        logger.info(f"Saved final LoRA adapter to {final_adapter_dir}")

        return train_result

    # ------------------------------------------------------------------
    # Private setup helpers
    # ------------------------------------------------------------------

    def _build_training_args(self) -> TrainingArguments:
        """Build shared TrainingArguments used by both backends."""
        import torch
        # Use bf16 when available (T4 supports it) — avoids GradScaler which
        # crashes when fp32 visual.merger params produce fp16 gradients.
        # fp16 GradScaler calls _unscale_grads_ with allow_fp16=False, which
        # raises ValueError on any fp16 gradient tensor.
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16 and torch.cuda.is_available()
        return TrainingArguments(
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
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=50,
            seed=self.config.seed,
            report_to="none",
            remove_unused_columns=False,
        )

    def _setup_standard(self):
        """Standard HuggingFace Trainer setup."""
        logger.info("Setting up standard HuggingFace TrainingArguments...")
        training_args = self._build_training_args()
        self._hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collator,
            callbacks=[CheckpointCallback(), MemoryCallback()],
        )
        logger.info("Standard QLoRATrainer setup complete.")

    def _setup_unsloth(self):
        """Unsloth fast trainer setup (2x speed, ~40% less memory).

        Falls back to standard Trainer on any import or setup error.
        """
        logger.info("Setting up Unsloth fast training path...")
        try:
            from unsloth import FastVisionModel, is_bf16_supported
            from trl import SFTConfig, SFTTrainer

            # Put model in training mode with Unsloth optimizations
            FastVisionModel.for_training(self.model)

            training_args = SFTConfig(
                output_dir=str(self.output_dir),
                learning_rate=self.config.learning_rate,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                max_grad_norm=1.0,
                weight_decay=0.01,
                save_strategy="steps",
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=50,
                seed=self.config.seed,
                report_to="none",
                remove_unused_columns=False,
                # SFTTrainer-specific: skip default dataset prep since we use a custom collator
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
            )

            self._hf_trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.processor.tokenizer,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                data_collator=self.collator,
                callbacks=[CheckpointCallback(), MemoryCallback()],
            )
            logger.info("Unsloth SFTTrainer setup complete.")

        except Exception as e:
            logger.warning(
                f"Unsloth setup failed ({e}). Falling back to standard HuggingFace Trainer."
            )
            self._use_unsloth = False
            self._setup_standard()
