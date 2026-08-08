"""Training package for QLoRA fine-tuning on VQAv2."""

from src.training.utils import TrainingConfig, VQACollator, VQADataset, compute_vqa_accuracy

def __getattr__(name):
    if name == "QLoRATrainer":
        from src.training.trainer import QLoRATrainer
        return QLoRATrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "QLoRATrainer",
    "TrainingConfig",
    "VQADataset",
    "VQACollator",
    "compute_vqa_accuracy",
]
