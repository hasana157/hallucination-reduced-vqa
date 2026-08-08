"""Unit tests for Module 2 QLoRA Training Pipeline."""

import pytest

from src.models import LoRAAdapter
from src.training.utils import TrainingConfig, VQADataset, compute_vqa_accuracy


def test_training_config_defaults():
    """TrainingConfig has correct SRS hyperparameter defaults."""
    cfg = TrainingConfig()
    assert cfg.quantization_bits == 4
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.learning_rate == 1e-4
    assert cfg.num_epochs == 2
    assert cfg.seed == 42
    assert "q_proj" in cfg.target_modules
    assert "v_proj" in cfg.target_modules


def test_vqa_accuracy_metric():
    """VQA accuracy exact match calculation."""
    preds = ["cat", "dog", "blue"]
    gts = [["cat", "cat", "cat"], ["dog", "dog", "dog"], ["red", "green", "yellow"]]
    acc = compute_vqa_accuracy(preds, gts)
    assert acc == pytest.approx(0.6666, abs=0.01)


def test_vqa_dataset_wrapping():
    """VQADataset wraps loader dict items correctly."""
    mock_dict = {
        0: {"image_path": "img0.jpg", "question": "Q0", "answers": ["A0"], "question_id": 100},
        1: {"image_path": "img1.jpg", "question": "Q1", "answers": ["A1"], "question_id": 101},
    }
    dataset = VQADataset(mock_dict)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["question"] == "Q0"
    assert sample["answers"] == ["A0"]
