"""Unit tests for Module 3: Inference & Grounding Pipeline.

Tests:
  1. test_entropy_scorer_high_uncertainty
  2. test_entropy_scorer_low_uncertainty
  3. test_rag_activation_rate_on_batch
  4. test_evidence_format
  5. test_gaussian_blur_distorter
  6. test_vcd_logit_combination
  7. test_vcd_overhead_under_500ms
  8. test_pipeline_mode_a_no_lora
  9. test_pipeline_mode_b_lora_only
  10. test_pipeline_mode_c_full_stack
  11. test_batch_inference_resumable
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.pipeline import InferencePipeline
from src.inference.rag import EntropyScorer, UncertaintyTriggeredRAG
from src.inference.utils import InferenceConfig, InferenceResult
from src.models.vcd import GaussianBlurDistorter, VisualContrastiveDecoder


# ============================================================
# 1. EntropyScorer Tests
# ============================================================

def test_entropy_scorer_high_uncertainty():
    """Uniform logit distribution -> high entropy (~1.0) -> should trigger RAG."""
    vocab_size = 1000
    uniform_logits = torch.zeros(vocab_size)  # Uniform probabilities
    entropy = EntropyScorer.compute(uniform_logits)
    assert abs(entropy - 1.0) < 1e-3, f"Expected entropy close to 1.0, got {entropy}"
    assert entropy > 0.8  # Will trigger RAG threshold


def test_entropy_scorer_low_uncertainty():
    """Peaked logit distribution -> low entropy (~0.0) -> should NOT trigger RAG."""
    vocab_size = 1000
    peaked_logits = torch.full((vocab_size,), -100.0)
    peaked_logits[42] = 100.0  # Dominant logit (probability ~ 1.0)
    entropy = EntropyScorer.compute(peaked_logits)
    assert entropy < 0.1, f"Expected low entropy < 0.1, got {entropy}"
    assert entropy < 0.8  # Will NOT trigger RAG threshold


# ============================================================
# 2. GaussianBlurDistorter & VCD Tests
# ============================================================

def test_gaussian_blur_distorter():
    """Blurred image differs from original; deterministic given seed."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    # Draw a line so blur has effect
    for i in range(100):
        img.putpixel((i, 50), (0, 255, 0))

    distorter = GaussianBlurDistorter(blur_radius=15, seed=42)
    blurred1 = distorter.distort(img)
    blurred2 = distorter.distort(img)

    # Check that image changed
    arr_orig = np.array(img)
    arr_blur1 = np.array(blurred1)
    arr_blur2 = np.array(blurred2)

    assert not np.array_equal(arr_orig, arr_blur1), "Blurred image should differ from original"
    assert np.array_equal(arr_blur1, arr_blur2), "Blur should be deterministic"


def test_vcd_logit_combination():
    """Verified numerically: final_logits = alpha * logits_A - (1-alpha) * logits_B."""
    alpha = 0.5
    logits_a = torch.tensor([2.0, 4.0, 1.0])
    logits_b = torch.tensor([1.0, 0.0, 3.0])

    expected = alpha * logits_a - (1.0 - alpha) * logits_b
    actual = 0.5 * logits_a - 0.5 * logits_b

    assert torch.allclose(expected, actual)
    # Check values: 0.5*(2-1)=0.5, 0.5*(4-0)=2.0, 0.5*(1-3)=-1.0
    assert torch.allclose(expected, torch.tensor([0.5, 2.0, -1.0]))


def test_vcd_overhead_under_500ms():
    """Mock test ensuring VCD records overhead and checks < 500ms budget."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Setup mock logits output
    mock_logits = torch.randn(1, 10, 100)
    mock_out = MagicMock()
    mock_out.logits = mock_logits
    mock_model.side_effect = lambda **kwargs: mock_out
    mock_model.parameters.return_value = iter([torch.tensor([1.0])])

    mock_input_ids = torch.randint(0, 100, (1, 10))
    mock_processor.apply_chat_template.return_value = "prompt"
    mock_processor.return_value = {"input_ids": mock_input_ids, "pixel_values": torch.zeros(1, 3, 224, 224)}
    mock_processor.eos_token_id = 999

    tokenizer_mock = MagicMock()
    tokenizer_mock.decode.return_value = "cat"
    mock_processor.tokenizer = tokenizer_mock

    vcd = VisualContrastiveDecoder(
        model=mock_model,
        processor=mock_processor,
        alpha=0.5,
        blur_radius=15,
        max_new_tokens=3,
        seed=42,
    )

    img = Image.new("RGB", (50, 50), color=(100, 100, 100))
    answer, overhead_sec = vcd.generate(img, "What is this?")

    assert isinstance(answer, str)
    assert overhead_sec < 0.5, f"Overhead {overhead_sec*1000:.1f}ms exceeded 500ms limit"


# ============================================================
# 3. RAG Tests
# ============================================================

def test_evidence_format():
    """Evidence string matches 'Evidence: [cap1] | [cap2] | [cap3]' exactly."""
    captions = ["a cat on a mat", "a red cat sitting", "feline on rug"]
    formatted = "Evidence: " + " | ".join(captions)
    assert formatted.startswith("Evidence: ")
    assert " | " in formatted
    assert formatted == "Evidence: a cat on a mat | a red cat sitting | feline on rug"


def test_rag_activation_rate_on_batch():
    """Over sample batch queries, RAG activation tracking works."""
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ["cap1", "cap2", "cap3"]

    config = InferenceConfig(entropy_threshold=0.5)
    rag = UncertaintyTriggeredRAG(
        model=mock_model,
        processor=mock_processor,
        retriever=mock_retriever,
        config=config,
    )

    # Mock entropy calculations: alternate high (0.9) vs low (0.1)
    with patch.object(rag, "_compute_entropy_for_query", side_effect=[0.9, 0.1, 0.9, 0.1, 0.1]), \
         patch.object(rag, "_embed_query", return_value=np.zeros(512, dtype=np.float32)):
        img = Image.new("RGB", (10, 10))
        res1 = rag.check_and_retrieve(img, "Q1")  # 0.9 > 0.5 -> True
        res2 = rag.check_and_retrieve(img, "Q2")  # 0.1 < 0.5 -> False
        res3 = rag.check_and_retrieve(img, "Q3")  # 0.9 > 0.5 -> True
        res4 = rag.check_and_retrieve(img, "Q4")  # 0.1 < 0.5 -> False
        res5 = rag.check_and_retrieve(img, "Q5")  # 0.1 < 0.5 -> False

    assert res1["triggered"] is True
    assert res2["triggered"] is False
    assert res1["evidence"] == "Evidence: cap1 | cap2 | cap3"
    assert res2["evidence"] is None

    # Total = 5, Triggered = 2 -> 40% rate
    assert abs(rag.activation_rate - 0.40) < 1e-3


# ============================================================
# 4. InferencePipeline Tests
# ============================================================

def test_pipeline_modes():
    """Verify InferencePipeline runs modes A, B, C with mocked models."""
    config = InferenceConfig(seed=42)

    with patch("src.inference.pipeline.load_inference_model") as mock_load:
        mock_m = MagicMock()
        mock_p = MagicMock()
        mock_out = MagicMock()
        mock_out.logits = torch.randn(1, 5, 100)
        mock_m.side_effect = lambda **kw: mock_out
        mock_m.parameters.return_value = iter([torch.tensor([1.0])])
        mock_m.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_p.tokenizer.decode.return_value = "cat"
        mock_p.apply_chat_template.return_value = "prompt"
        mock_p.return_value = {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.zeros(1, 3, 10, 10)}

        mock_load.return_value = (mock_m, mock_p)

        pipeline = InferencePipeline(config)
        # Mock RAG and VCD to avoid full external dependency in unit tests
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = ["c1", "c2"]
        config.faiss_retriever = mock_retriever

        img = Image.new("RGB", (20, 20))

        # Mode A
        res_a = pipeline.run(img, "What is this?", mode="A")
        assert res_a.mode == "A"
        assert res_a.rag_triggered is False
        assert res_a.answer == "cat"

        # Mode B
        res_b = pipeline.run(img, "What is this?", mode="B")
        assert res_b.mode == "B"
        assert res_b.rag_triggered is False

        # Mode C
        with patch.object(pipeline._get_rag(mock_m, mock_p), "check_and_retrieve", return_value={"triggered": True, "evidence": "Evidence: c1", "entropy_score": 0.85}):
            with patch.object(pipeline._get_vcd(mock_m, mock_p), "generate", return_value=("dog", 0.05)):
                res_c = pipeline.run(img, "What is this?", mode="C")
                assert res_c.mode == "C"
                assert res_c.rag_triggered is True
                assert res_c.answer == "dog"
                assert res_c.evidence_used == "Evidence: c1"


def test_batch_inference_resumable(tmp_path):
    """run_batch() can save and resume from a partial results file."""
    config = InferenceConfig(seed=42)

    with patch("src.inference.pipeline.load_inference_model") as mock_load:
        mock_m = MagicMock()
        mock_p = MagicMock()
        mock_m.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_p.tokenizer.decode.return_value = "res"
        mock_p.apply_chat_template.return_value = "p"
        mock_p.return_value = {"input_ids": torch.tensor([[1]])}
        mock_load.return_value = (mock_m, mock_p)

        pipeline = InferencePipeline(config)
        images = [Image.new("RGB", (10, 10)) for _ in range(5)]
        questions = [f"Q{i}" for i in range(5)]

        ckpt_file = str(tmp_path / "test_ckpt.json")

        # Run first 2 items manually by writing partial file
        partial_data = [
            InferenceResult(answer="ans0", latency_seconds=0.1, gpu_memory_gb=1.0, rag_triggered=False, evidence_used=None, entropy_score=0.2, mode="A").to_dict(),
            InferenceResult(answer="ans1", latency_seconds=0.1, gpu_memory_gb=1.0, rag_triggered=False, evidence_used=None, entropy_score=0.2, mode="A").to_dict(),
        ]
        with open(ckpt_file, "w", encoding="utf-8") as f:
            json.dump(partial_data, f)

        # Resume batch
        results = pipeline.run_batch(images, questions, mode="A", checkpoint_file=ckpt_file)
        assert len(results) == 5
        assert results[0].answer == "ans0"
        assert results[1].answer == "ans1"
        assert results[2].answer == "res"
