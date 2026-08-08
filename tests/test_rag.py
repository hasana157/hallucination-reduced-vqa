"""Unit tests for RAG Module (Module 3)."""

from src.inference.rag import EntropyScorer, UncertaintyTriggeredRAG


def test_entropy_scorer():
    """Verify EntropyScorer calculation."""
    import torch
    logits = torch.tensor([10.0, -10.0, -10.0])
    entropy = EntropyScorer.compute(logits)
    assert entropy < 0.1
