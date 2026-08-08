"""Inference module: pipeline, RAG, utilities, and configuration.

Provides the unified InferencePipeline consumed by Module 4 (Evaluation)
and the scripts/inference.py CLI entry point.

Example:
    >>> from src.inference import InferencePipeline, InferenceConfig, InferenceResult
    >>> config = InferenceConfig(lora_path="checkpoints/lora_weights")
    >>> pipeline = InferencePipeline(config)
    >>> result = pipeline.run(image, "What is on the table?", mode="C")
"""

from src.inference.pipeline import InferencePipeline
from src.inference.rag import EntropyScorer, UncertaintyTriggeredRAG
from src.inference.utils import InferenceConfig, InferenceResult, load_inference_model

__all__ = [
    "InferencePipeline",
    "InferenceConfig",
    "InferenceResult",
    "EntropyScorer",
    "UncertaintyTriggeredRAG",
    "load_inference_model",
]
