"""Unit tests for Module 1 Data Preparation & Retrieval."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.data import CaptionLoader, DataValidator, FAISSIndexBuilder, FAISSRetriever, POPELoader


def test_caption_loader():
    """CaptionLoader returns 1000 non-empty strings."""
    loader = CaptionLoader(caption_file="data/captions/training_captions.json")
    captions = loader.load()
    assert len(captions) == 1000
    assert all(isinstance(c, str) for c in captions)
    assert all(len(c.strip()) > 0 for c in captions)


def test_pope_loader_modes():
    """POPELoader returns data for all 3 modes."""
    loader = POPELoader(data_root="./data")
    for mode in ["random", "popular", "adversarial"]:
        items = loader.load(mode=mode)
        assert len(items) > 0
        assert "text" in items[0] or "question_id" in items[0]


def test_faiss_builder_and_retriever():
    """FAISS index builds, saves, loads, and queries top-k correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        captions = [f"Caption sample number {i}" for i in range(100)]
        embeddings = np.random.randn(100, 512).astype(np.float32)

        # 1. Build & Save
        builder = FAISSIndexBuilder()
        index = builder.build(embeddings)
        assert index.ntotal == 100

        builder.save(index, output_dir=tmp_path, captions=captions)

        assert (tmp_path / "index.faiss").exists() or (tmp_path / "caption_embeddings.npy").exists()
        assert (tmp_path / "caption_mapping.json").exists()

        # 2. Load & Retrieve
        retriever = FAISSRetriever(index_dir=tmp_path)
        query_vec = np.random.randn(512).astype(np.float32)
        results = retriever.retrieve(query_vec, k=3)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


def test_data_validator():
    """DataValidator passes validation suite."""
    validator = DataValidator(config={"data": {"root": "./data"}})
    report = validator.validate()
    assert report["passed"] is True
