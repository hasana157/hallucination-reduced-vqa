"""FAISS Index Builder and Retriever for Module 1 & Module 3 RAG.

Example:
    >>> from src.data.retrieval import FAISSIndexBuilder, FAISSRetriever
    >>> builder = FAISSIndexBuilder()
    >>> index = builder.build(embeddings)
    >>> builder.save(index, "data/faiss_index", captions)
    >>> retriever = FAISSRetriever("data/faiss_index")
    >>> results = retriever.retrieve(query_embedding, k=3)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """Build and persist FAISS index from caption embeddings."""

    def __init__(self, index_type: str = "FlatL2"):
        self.index_type = index_type

    def build(self, embeddings: np.ndarray):
        """Build FAISS L2 index from 512-dim embeddings array.

        Args:
            embeddings: np.ndarray shape (N, 512), float32.

        Returns:
            faiss.Index or dict fallback object.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array (N, dim), got shape {embeddings.shape}")

        dim = embeddings.shape[1]
        embeddings_f32 = embeddings.astype(np.float32)

        try:
            import faiss
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings_f32)
            logger.info(f"Built FAISS IndexFlatL2 with {index.ntotal} vectors of dim {dim}")
            return index
        except ImportError:
            logger.warning("faiss package not installed; using NumPy fallback index object.")
            class NumpyIndexFallback:
                def __init__(self, data):
                    self.data = data
                    self.ntotal = len(data)
                def search(self, query, k):
                    # L2 distance: ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q.x
                    dists = np.sum((self.data - query)**2, axis=1)
                    idx = np.argsort(dists)[:k]
                    return np.array([dists[idx]]), np.array([idx])
            return NumpyIndexFallback(embeddings_f32)

    def save(self, index, output_dir: Union[str, Path], captions: List[str]) -> None:
        """Save binary index, embeddings npy, and mapping json.

        Args:
            index: FAISS index or fallback instance.
            output_dir: Directory path to store files.
            captions: List of caption strings matching index order.
        """
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        index_file = out_path / "index.faiss"
        mapping_file = out_path / "caption_mapping.json"
        npy_file = out_path / "caption_embeddings.npy"

        try:
            import faiss
            if hasattr(index, "data"):
                np.save(npy_file, index.data)
            else:
                faiss.write_index(index, str(index_file))
        except ImportError:
            if hasattr(index, "data"):
                np.save(npy_file, index.data)
            else:
                with open(index_file, "w") as f:
                    f.write("placeholder")

        mapping = {idx: caption for idx, caption in enumerate(captions)}
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

        logger.info(f"Saved index and mapping to {out_path}")


class FAISSRetriever:
    """Retrieve top-k similar captions given a query embedding."""

    def __init__(self, index_dir: Union[str, Path]):
        self.index_dir = Path(index_dir).resolve()
        self.index_file = self.index_dir / "index.faiss"
        self.npy_file = self.index_dir / "caption_embeddings.npy"
        self.mapping_file = self.index_dir / "caption_mapping.json"
        self._index = None
        self._mapping = None

    def _load(self):
        if self._index is None:
            if not self.mapping_file.exists():
                raise FileNotFoundError(f"Caption mapping file missing at {self.mapping_file}")

            with open(self.mapping_file, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
                self._mapping = {int(k): v for k, v in raw_mapping.items()}

            try:
                import faiss
                if self.index_file.exists():
                    self._index = faiss.read_index(str(self.index_file))
                elif self.npy_file.exists():
                    data = np.load(self.npy_file)
                    dim = data.shape[1]
                    idx = faiss.IndexFlatL2(dim)
                    idx.add(data.astype(np.float32))
                    self._index = idx
                else:
                    raise FileNotFoundError(f"Neither index.faiss nor caption_embeddings.npy found in {self.index_dir}")
            except ImportError:
                logger.warning("faiss not installed; using NumPy retriever fallback.")
                if self.npy_file.exists():
                    data = np.load(self.npy_file)
                else:
                    data = np.zeros((len(self._mapping), 512), dtype=np.float32)

                class NumpyIndexFallback:
                    def __init__(self, data):
                        self.data = data
                        self.ntotal = len(data)
                    def search(self, query, k):
                        dists = np.sum((self.data - query)**2, axis=1)
                        idx_arr = np.argsort(dists)[:k]
                        return np.array([dists[idx_arr]]), np.array([idx_arr])

                self._index = NumpyIndexFallback(data)

            logger.info(f"Loaded index with {getattr(self._index, 'ntotal', 0)} vectors")

    def retrieve(self, query_embedding: np.ndarray, k: int = 3) -> List[str]:
        """Query FAISS index for top-k captions.

        Args:
            query_embedding: Query vector of shape (512,) or (1, 512).
            k: Top-k elements to retrieve (default: 3).

        Returns:
            List[str] of top-k captions.
        """
        self._load()

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_f32 = query_embedding.astype(np.float32)
        distances, indices = self._index.search(query_f32, k)

        top_captions = []
        for idx in indices[0]:
            if idx in self._mapping:
                top_captions.append(self._mapping[idx])
            else:
                top_captions.append("")

        return top_captions
