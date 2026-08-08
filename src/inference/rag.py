"""Uncertainty-Triggered Retrieval-Augmented Generation (RAG) for VQA.

Implements entropy-based confidence scoring to selectively retrieve evidence
captions from the FAISS index only when the model is uncertain. This avoids
constant retrieval overhead while providing grounding when needed.

Trigger Mechanism:
    - Compute first-token prediction entropy from a quick forward pass
    - Normalized entropy H(p) / log(vocab_size) in [0, 1]
    - If entropy > threshold (0.8): retrieve top-3 captions via FAISS+CLIP
    - Evidence format: "Evidence: [cap1] | [cap2] | [cap3]"
    - Target activation rate: 25-30% over representative batch (SRS FG-5)

Example:
    >>> from src.inference.rag import EntropyScorer, UncertaintyTriggeredRAG
    >>> scorer = EntropyScorer()
    >>> entropy = scorer.compute(logits)  # float in [0, 1]
    >>> rag = UncertaintyTriggeredRAG(model, processor, retriever, config)
    >>> result = rag.check_and_retrieve(image, question)
    >>> # result = {"triggered": True, "evidence": "Evidence: ...", "entropy_score": 0.87}
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.inference.utils import InferenceConfig

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
ENTROPY_THRESHOLD: float = 0.8
RAG_TOP_K: int = 3
EVIDENCE_SEPARATOR: str = " | "
EVIDENCE_PREFIX: str = "Evidence: "


class EntropyScorer:
    """Compute normalized prediction entropy from first-token logits.

    Uses Shannon entropy normalized by log(vocab_size) so the result
    always falls in [0, 1]:
        H(p) = -Σ p_i * log(p_i)
        H_norm = H(p) / log(vocab_size)

    High entropy (~1.0) → model is uncertain → trigger RAG.
    Low entropy  (~0.0) → model is confident → skip RAG.
    """

    @staticmethod
    def compute(logits: torch.Tensor) -> float:
        """Compute normalized entropy from raw logit tensor.

        Args:
            logits: Raw logits of shape (vocab_size,) or (1, vocab_size)
                    or (batch, seq_len, vocab_size). If 3D, uses last position
                    of first batch element.

        Returns:
            Normalized entropy in [0.0, 1.0]. Returns 1.0 on any error
            (conservative: triggers RAG when uncertain about the score).
        """
        try:
            t = logits.detach().float()

            # Flatten to 1D (vocab_size,)
            if t.dim() == 3:
                t = t[0, -1, :]   # (vocab_size,)
            elif t.dim() == 2:
                t = t[0, :]       # (vocab_size,)
            # else already (vocab_size,)

            vocab_size = t.shape[0]
            if vocab_size < 2:
                return 1.0

            probs = torch.softmax(t, dim=-1)  # (vocab_size,)

            # Clamp to avoid log(0)
            probs_clamped = probs.clamp(min=1e-10)
            raw_entropy = -torch.sum(probs_clamped * torch.log(probs_clamped)).item()

            max_entropy = math.log(vocab_size)
            normalized = raw_entropy / max_entropy if max_entropy > 0 else 1.0

            # Clamp to [0, 1] for floating-point safety
            return float(max(0.0, min(1.0, normalized)))

        except Exception as e:
            logger.warning(f"EntropyScorer.compute failed: {e}. Returning 1.0 (trigger).")
            return 1.0


class UncertaintyTriggeredRAG:
    """Orchestrates entropy-gated retrieval for RAG-augmented VQA inference.

    Flow:
        1. Quick single forward pass (no grad) → get first-token logits
        2. EntropyScorer.compute() → entropy in [0, 1]
        3. If entropy > threshold → embed question text with CLIP
        4. FAISSRetriever.retrieve(embedding, k=top_k) → captions
        5. Format: "Evidence: cap1 | cap2 | cap3"

    Tracks running activation_count / total_queries for batch rate reporting.

    Args:
        model: Language model for entropy forward pass.
        processor: Tokenizer/processor for input preparation.
        retriever: FAISSRetriever instance (from Module 1).
        config: InferenceConfig with entropy_threshold, top_k, clip_model.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        retriever: Any,
        config: InferenceConfig,
    ) -> None:
        self.model = model
        self.processor = processor
        self.retriever = retriever
        self.config = config

        # Running counters for activation rate
        self._total_queries: int = 0
        self._triggered_queries: int = 0

        # Lazy-loaded CLIP embedder
        self._clip_model: Optional[Any] = None
        self._clip_tokenizer: Optional[Any] = None
        self._clip_device: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_retrieve(
        self,
        image: Image.Image,
        question: str,
    ) -> Dict[str, Any]:
        """Check uncertainty and optionally retrieve evidence.

        Args:
            image: PIL Image for the VQA query.
            question: The question string.

        Returns:
            Dict with keys:
                - triggered (bool): Whether RAG retrieval fired.
                - evidence (Optional[str]): Formatted evidence string or None.
                - entropy_score (float): Normalized entropy value.
        """
        self._total_queries += 1
        entropy_score = self._compute_entropy_for_query(image, question)

        triggered = entropy_score > self.config.entropy_threshold
        evidence: Optional[str] = None

        logger.info(
            f"RAG check: entropy={entropy_score:.3f}, "
            f"threshold={self.config.entropy_threshold}, "
            f"triggered={triggered} "
            f"[query {self._total_queries}]"
        )

        if triggered:
            self._triggered_queries += 1
            evidence = self._retrieve_evidence(question)

        return {
            "triggered": triggered,
            "evidence": evidence,
            "entropy_score": entropy_score,
        }

    @property
    def activation_rate(self) -> float:
        """Compute current RAG activation rate over all processed queries.

        Returns:
            Float in [0, 1], or 0.0 if no queries processed yet.
        """
        if self._total_queries == 0:
            return 0.0
        return self._triggered_queries / self._total_queries

    def log_activation_summary(self) -> None:
        """Log summary of RAG activation rate for the current batch."""
        rate = self.activation_rate
        target_min, target_max = (0.25, 0.30)
        in_range = target_min <= rate <= target_max
        logger.info(
            f"RAG activation summary: {self._triggered_queries}/{self._total_queries} "
            f"queries triggered ({rate:.1%}) "
            f"[target: {target_min:.0%}–{target_max:.0%}, in_range={in_range}]"
        )

    def reset_counters(self) -> None:
        """Reset activation rate counters (e.g., between evaluation runs)."""
        self._total_queries = 0
        self._triggered_queries = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model_device(self) -> str:
        """Resolve the model's primary device string."""
        try:
            p = next(self.model.parameters())
            dev = getattr(p, "device", "cpu")
            if isinstance(dev, torch.device):
                return dev.type
            if isinstance(dev, str):
                return dev
            if hasattr(dev, "type") and isinstance(dev.type, str):
                return dev.type
            return "cpu"
        except Exception:
            return "cpu"

    def _compute_entropy_for_query(
        self, image: Image.Image, question: str
    ) -> float:
        """Run a minimal forward pass to get first-token logits, then compute entropy.

        Args:
            image: PIL Image.
            question: Question string.

        Returns:
            Normalized entropy float in [0, 1].
        """
        device = self._get_model_device()

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text], images=[image], return_tensors="pt"
                )
            except Exception:
                inputs = self.processor(
                    text=[question], return_tensors="pt"
                )

            inputs = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits  # (batch, seq_len, vocab_size)
            return EntropyScorer.compute(logits)

        except Exception as e:
            logger.warning(
                f"Entropy computation failed: {e}. "
                "Defaulting to entropy=1.0 (conservative trigger)."
            )
            return 1.0

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed the question text using CLIP for FAISS retrieval.

        Args:
            text: Query text (question).

        Returns:
            np.ndarray of shape (512,), float32, L2-normalized.
        """
        if self._clip_model is None:
            self._load_clip()

        import torch
        inputs = self._clip_tokenizer(
            [text], padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self._clip_device)

        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            embeds = outputs.text_embeds  # (1, 512)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)

        return embeds[0].cpu().numpy().astype(np.float32)

    def _load_clip(self) -> None:
        """Lazy-load CLIP text model for query embedding."""
        import torch
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_device = device
        logger.info(f"Loading CLIP model '{self.config.clip_model}' for RAG...")
        self._clip_tokenizer = AutoTokenizer.from_pretrained(self.config.clip_model)
        self._clip_model = CLIPTextModelWithProjection.from_pretrained(
            self.config.clip_model
        ).to(device)
        self._clip_model.eval()

    def _retrieve_evidence(self, question: str) -> Optional[str]:
        """Embed query and retrieve top-k captions from FAISS.

        Args:
            question: Question text for embedding.

        Returns:
            Formatted evidence string or None on failure.
        """
        try:
            query_embedding = self._embed_query(question)
            captions: List[str] = self.retriever.retrieve(
                query_embedding, k=self.config.top_k
            )
            # Filter empty strings
            captions = [c for c in captions if c.strip()]
            if not captions:
                logger.warning("FAISS returned no captions; skipping evidence.")
                return None

            formatted = EVIDENCE_PREFIX + EVIDENCE_SEPARATOR.join(captions)
            logger.debug(f"RAG evidence: {formatted[:120]}...")
            return formatted

        except Exception as e:
            logger.warning(
                f"RAG retrieval failed, proceeding without evidence: {e}"
            )
            return None
