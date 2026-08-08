"""Inference utilities: configuration, result dataclasses, and model loading.

Provides the shared types and model assembly logic consumed by InferencePipeline.
Supports loading with and without LoRA adapters (Configurations A vs B/C).

Example:
    >>> from src.inference.utils import InferenceConfig, load_inference_model
    >>> config = InferenceConfig(lora_path="checkpoints/lora_weights")
    >>> model, processor = load_inference_model(config, use_lora=True)
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ============================================================
# Constants (from SRS)
# ============================================================
ENTROPY_THRESHOLD: float = 0.8
VCD_ALPHA: float = 0.5
RAG_TOP_K: int = 3
GAUSSIAN_BLUR_RADIUS: int = 15
TARGET_RAG_ACTIVATION_RATE: Tuple[float, float] = (0.25, 0.30)
MAX_VCD_OVERHEAD_MS: int = 500
DEFAULT_BASE_MODEL: str = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_CLIP_MODEL: str = "openai/clip-vit-base-patch32"


@dataclass
class InferenceConfig:
    """Configuration for the unified inference pipeline.

    All hyperparameters come from here — no global state.

    Args:
        base_model: HuggingFace model identifier.
        lora_path: Path to trained LoRA adapter weights directory.
        faiss_index_path: Path to FAISS index directory.
        quantization_bits: Quantization precision (4 or 8).
        clip_model: CLIP model identifier for RAG embeddings.
        entropy_threshold: RAG trigger threshold (SRS: 0.8).
        top_k: Number of captions to retrieve from FAISS (SRS: 3).
        vcd_alpha: VCD logit combination weight (SRS: 0.5).
        blur_radius: Gaussian blur radius for VCD distorted image (SRS: 15).
        max_new_tokens: Maximum tokens to generate.
        seed: Global reproducibility seed (SRS: 42).
        device_map: HuggingFace device placement strategy.
        faiss_retriever: Optional pre-instantiated FAISSRetriever (skip loading).
    """

    # Model settings
    base_model: str = DEFAULT_BASE_MODEL
    lora_path: str = "checkpoints/lora_weights"
    quantization_bits: int = 4
    device_map: str = "auto"

    # RAG settings
    faiss_index_path: str = "data/faiss_index"
    clip_model: str = DEFAULT_CLIP_MODEL
    entropy_threshold: float = ENTROPY_THRESHOLD
    top_k: int = RAG_TOP_K

    # VCD settings
    vcd_alpha: float = VCD_ALPHA
    blur_radius: int = GAUSSIAN_BLUR_RADIUS
    max_new_tokens: int = 20

    # Performance
    seed: int = 42

    # Optional pre-instantiated retriever (injected by Module 4 or tests)
    faiss_retriever: Optional[Any] = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "InferenceConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to inference_config.yaml.

        Returns:
            InferenceConfig instance populated from YAML.
        """
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        model_cfg = raw.get("model", {})
        rag_cfg = raw.get("rag", {})
        vcd_cfg = raw.get("vcd", {})
        perf_cfg = raw.get("performance", {})

        return cls(
            base_model=model_cfg.get("base_model", DEFAULT_BASE_MODEL),
            lora_path=model_cfg.get("lora_path", "checkpoints/lora_weights"),
            quantization_bits=model_cfg.get("quantization_bits", 4),
            device_map=model_cfg.get("device_map", "auto"),
            faiss_index_path=rag_cfg.get("faiss_index_path", "data/faiss_index"),
            clip_model=rag_cfg.get("clip_model", DEFAULT_CLIP_MODEL),
            entropy_threshold=rag_cfg.get("entropy_threshold", ENTROPY_THRESHOLD),
            top_k=rag_cfg.get("top_k", RAG_TOP_K),
            vcd_alpha=vcd_cfg.get("alpha", VCD_ALPHA),
            blur_radius=vcd_cfg.get("blur_radius", GAUSSIAN_BLUR_RADIUS),
            max_new_tokens=vcd_cfg.get("max_new_tokens", 20),
            seed=perf_cfg.get("seed", 42),
        )

    def validate_lora_path(self) -> bool:
        """Check LoRA path exists and contains no path traversal components.

        Returns:
            True if path is valid and safe.

        Raises:
            ValueError: If the path contains traversal sequences.
            FileNotFoundError: If the path does not exist.
        """
        raw_path = str(self.lora_path)
        if ".." in raw_path.split(os.sep) or ".." in raw_path.split("/"):
            raise ValueError(
                f"LoRA path contains path traversal sequences: {self.lora_path}"
            )

        resolved = Path(self.lora_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"LoRA checkpoint directory not found: {resolved}"
            )
        logger.debug(f"LoRA path validated: {resolved}")
        return True


@dataclass
class InferenceResult:
    """Structured result from a single inference call.

    Consumed by Module 4 (Evaluator) to compute metrics.

    Args:
        answer: Generated answer string.
        latency_seconds: Wall-clock inference time (seconds).
        gpu_memory_gb: Peak GPU memory used (GB), 0.0 if CPU-only.
        rag_triggered: Whether RAG retrieval fired for this query.
        evidence_used: RAG evidence string if triggered, else None.
        entropy_score: First-token entropy score (always logged).
        mode: Configuration mode — "A", "B", or "C".
        question_id: Optional question ID for traceability.
        image_path: Optional image path for traceability (not the raw image).
    """

    answer: str
    latency_seconds: float
    gpu_memory_gb: float
    rag_triggered: bool
    evidence_used: Optional[str]
    entropy_score: Optional[float]
    mode: str
    question_id: Optional[int] = None
    image_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to JSON-serializable dict."""
        return {
            "answer": self.answer,
            "latency_seconds": self.latency_seconds,
            "gpu_memory_gb": self.gpu_memory_gb,
            "rag_triggered": self.rag_triggered,
            "evidence_used": self.evidence_used,
            "entropy_score": self.entropy_score,
            "mode": self.mode,
            "question_id": self.question_id,
            "image_path": self.image_path,
        }


def load_inference_model(
    config: InferenceConfig,
    use_lora: bool = False,
) -> Tuple[Any, Any]:
    """Load the base Qwen2-VL model, optionally attaching trained LoRA weights.

    Args:
        config: InferenceConfig with model settings.
        use_lora: If True, attach LoRA adapter via PeftModel.from_pretrained().
                  Must be False for Configuration A (baseline).

    Returns:
        Tuple of (model, processor) ready for .generate() or .forward().

    Raises:
        FileNotFoundError: If use_lora=True and LoRA path is invalid.
        RuntimeError: If model loading fails.
    """
    from src.models.qwen_vl import QwenVLQuantizer

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info(
        f"Loading model: {config.base_model} "
        f"(quantization={config.quantization_bits}-bit, use_lora={use_lora})"
    )

    quantizer = QwenVLQuantizer(
        model_name=config.base_model,
        quantization_bits=config.quantization_bits,
        device_map=config.device_map,
    )
    model, processor = quantizer.load()

    if use_lora:
        config.validate_lora_path()
        lora_resolved = str(Path(config.lora_path).resolve())
        logger.info(f"Attaching LoRA adapter from: {lora_resolved}")

        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_resolved)
            model.eval()
            logger.info("LoRA adapter attached successfully.")
        except Exception as e:
            logger.error(f"Failed to attach LoRA adapter: {e}")
            raise RuntimeError(f"LoRA loading failed: {e}") from e
    else:
        model.eval()
        logger.info("Loaded base model without LoRA (Configuration A baseline).")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"Post-load GPU memory: {mem_gb:.2f} GB")

    return model, processor


def get_peak_gpu_memory_gb() -> float:
    """Return current peak GPU memory allocated in GB.

    Returns:
        Float GB value, or 0.0 if CUDA not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def reset_peak_gpu_memory() -> None:
    """Reset PyTorch peak GPU memory tracker."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
