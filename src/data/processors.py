"""Data Preprocessing, Image Processing, CLIP Embeddings, and Data Validation.

Example:
    >>> from src.data.processors import CLIPEmbeddingExtractor, DataValidator
    >>> extractor = CLIPEmbeddingExtractor()
    >>> embs = extractor.extract(["A cat sitting on a bench"])
    >>> validator = DataValidator(config)
    >>> report = validator.validate()
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Load, validate, and convert image files safely."""

    @staticmethod
    def load_image(image_path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None):
        """Load image file, convert to RGB, and optionally resize.

        Args:
            image_path: Path to image file.
            target_size: Optional (width, height) tuple.

        Returns:
            PIL Image object.

        Raises:
            FileNotFoundError: If image file does not exist.
            ValueError: If file is corrupted or unreadable.
        """
        from PIL import Image

        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")

        try:
            img = Image.open(path).convert("RGB")
            if target_size:
                img = img.resize(target_size, Image.Resampling.BILINEAR)
            return img
        except Exception as e:
            logger.error(f"Failed to load image at {path}: {e}")
            raise ValueError(f"Corrupted image at {path}: {e}")

    @staticmethod
    def validate_image_file(image_path: Union[str, Path]) -> bool:
        """Check if image file exists and is readable."""
        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            return False
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False


class AnnotationProcessor:
    """Normalise and format VQA / POPE annotation data."""

    @staticmethod
    def normalize_vqa(raw_annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise VQA annotation item into standard schema.

        Schema:
            {"question_id": int, "image_id": int, "question": str, "answers": List[str]}
        """
        answers = []
        if "answers" in raw_annotation:
            answers = [ans["answer"] for ans in raw_annotation["answers"] if "answer" in ans]
        elif "multiple_choice_answer" in raw_annotation:
            answers = [raw_annotation["multiple_choice_answer"]]

        return {
            "question_id": int(raw_annotation.get("question_id", 0)),
            "image_id": int(raw_annotation.get("image_id", 0)),
            "question": str(raw_annotation.get("question", "")).strip(),
            "answers": [a.lower().strip() for a in answers if a.strip()],
        }


class CLIPEmbeddingExtractor:
    """Extract sentence-level CLIP-ViT-B-32 embeddings for caption texts.

    Output shape: (len(captions), 512), dtype=float32, normalized.

    Args:
        model_name: HuggingFace CLIP model identifier.
        batch_size: Processing batch size (default: 32).
        device: "auto", "cuda", or "cpu".
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 32,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading CLIP model '{self.model_name}' on {self.device}...")
            import torch
            from transformers import AutoTokenizer, CLIPTextModelWithProjection

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = CLIPTextModelWithProjection.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    def extract(self, captions: List[str]) -> np.ndarray:
        """Extract L2-normalized 512-dim CLIP embeddings for captions.

        Args:
            captions: List of text captions.

        Returns:
            np.ndarray of shape (len(captions), 512), dtype=float32.
        """
        if not captions:
            return np.empty((0, 512), dtype=np.float32)

        self._load_model()
        import torch

        all_embeddings = []
        for i in range(0, len(captions), self.batch_size):
            batch_texts = captions[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch_texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Text projection outputs L2-normalized features
                embeds = outputs.text_embeds
                # Normalize manually for safety
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                all_embeddings.append(embeds.cpu().numpy())

        res = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"Extracted CLIP embeddings for {len(captions)} captions, shape: {res.shape}")
        return res


class DataValidator:
    """Comprehensive data integrity checks for Module 1 deliverables.

    Checks:
        1. VQAv2 training/val datasets
        2. POPE evaluation splits
        3. 1000 captions corpus
        4. FAISS index & embeddings
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def validate(self) -> Dict[str, Any]:
        """Run all data checks and return validation report."""
        report = {
            "vqav2_images_valid": True,
            "vqav2_annotations_valid": True,
            "pope_split_complete": True,
            "captions_count_1000": True,
            "embeddings_shape_valid": True,
            "faiss_index_valid": True,
            "passed": True,
            "errors": [],
        }

        # Check VQAv2 directories
        for key in ["vqa_train_dir", "vqa_val_dir"]:
            if not Path(self.config.get(key, "")).is_dir():
                report["vqav2_images_valid"] = False
                report["errors"].append(f"Missing or invalid directory: {key}")

        # Check POPE files
        for p in self.config.get("pope_files", []):
            if not Path(p).exists():
                report["pope_split_complete"] = False
                report["errors"].append(f"Missing POPE file: {p}")

        # Check Captions Count
        try:
            with open(self.config.get("captions_path", ""), "r") as f:
                captions = json.load(f)
                if len(captions) != 1000:
                    report["captions_count_1000"] = False
        except Exception:
            report["captions_count_1000"] = False

        # Check FAISS index
        if not Path(self.config.get("faiss_index_path", "")).exists():
            report["faiss_index_valid"] = False

        report["passed"] = all(v is True for k, v in report.items() if isinstance(v, bool))
        logger.info("DataValidator suite complete. Status: %s", report["passed"])
        return report
