"""Unified inference pipeline supporting Configurations A, B, and C.

Orchestrates three configurations required by the SRS Connector Bottleneck
Analysis (Section 2.3):

    Mode A — Baseline:
        Base model (no LoRA), greedy decoding, no RAG, no VCD.

    Mode B — QLoRA Fine-Tuned:
        LoRA adapter attached, greedy decoding, no RAG, no VCD.

    Mode C — Proposed System:
        LoRA adapter + Uncertainty-Triggered RAG + Visual Contrastive Decoding.

The three configurations are exactly comparable (same images, same questions,
same seed) so Module 4 can attribute differences purely to the technique.

Example:
    >>> from src.inference import InferencePipeline, InferenceConfig
    >>> from src.data import FAISSRetriever
    >>> config = InferenceConfig(
    ...     lora_path="checkpoints/lora_weights",
    ...     faiss_index_path="data/faiss_index",
    ... )
    >>> pipeline = InferencePipeline(config)
    >>> result = pipeline.run(image, "What is on the table?", mode="C")
    >>> print(result.answer, result.rag_triggered, result.latency_seconds)
"""

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from src.inference.rag import UncertaintyTriggeredRAG
from src.inference.utils import (
    InferenceConfig,
    InferenceResult,
    get_peak_gpu_memory_gb,
    load_inference_model,
    reset_peak_gpu_memory,
)
from src.models.vcd import VisualContrastiveDecoder

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
VALID_MODES = ("A", "B", "C")


class InferencePipeline:
    """Unified inference pipeline for configurations A, B, and C.

    Loads and caches models on first use. Mode switching is handled by a single
    run() method — no separate code paths that could drift.

    Args:
        config: InferenceConfig with all hyperparameters.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Lazy-initialized model caches
        self._model_a: Optional[Any] = None       # Base model, no LoRA
        self._model_bc: Optional[Any] = None      # LoRA model for B and C
        self._processor: Optional[Any] = None     # Shared processor

        # Lazy-initialized components
        self._rag: Optional[UncertaintyTriggeredRAG] = None
        self._vcd: Optional[VisualContrastiveDecoder] = None

        logger.info(f"InferencePipeline initialized (seed={config.seed})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        question: str,
        mode: str,
    ) -> InferenceResult:
        """Run inference for a single image+question pair.

        Args:
            image: PIL Image (RGB).
            question: The VQA question string.
            mode: "A" (baseline), "B" (QLoRA only), or "C" (QLoRA + RAG + VCD).

        Returns:
            InferenceResult with answer, latency, memory, and grounding metadata.

        Raises:
            ValueError: If mode is not one of "A", "B", "C".
        """
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}.")

        reset_peak_gpu_memory()
        start_time = time.perf_counter()

        rag_triggered = False
        evidence_used: Optional[str] = None
        entropy_score: Optional[float] = None
        answer = ""

        try:
            if mode == "A":
                answer = self._run_mode_a(image, question)

            elif mode == "B":
                answer = self._run_mode_b(image, question)

            elif mode == "C":
                answer, rag_triggered, evidence_used, entropy_score = self._run_mode_c(
                    image, question
                )

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM during mode {mode} inference: {e}. Clearing cache.")
            torch.cuda.empty_cache()
            answer = "[OOM]"

        except Exception as e:
            logger.error(f"Inference error (mode={mode}): {e}\n{traceback.format_exc()}")
            answer = "[ERROR]"

        latency = time.perf_counter() - start_time
        gpu_memory = get_peak_gpu_memory_gb()

        logger.info(
            f"mode={mode} | latency={latency:.3f}s | memory={gpu_memory:.2f}GB "
            f"| rag={rag_triggered} | answer='{answer[:50]}'"
        )

        return InferenceResult(
            answer=answer,
            latency_seconds=latency,
            gpu_memory_gb=gpu_memory,
            rag_triggered=rag_triggered,
            evidence_used=evidence_used,
            entropy_score=entropy_score,
            mode=mode,
        )

    def run_batch(
        self,
        images: List[Image.Image],
        questions: List[str],
        mode: str,
        checkpoint_file: Optional[str] = None,
        question_ids: Optional[List[int]] = None,
        image_paths: Optional[List[str]] = None,
    ) -> List[InferenceResult]:
        """Run batch inference over a dataset with checkpointing support.

        Resumes from a partial results file if checkpoint_file exists and
        contains already-processed records.

        Args:
            images: List of PIL Images.
            questions: List of question strings (same length as images).
            mode: "A", "B", or "C".
            checkpoint_file: Optional path to save/resume partial results (JSON).
            question_ids: Optional list of question IDs for traceability.
            image_paths: Optional list of image paths for traceability.

        Returns:
            List of InferenceResult for each image+question pair.
        """
        if len(images) != len(questions):
            raise ValueError(
                f"images ({len(images)}) and questions ({len(questions)}) must have the same length."
            )

        results: List[InferenceResult] = []
        start_idx = 0

        # Resume from checkpoint if available
        if checkpoint_file and Path(checkpoint_file).exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                results = [
                    InferenceResult(**{k: v for k, v in r.items() if k in InferenceResult.__dataclass_fields__})
                    for r in saved
                ]
                start_idx = len(results)
                logger.info(f"Resumed batch from checkpoint: {start_idx} results loaded.")
            except Exception as e:
                logger.warning(f"Could not load checkpoint {checkpoint_file}: {e}. Starting fresh.")
                results = []
                start_idx = 0

        n = len(images)
        rag_triggered_count = 0

        for i in range(start_idx, n):
            img = images[i]
            q = questions[i]
            qid = question_ids[i] if question_ids else None
            img_path = image_paths[i] if image_paths else None

            try:
                result = self.run(img, q, mode=mode)
                result.question_id = qid
                result.image_path = img_path
                results.append(result)

                if result.rag_triggered:
                    rag_triggered_count += 1

            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM at item {i}/{n}. Clearing cache and retrying.")
                torch.cuda.empty_cache()
                try:
                    result = self.run(img, q, mode=mode)
                    result.question_id = qid
                    result.image_path = img_path
                    results.append(result)
                except Exception as retry_err:
                    logger.error(f"Retry failed for item {i}: {retry_err}")
                    results.append(
                        InferenceResult(
                            answer="[OOM]",
                            latency_seconds=0.0,
                            gpu_memory_gb=0.0,
                            rag_triggered=False,
                            evidence_used=None,
                            entropy_score=None,
                            mode=mode,
                            question_id=qid,
                            image_path=img_path,
                        )
                    )

            except Exception as e:
                logger.error(f"Unexpected error at item {i}: {e}")
                results.append(
                    InferenceResult(
                        answer="[ERROR]",
                        latency_seconds=0.0,
                        gpu_memory_gb=0.0,
                        rag_triggered=False,
                        evidence_used=None,
                        entropy_score=None,
                        mode=mode,
                        question_id=qid,
                        image_path=img_path,
                    )
                )

            # Checkpoint every 50 items
            if checkpoint_file and (i + 1) % 50 == 0:
                self._save_checkpoint(results, checkpoint_file)
                logger.info(f"Checkpoint saved at item {i + 1}/{n}")

        # Final checkpoint
        if checkpoint_file:
            self._save_checkpoint(results, checkpoint_file)

        # Batch summary
        completed = len(results)
        valid_latencies = [r.latency_seconds for r in results if r.latency_seconds > 0]
        avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0.0
        avg_memory = sum(r.gpu_memory_gb for r in results) / completed if completed else 0.0
        rag_rate = rag_triggered_count / completed if completed else 0.0

        logger.info(
            f"Batch complete (mode={mode}): {completed}/{n} items | "
            f"avg_latency={avg_latency:.2f}s | avg_memory={avg_memory:.2f}GB | "
            f"RAG activation={rag_rate:.1%}"
        )

        # Log RAG activation rate if mode C
        if mode == "C" and self._rag is not None:
            self._rag.log_activation_summary()

        return results

    # ------------------------------------------------------------------
    # Mode-specific runners
    # ------------------------------------------------------------------

    def _run_mode_a(self, image: Image.Image, question: str) -> str:
        """Mode A: Base model, greedy decode, no RAG, no VCD."""
        model, processor = self._get_model_a()
        return self._greedy_generate(model, processor, image, question)

    def _run_mode_b(self, image: Image.Image, question: str) -> str:
        """Mode B: LoRA model, greedy decode, no RAG, no VCD."""
        model, processor = self._get_model_bc()
        return self._greedy_generate(model, processor, image, question)

    def _run_mode_c(
        self, image: Image.Image, question: str
    ) -> Tuple[str, bool, Optional[str], float]:
        """Mode C: LoRA model + RAG + VCD.

        Returns:
            Tuple of (answer, rag_triggered, evidence_used, entropy_score).
        """
        model, processor = self._get_model_bc()
        rag = self._get_rag(model, processor)
        vcd = self._get_vcd(model, processor)

        # Step 1: Check entropy and possibly retrieve evidence
        rag_result = rag.check_and_retrieve(image, question)
        rag_triggered: bool = rag_result["triggered"]
        evidence: Optional[str] = rag_result["evidence"]
        entropy_score: float = rag_result["entropy_score"]

        # Step 2: VCD token-by-token generation
        try:
            answer, vcd_overhead = vcd.generate(image, question, evidence=evidence)
        except Exception as e:
            logger.warning(
                f"VCD generation failed: {e}. Falling back to greedy decode (mode B behavior)."
            )
            answer = self._greedy_generate(model, processor, image, question, evidence=evidence)
            vcd_overhead = 0.0

        return answer, rag_triggered, evidence, entropy_score

    # ------------------------------------------------------------------
    # Greedy generation helper
    # ------------------------------------------------------------------

    def _greedy_generate(
        self,
        model: Any,
        processor: Any,
        image: Image.Image,
        question: str,
        evidence: Optional[str] = None,
    ) -> str:
        """Standard greedy decode using model.generate().

        Args:
            model: Language model.
            processor: Tokenizer/processor.
            image: PIL Image.
            question: Question string.
            evidence: Optional RAG evidence to prepend.

        Returns:
            Generated answer string.
        """
        system_prompt = (
            "You are a careful visual assistant. "
            "Answer only based on what is clearly visible in the image. "
            "If the answer is not visible or unsupported, say 'No' or 'Not visible'."
        )

        user_content = []
        if evidence:
            user_content.append({"type": "text", "text": f"Context:\n{evidence}"})
        user_content.append({"type": "image", "image": image})

        q_text = question
        if self._is_yes_no_question(question):
            q_text += " Answer with only 'yes' or 'no'."
        user_content.append({"type": "text", "text": q_text})

        device = self._get_device(model)

        try:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": user_content},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[image], return_tensors="pt"
            )
        except Exception:
            prompt = f"{evidence}\n\n{question}" if evidence else question
            inputs = processor(text=[prompt], return_tensors="pt")
            inputs = processor(text=[prompt], return_tensors="pt")

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][input_len:]
        tokenizer = getattr(processor, "tokenizer", processor)
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Lazy model getters
    # ------------------------------------------------------------------

    def _get_model_a(self) -> Tuple[Any, Any]:
        """Return (base_model, processor), loading once."""
        if self._model_a is None:
            logger.info("Loading Model A (base, no LoRA)...")
            self._model_a, self._processor = load_inference_model(
                self.config, use_lora=False
            )
        return self._model_a, self._processor

    def _get_model_bc(self) -> Tuple[Any, Any]:
        """Return (lora_model, processor), loading once."""
        if self._model_bc is None:
            logger.info("Loading Model B/C (LoRA attached)...")
            self._model_bc, proc = load_inference_model(
                self.config, use_lora=True
            )
            if self._processor is None:
                self._processor = proc
        return self._model_bc, self._processor

    def _get_rag(
        self, model: Any, processor: Any
    ) -> UncertaintyTriggeredRAG:
        """Return UncertaintyTriggeredRAG, initializing once."""
        if self._rag is None:
            retriever = self._get_retriever()
            self._rag = UncertaintyTriggeredRAG(
                model=model,
                processor=processor,
                retriever=retriever,
                config=self.config,
            )
            logger.info("UncertaintyTriggeredRAG initialized.")
        return self._rag

    def _get_vcd(
        self, model: Any, processor: Any
    ) -> VisualContrastiveDecoder:
        """Return VisualContrastiveDecoder, initializing once."""
        if self._vcd is None:
            self._vcd = VisualContrastiveDecoder(
                model=model,
                processor=processor,
                alpha=self.config.vcd_alpha,
                blur_radius=self.config.blur_radius,
                max_new_tokens=self.config.max_new_tokens,
                seed=self.config.seed,
            )
            logger.info("VisualContrastiveDecoder initialized.")
        return self._vcd

    def _get_retriever(self) -> Any:
        """Return FAISSRetriever, using injected instance or loading from path."""
        if self.config.faiss_retriever is not None:
            return self.config.faiss_retriever

        from src.data.retrieval import FAISSRetriever
        logger.info(f"Loading FAISSRetriever from: {self.config.faiss_index_path}")
        return FAISSRetriever(self.config.faiss_index_path)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_device(model: Any) -> str:
        """Resolve model device as a string ('cuda', 'cpu')."""
        try:
            p = next(model.parameters())
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

    @staticmethod
    def _is_yes_no_question(question: str) -> bool:
        """Detect POPE-style yes/no object-presence questions.

        Matches questions starting with interrogative verbs that typically
        expect a binary answer (e.g. 'Is there a cat?', 'Does this image
        contain a dog?', 'Are there any chairs?').

        Returns:
            True if the question expects a yes/no answer.
        """
        import re
        q = question.strip().lower()
        yes_no_patterns = [
            r"^is there\b",
            r"^are there\b",
            r"^does (the|this|an?)? ?image\b",
            r"^do (the|these)? ?images\b",
            r"^can you (see|spot|find)\b",
            r"^is (a|an|the)\b",
            r"^are (the|any)\b",
        ]
        return any(re.search(p, q) for p in yes_no_patterns)

    @staticmethod
    def _save_checkpoint(results: List[InferenceResult], path: str) -> None:
        """Persist results list to JSON checkpoint file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
