"""Evaluation metrics: VQA Accuracy, CHAIR Score, POPE F1, and Performance Aggregation.

Implements the 3 core metrics specified in SRS Section 3 & 4:
  1. VQA Accuracy: Official VQAv2 scoring formula min(#annotators_agreeing / 3, 1).
  2. CHAIR Score: Object hallucination rate (CHAIR_i instance-level, CHAIR_s sentence-level).
  3. POPE F1: Precision, Recall, and F1 over yes/no object-presence questions across random, popular, adversarial modes.
  4. Performance Aggregation: Latency stats (mean, p50, p95) and peak GPU memory.

Example:
    >>> from src.evaluation.metrics import compute_vqa_accuracy, compute_chair_score, compute_pope_f1
    >>> acc = compute_vqa_accuracy(["cat", "dog"], [["cat", "cat", "cat"], ["dog", "cat"]])
    >>> chair = compute_chair_score(["a cat on a chair"], [["cat"]])
    >>> pope = compute_pope_f1(["yes", "no"], ["yes", "yes"])
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.evaluation.utils import (
    MSCOCO_80_CLASSES,
    MSCOCO_SYNONYMS,
    extract_object_mentions,
    extract_yes_no,
    normalize_answer,
)
from src.inference.utils import InferenceResult

logger = logging.getLogger(__name__)


def compute_vqa_accuracy(
    predictions: List[str],
    ground_truth_annotations: List[Union[List[str], Dict[str, Any]]],
) -> float:
    """Compute official VQAv2 accuracy percentage.

    Formula:
        acc(ans) = min(count(ans in human_annotators) / 3.0, 1.0)
        final_accuracy = mean(acc(ans_i)) * 100.0

    Args:
        predictions: List of predicted answer strings.
        ground_truth_annotations: List of human answer lists (e.g. [["cat", "cat", "dog"], ...])
                                  or dict objects containing an 'answers' list.

    Returns:
        Float VQA accuracy score in percentage range [0.0, 100.0].
    """
    if not predictions or len(predictions) != len(ground_truth_annotations):
        logger.warning(
            f"VQA accuracy compute length mismatch or empty: "
            f"preds={len(predictions)}, gt={len(ground_truth_annotations)}"
        )
        return 0.0

    scores: List[float] = []

    for pred, gt in zip(predictions, ground_truth_annotations):
        norm_pred = normalize_answer(pred)

        # Extract human answer strings
        if isinstance(gt, dict):
            raw_answers = gt.get("answers", [])
        elif isinstance(gt, list):
            raw_answers = gt
        else:
            raw_answers = [str(gt)]

        norm_gt_answers = [normalize_answer(a) for a in raw_answers if a]

        if not norm_gt_answers:
            continue

        # Count matching annotators
        matching_count = sum(1 for a in norm_gt_answers if a == norm_pred)
        score = min(1.0, matching_count / 3.0)
        scores.append(score)

    if not scores:
        return 0.0

    accuracy_pct = float(np.mean(scores) * 100.0)
    logger.info(f"Computed VQA Accuracy: {accuracy_pct:.2f}% over {len(scores)} questions")
    return accuracy_pct


def compute_chair_score(
    predictions: List[str],
    ground_truth_objects: List[List[str]],
    coco_classes: Optional[List[str]] = None,
    synonyms: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Compute CHAIR (Caption Hallucination Assessment with Image Relevance) metrics.

    CHAIR_i (Instance-level):
        CHAIR_i = (count of hallucinated object mentions) / (total count of object mentions)

    CHAIR_s (Sentence-level):
        CHAIR_s = (count of sentences with >=1 hallucinated object) / (total count of sentences)

    Args:
        predictions: Generated answer texts.
        ground_truth_objects: List of lists containing ground-truth COCO objects present per image.
        coco_classes: Optional COCO vocabulary.
        synonyms: Optional synonym map.

    Returns:
        Dict with keys:
            - chair_i (float): Instance-level hallucination rate in [0.0, 1.0].
            - chair_s (float): Sentence-level hallucination rate in [0.0, 1.0].
            - total_objects_mentioned (int): Total count of detected object mentions.
            - total_hallucinated_objects (int): Total count of hallucinated object mentions.
    """
    if not predictions or len(predictions) != len(ground_truth_objects):
        logger.warning(
            f"CHAIR score compute length mismatch or empty: "
            f"preds={len(predictions)}, gt={len(ground_truth_objects)}"
        )
        return {
            "chair_i": 0.0,
            "chair_s": 0.0,
            "total_objects_mentioned": 0,
            "total_hallucinated_objects": 0,
        }

    total_objects_mentioned = 0
    total_hallucinated_objects = 0
    sentences_with_hallucination = 0
    valid_sentences = 0

    for pred, gt_objs in zip(predictions, ground_truth_objects):
        valid_sentences += 1
        # Normalize ground truth object names
        norm_gt = set()
        for o in gt_objs:
            norm_o = normalize_answer(o)
            mapped_o = MSCOCO_SYNONYMS.get(norm_o, norm_o)
            norm_gt.add(mapped_o)

        # Extract objects mentioned in candidate answer
        mentioned = extract_object_mentions(pred, coco_classes, synonyms)
        total_objects_mentioned += len(mentioned)

        has_hallucination = False
        for obj in mentioned:
            if obj not in norm_gt:
                total_hallucinated_objects += 1
                has_hallucination = True

        if has_hallucination:
            sentences_with_hallucination += 1

    chair_i = (
        total_hallucinated_objects / total_objects_mentioned
        if total_objects_mentioned > 0
        else 0.0
    )
    chair_s = (
        sentences_with_hallucination / valid_sentences
        if valid_sentences > 0
        else 0.0
    )

    logger.info(
        f"Computed CHAIR Scores: CHAIR_i={chair_i:.4f}, CHAIR_s={chair_s:.4f} "
        f"(hallucinated {total_hallucinated_objects}/{total_objects_mentioned} mentions across {valid_sentences} sentences)"
    )

    return {
        "chair_i": float(chair_i),
        "chair_s": float(chair_s),
        "total_objects_mentioned": total_objects_mentioned,
        "total_hallucinated_objects": total_hallucinated_objects,
    }


def compute_pope_f1(
    predictions: List[str],
    ground_truth_labels: List[str],
) -> Dict[str, float]:
    """Compute POPE F1, Precision, and Recall scores for yes/no object existence VQA.

    Args:
        predictions: Model predictions (raw strings or yes/no).
        ground_truth_labels: Target yes/no strings ("yes" or "no").

    Returns:
        Dict with keys:
            - precision (float): Precision score in [0.0, 1.0].
            - recall (float): Recall score in [0.0, 1.0].
            - f1 (float): F1 score in [0.0, 1.0].
            - accuracy (float): Accuracy in [0.0, 1.0].
            - tp, fp, tn, fn (int): Confusion matrix counts.
    """
    if not predictions or len(predictions) != len(ground_truth_labels):
        logger.warning(
            f"POPE F1 compute length mismatch or empty: "
            f"preds={len(predictions)}, gt={len(ground_truth_labels)}"
        )
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        }

    tp = fp = tn = fn = 0

    for pred, gt in zip(predictions, ground_truth_labels):
        pred_yn = extract_yes_no(pred)
        gt_yn = normalize_answer(gt)

        if gt_yn not in ("yes", "no"):
            gt_yn = "yes" if "yes" in gt_yn else "no"

        # If model answer is ambiguous, treat as incorrect
        if pred_yn is None:
            if gt_yn == "yes":
                fn += 1
            else:
                fp += 1
            continue

        if pred_yn == "yes" and gt_yn == "yes":
            tp += 1
        elif pred_yn == "yes" and gt_yn == "no":
            fp += 1
        elif pred_yn == "no" and gt_yn == "no":
            tn += 1
        elif pred_yn == "no" and gt_yn == "yes":
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    logger.info(
        f"Computed POPE F1: F1={f1:.4f}, Precision={precision:.4f}, "
        f"Recall={recall:.4f}, Acc={accuracy:.4f} (TP={tp}, FP={fp}, TN={tn}, FN={fn})"
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def aggregate_performance_metrics(
    inference_results: List[InferenceResult],
) -> Dict[str, float]:
    """Aggregate inference latency and GPU memory usage from InferenceResult list.

    Args:
        inference_results: List of InferenceResult objects.

    Returns:
        Dict with keys:
            - avg_latency (float): Mean wall-clock time per image (seconds).
            - p50_latency (float): Median latency (seconds).
            - p95_latency (float): 95th percentile latency (seconds).
            - avg_memory_gb (float): Mean peak GPU memory (GB).
            - peak_memory_gb (float): Maximum peak GPU memory recorded (GB).
            - rag_activation_rate (float): Fraction of queries that triggered RAG.
    """
    if not inference_results:
        return {
            "avg_latency": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "avg_memory_gb": 0.0,
            "peak_memory_gb": 0.0,
            "rag_activation_rate": 0.0,
        }

    valid_latencies = [
        r.latency_seconds for r in inference_results if r.latency_seconds > 0
    ]
    memories = [r.gpu_memory_gb for r in inference_results]
    rag_triggers = [1.0 if r.rag_triggered else 0.0 for r in inference_results]

    if valid_latencies:
        avg_lat = float(np.mean(valid_latencies))
        p50_lat = float(np.percentile(valid_latencies, 50))
        p95_lat = float(np.percentile(valid_latencies, 95))
    else:
        avg_lat = p50_lat = p95_lat = 0.0

    avg_mem = float(np.mean(memories)) if memories else 0.0
    peak_mem = float(np.max(memories)) if memories else 0.0
    rag_rate = float(np.mean(rag_triggers)) if rag_triggers else 0.0

    return {
        "avg_latency": round(avg_lat, 3),
        "p50_latency": round(p50_lat, 3),
        "p95_latency": round(p95_lat, 3),
        "avg_memory_gb": round(avg_mem, 2),
        "peak_memory_gb": round(peak_mem, 2),
        "rag_activation_rate": round(rag_rate, 4),
    }
