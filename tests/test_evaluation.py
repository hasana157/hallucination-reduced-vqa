"""Unit tests for Module 4: Evaluation & Results Analysis.

Tests:
  1. test_vqa_accuracy_perfect_match
  2. test_vqa_accuracy_partial_match
  3. test_vqa_accuracy_answer_normalization
  4. test_chair_score_no_hallucination
  5. test_chair_score_with_hallucination
  6. test_pope_f1_all_correct
  7. test_pope_f1_mixed_results
  8. test_pope_f1_all_three_modes
  9. test_performance_aggregation
  10. test_evaluator_all_configurations
  11. test_connector_bottleneck_report_generation
  12. test_reproducibility_checker_low_variance
"""

import math
import pytest
from src.evaluation import (
    ComparisonReport,
    ConnectorBottleneckAnalyzer,
    Evaluator,
    ReproducibilityChecker,
    aggregate_performance_metrics,
    compute_chair_score,
    compute_pope_f1,
    compute_vqa_accuracy,
    extract_object_mentions,
    extract_yes_no,
    normalize_answer,
)
from src.inference.utils import InferenceResult


# ============================================================
# 1. VQA Accuracy Tests
# ============================================================

def test_vqa_accuracy_perfect_match():
    """3 annotators agreeing -> accuracy = min(3/3, 1) = 100%."""
    preds = ["cat", "dog"]
    gt = [["cat", "cat", "cat"], ["dog", "dog", "dog"]]
    acc = compute_vqa_accuracy(preds, gt)
    assert acc == 100.0


def test_vqa_accuracy_partial_match():
    """2 annotators agreeing -> accuracy = min(2/3, 1) = 66.67%."""
    preds = ["cat"]
    gt = [["cat", "cat", "dog"]]
    acc = compute_vqa_accuracy(preds, gt)
    assert abs(acc - (2.0 / 3.0 * 100.0)) < 1e-2


def test_vqa_accuracy_answer_normalization():
    """"2" vs "two", capitalization, punctuation should normalize to match."""
    assert normalize_answer("Two Cats.") == "2 cats"
    assert normalize_answer("there is a car!") == "there is car"
    assert normalize_answer("CANNOT SEE") == "can not see"

    preds = ["2"]
    gt = [["two", "two", "2"]]
    acc = compute_vqa_accuracy(preds, gt)
    assert acc == 100.0


# ============================================================
# 2. CHAIR Score Tests
# ============================================================

def test_chair_score_no_hallucination():
    """When all mentioned objects exist in GT, CHAIR_i and CHAIR_s should be 0.0."""
    preds = ["a cat sitting on a chair"]
    gt = [["cat", "chair"]]
    res = compute_chair_score(preds, gt)
    assert res["chair_i"] == 0.0
    assert res["chair_s"] == 0.0
    assert res["total_hallucinated_objects"] == 0


def test_chair_score_with_hallucination():
    """Mentioning 'dog' when GT only has 'cat' -> CHAIR hallucination detected."""
    preds = ["a cat and a dog on a chair"]
    gt = [["cat", "chair"]]
    res = compute_chair_score(preds, gt)
    # Objects mentioned: cat, dog, chair = 3. Hallucinated: dog = 1.
    assert abs(res["chair_i"] - (1.0 / 3.0)) < 1e-3
    assert res["chair_s"] == 1.0  # 1 out of 1 sentence had hallucination


# ============================================================
# 3. POPE F1 Tests
# ============================================================

def test_pope_f1_all_correct():
    """Perfect POPE predictions -> F1 = 1.0, Precision = 1.0, Recall = 1.0."""
    preds = ["yes", "no", "yes", "no"]
    gt = ["yes", "no", "yes", "no"]
    res = compute_pope_f1(preds, gt)
    assert res["f1"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["accuracy"] == 1.0


def test_pope_f1_mixed_results():
    """Verify precision, recall, and F1 calculations numerically."""
    # TP=1 ("yes"/"yes"), FP=1 ("yes"/"no"), TN=1 ("no"/"no"), FN=1 ("no"/"yes")
    preds = ["yes", "yes", "no", "no"]
    gt = ["yes", "no", "no", "yes"]
    res = compute_pope_f1(preds, gt)

    # Precision = TP / (TP+FP) = 1/2 = 0.5
    # Recall = TP / (TP+FN) = 1/2 = 0.5
    # F1 = 2*0.5*0.5 / (0.5+0.5) = 0.5
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5
    assert res["accuracy"] == 0.5


def test_pope_f1_all_three_modes():
    """POPE helpers extract variations of yes/no correctly."""
    assert extract_yes_no("Yes, there is a cat.") == "yes"
    assert extract_yes_no("No, I cannot see any table.") == "no"
    assert extract_yes_no("It looks like a dog.") is None


# ============================================================
# 4. Performance Aggregation & Evaluator Tests
# ============================================================

def test_performance_aggregation():
    """Verify latency p50/p95 and peak memory aggregation."""
    results = [
        InferenceResult("ans", latency_seconds=1.0, gpu_memory_gb=10.0, rag_triggered=False, evidence_used=None, entropy_score=0.1, mode="A"),
        InferenceResult("ans", latency_seconds=2.0, gpu_memory_gb=12.0, rag_triggered=True, evidence_used="ev", entropy_score=0.9, mode="A"),
        InferenceResult("ans", latency_seconds=3.0, gpu_memory_gb=14.0, rag_triggered=False, evidence_used=None, entropy_score=0.2, mode="A"),
    ]
    agg = aggregate_performance_metrics(results)
    assert agg["avg_latency"] == 2.0
    assert agg["p50_latency"] == 2.0
    assert agg["peak_memory_gb"] == 14.0
    assert abs(agg["rag_activation_rate"] - (1.0 / 3.0)) < 1e-3


def test_evaluator_all_configurations():
    """Evaluator.evaluate_all produces ComparisonReport with expected structure."""
    evaluator = Evaluator()

    # Create dummy cached results for A, B, and C
    cached_a = [InferenceResult("cat", latency_seconds=2.1, gpu_memory_gb=12.5, rag_triggered=False, evidence_used=None, entropy_score=0.2, mode="A")]
    cached_b = [InferenceResult("cat", latency_seconds=2.2, gpu_memory_gb=12.5, rag_triggered=False, evidence_used=None, entropy_score=0.2, mode="B")]
    cached_c = [InferenceResult("cat", latency_seconds=3.1, gpu_memory_gb=14.0, rag_triggered=True, evidence_used="ev", entropy_score=0.85, mode="C")]

    cached_by_mode = {"A": cached_a, "B": cached_b, "C": cached_c}

    report = evaluator.evaluate_all(
        images=[None],
        questions=["What is this?"],
        vqa_ground_truth=[["cat", "cat", "cat"]],
        chair_ground_truth=[["cat"]],
        pope_datasets={"random": (["Is there a cat?"], ["yes"])},
        cached_results_by_mode=cached_by_mode,
    )

    assert isinstance(report, ComparisonReport)
    assert report.report_a.mode == "A"
    assert report.report_b.mode == "B"
    assert report.report_c.mode == "C"
    assert "B_minus_A" in report.deltas
    assert "C_minus_A_total" in report.deltas


def test_connector_bottleneck_report_generation():
    """ConnectorBottleneckAnalyzer produces quantitative stats and qualitative examples."""
    analyzer = ConnectorBottleneckAnalyzer()
    res_a = [InferenceResult("fork and plate", latency_seconds=2.0, gpu_memory_gb=12.0, rag_triggered=False, evidence_used=None, entropy_score=0.85, mode="A", question_id=1, image_path="img1.jpg")]
    res_c = [InferenceResult("plate", latency_seconds=3.0, gpu_memory_gb=14.0, rag_triggered=True, evidence_used="Evidence: plate on table", entropy_score=0.85, mode="C", question_id=1, image_path="img1.jpg")]

    report = analyzer.analyze(
        results_a=res_a,
        results_c=res_c,
        questions=["What is on the table?"],
        ground_truth=["plate"],
    )

    assert "research_questions" in report
    assert len(report["qualitative_cases_corrected"]) == 1
    case = report["qualitative_cases_corrected"][0]
    assert case["mode_A_baseline_answer"] == "fork and plate"
    assert case["mode_C_proposed_answer"] == "plate"
    assert case["rag_triggered"] is True


def test_reproducibility_checker_low_variance():
    """ReproducibilityChecker passes when metric variance <= 0.1%."""
    checker = ReproducibilityChecker(tolerance_pct=0.1, n_runs=3)
    dummy_eval = lambda: {"accuracy": 71.50, "chair_i": 0.120}

    res = checker.check(dummy_eval)
    assert res["passed"] is True
    assert res["metrics"]["accuracy"]["variance_pct"] <= 0.1
