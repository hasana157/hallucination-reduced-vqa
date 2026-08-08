"""Evaluation module: metrics, evaluator, and bottleneck analysis."""

from src.evaluation.evaluator import (
    ComparisonReport,
    ConfigurationReport,
    ConnectorBottleneckAnalyzer,
    Evaluator,
)
from src.evaluation.metrics import (
    aggregate_performance_metrics,
    compute_chair_score,
    compute_pope_f1,
    compute_vqa_accuracy,
)
from src.evaluation.utils import (
    ReproducibilityChecker,
    extract_object_mentions,
    extract_yes_no,
    normalize_answer,
)

__all__ = [
    "Evaluator",
    "ConfigurationReport",
    "ComparisonReport",
    "ConnectorBottleneckAnalyzer",
    "compute_vqa_accuracy",
    "compute_chair_score",
    "compute_pope_f1",
    "aggregate_performance_metrics",
    "normalize_answer",
    "extract_yes_no",
    "extract_object_mentions",
    "ReproducibilityChecker",
]
