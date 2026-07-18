"""
`ai/models/ml/evaluation/` — regression/classification metrics and
model-comparison reporting.

    from ai.models.ml.evaluation import compute_regression_metrics, ModelComparator
"""

from ai.models.ml.evaluation.comparator import ModelComparator, ModelResult
from ai.models.ml.evaluation.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    compute_classification_metrics,
    compute_regression_metrics,
)

__all__ = [
    "compute_regression_metrics",
    "compute_classification_metrics",
    "RegressionMetrics",
    "ClassificationMetrics",
    "ModelComparator",
    "ModelResult",
]