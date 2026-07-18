"""
Model evaluation metrics for `ai/models/ml/`.

Implements exactly the metric list the Sprint 3 spec calls out:
regression = MAE, MSE, RMSE, MAPE, R2, Adjusted R2; classification =
Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class RegressionMetrics:
    mae: float
    mse: float
    rmse: float
    mape: Optional[float]  # None if every y_true value is ~0 (undefined for that target)
    r2: float
    adjusted_r2: Optional[float]  # None if n_samples <= n_features + 1 (undefined)
    n_samples: int
    n_features: int

    def as_dict(self) -> dict:
        return {
            "mae": round(self.mae, 6),
            "mse": round(self.mse, 6),
            "rmse": round(self.rmse, 6),
            "mape_pct": round(self.mape, 4) if self.mape is not None else None,
            "r2": round(self.r2, 6),
            "adjusted_r2": round(self.adjusted_r2, 6) if self.adjusted_r2 is not None else None,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
        }


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]  # None if y_proba wasn't supplied or the split has only one class present
    confusion_matrix: list  # [[TN, FP], [FN, TP]]
    n_samples: int

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1_score": round(self.f1_score, 6),
            "roc_auc": round(self.roc_auc, 6) if self.roc_auc is not None else None,
            "confusion_matrix": self.confusion_matrix,
            "n_samples": self.n_samples,
        }


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> Optional[float]:
    """Mean Absolute Percentage Error, excluding rows where the true value is
    within `epsilon` of zero. Financial return targets (e.g.
    `future_return_5_day`) legitimately take values near zero, where a raw
    percentage-error calculation diverges toward infinity and would dominate
    (and mislead) the metric — this is a documented limitation of MAPE on
    return-scale targets, not a bug in this implementation. Prefer
    RMSE/R2 over MAPE when evaluating `future_return_*` targets for exactly
    this reason.
    """
    mask = np.abs(y_true) > epsilon
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_regression_metrics(y_true, y_pred, n_features: int) -> RegressionMetrics:
    """Compute MAE, MSE, RMSE, MAPE, R2, Adjusted R2 for one prediction set.

    Args:
        n_features: number of features the model was trained on, needed for
            the Adjusted R2 degrees-of-freedom correction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mape = _safe_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    denom = n - n_features - 1
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / denom if denom > 0 else None

    return RegressionMetrics(
        mae=mae, mse=mse, rmse=rmse, mape=mape, r2=r2, adjusted_r2=adjusted_r2,
        n_samples=n, n_features=n_features,
    )


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> ClassificationMetrics:
    """Compute Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix.

    Args:
        y_proba: positive-class probabilities (1-D), e.g.
            `model.predict_proba(X)[:, 1]`. ROC AUC is `None` if omitted or
            if `y_true` contains only one class (ROC AUC is undefined then).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_proba))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return ClassificationMetrics(
        accuracy=accuracy, precision=precision, recall=recall, f1_score=f1,
        roc_auc=roc_auc, confusion_matrix=cm, n_samples=len(y_true),
    )